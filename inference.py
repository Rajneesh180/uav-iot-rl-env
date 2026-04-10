#!/usr/bin/env python3
"""Run an LLM agent against the UAV-IoT OpenEnv environment.

Emits [START]/[STEP]/[END] structured logs for the hackathon harness.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import sys
from typing import List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Required environment variables (hackathon spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "uav_iot_env"
TASK_STEPS = {"easy": 80, "medium": 120, "hard": 160, "expert": 200}
TASKS = list(TASK_STEPS.keys())
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    err_str = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert UAV navigation agent controlling an autonomous drone.

## Mission
Navigate a 2D area with obstacles to visit Rendezvous Points (RPs) where IoT sensors have data to collect, then return safely to base before battery runs out.

## Available Actions (respond with EXACTLY one)
- move_N, move_NE, move_E, move_SE, move_S, move_SW, move_W, move_NW — move ~50m in that direction
- hover_collect — hover at current position to collect data from nearby RPs (within 60m)
- return_base — fly directly back to base (obstacle-aware routing)

## Strategy
1. Plan an efficient route to visit high-priority RPs first
2. Avoid obstacles — moves into obstacles waste energy and fail
3. Monitor battery — always keep enough to return to base
4. When close to an RP (within 60m), use hover_collect
5. When battery < 30% or all RPs visited, use return_base
6. Prefer visiting nearby unvisited RPs to minimize travel
7. On windy tasks, check wind_x/wind_y and compensate — the wind drifts your position each step

## Response Format
Respond with ONLY a JSON object: {"action": "ACTION_NAME", "reasoning": "brief explanation"}
Do NOT include any other text outside the JSON."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_observation(obs: dict) -> str:
    lines = [
        f"Step {obs.get('step_num', '?')}: {obs.get('message', '')}",
        f"Position: ({obs.get('uav_x', 0):.0f}, {obs.get('uav_y', 0):.0f})",
        f"Battery: {obs.get('battery_pct', 0):.1%}",
        f"RPs: {obs.get('rps_visited', 0)}/{obs.get('rps_total', 0)}",
        f"Data: {obs.get('data_collected', 0):.0f}/{obs.get('data_possible', 0):.0f}",
        f"Dist to base: {obs.get('dist_to_base', 0):.0f}m",
        f"Map: {obs.get('map_width', 0):.0f}x{obs.get('map_height', 0):.0f}m",
        f"Energy cost: {obs.get('energy_per_move', 850):.0f} J per 50m move",
    ]

    wx, wy = obs.get("wind_x", 0), obs.get("wind_y", 0)
    if abs(wx) > 0.1 or abs(wy) > 0.1:
        speed = (wx**2 + wy**2) ** 0.5
        lines.append(f"Wind: ({wx:+.1f}, {wy:+.1f}) m/s — {speed:.1f} m/s total")

    lines.append("")
    lines.append("Unvisited RPs:")
    sensors = obs.get("sensors", [])
    unvisited = sorted(
        [s for s in sensors if not s.get("visited")],
        key=lambda s: s.get("distance", 9999),
    )
    for s in unvisited:
        lines.append(
            f"  RP-{s['id']}: ({s['x']:.0f},{s['y']:.0f}) "
            f"pri={s['priority']} dist={s['distance']:.0f}m"
        )
    if not unvisited:
        lines.append("  (all visited)")

    obstacles = obs.get("obstacles", [])
    if obstacles:
        lines.append("\nObstacles:")
        for o in obstacles:
            lines.append(
                f"  OBS-{o['id']}: ({o['x1']:.0f},{o['y1']:.0f})-({o['x2']:.0f},{o['y2']:.0f})"
            )
    return "\n".join(lines)


def parse_action(text: str) -> str:
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "action" in data:
            return data["action"]
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^}]*"action"\s*:\s*"([^"]+)"[^}]*\}', text)
    if m:
        return m.group(1)
    valid = [
        "move_N", "move_NE", "move_E", "move_SE",
        "move_S", "move_SW", "move_W", "move_NW",
        "hover_collect", "return_base",
    ]
    for a in valid:
        if a in text:
            return a
    return "return_base"


def get_model_response(client: OpenAI, messages: list) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return '{"action": "return_base", "reasoning": "LLM error fallback"}'


# ---------------------------------------------------------------------------
# Episode runner — uses from_docker_image() or base_url connection
# ---------------------------------------------------------------------------


async def run_task(client: OpenAI, task_id: str) -> None:
    from uav_iot_env import UAVAction, UAVIoTEnv

    max_steps = TASK_STEPS.get(task_id, 120)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.01

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Connect: prefer from_docker_image if LOCAL_IMAGE_NAME is set
    if LOCAL_IMAGE_NAME:
        env = await UAVIoTEnv.from_docker_image(
            LOCAL_IMAGE_NAME, ports={8000: 8000}
        )
    else:
        env = UAVIoTEnv(base_url=os.getenv("ENV_URL", "http://localhost:8000"))
        await env.connect()

    try:
        result = await env.reset(task_id=task_id, seed=42)
        obs = result.observation.model_dump()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Episode started.\n\n{format_observation(obs)}\n\nChoose your first action."},
        ]

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Run sync OpenAI call in thread to keep WS alive
            llm_text = await asyncio.to_thread(
                get_model_response, client, messages
            )
            action_str = parse_action(llm_text)

            result = await env.step(UAVAction(action=action_str))
            obs = result.observation.model_dump()

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            messages.append({"role": "assistant", "content": llm_text})
            if not done:
                messages.append({
                    "role": "user",
                    "content": f"Reward: {reward:+.3f}\n\n{format_observation(obs)}\n\nNext action?",
                })
            # Keep context window manageable
            if len(messages) > 40:
                messages = [messages[0]] + messages[-20:]

            if done:
                break

        # Fetch the environment's composite score
        try:
            state = await env.state()
            score = state.score
        except Exception:
            score = 0.01
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in TASKS:
        await run_task(client, task_id)


if __name__ == "__main__":
    asyncio.run(async_main())
