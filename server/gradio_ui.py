"""Custom Gradio visualization tab — interactive 2D map with SVG rendering.

Provides a visual simulation interface alongside the standard OpenEnv playground.
The Simulation tab lets users see the UAV, sensors, obstacles, and flight trail
on a dark-themed map with direction-button controls and auto-play.
"""

from __future__ import annotations

import math
import time

import gradio as gr

try:
    from .simulation import (
        BASE_PROXIMITY, COLLECT_RADIUS, CRUISE_SPEED,
        OBSTACLE_ATTEMPT_FRAC, R_COLLECT_PER_PRI, R_CRASH, R_HOVER_MISS,
        R_OBSTACLE_HIT, R_PATH_BLOCKED, R_RETURN_BASE, R_RETURN_COVERAGE,
        R_STRANDED, R_TIME_PENALTY, STEP_SIZE, TASKS, WIND_PERSISTENCE,
        W_COVERAGE, W_ENERGY, W_PRIORITY, W_SAFE_RETURN,
        Coord, Node, Obstacle,
    )
    from .simulation.deployment import deploy_nodes, deploy_obstacles
    from .simulation.energy import energy_for_distance, energy_for_hover
    from .simulation.path_planning import (
        blocked, point_inside_any_obstacle, shortest_obstacle_free_path,
    )
    from .simulation.rp_selection import select_rendezvous_points
    from .simulation.wind import WindModel
except ImportError:
    from server.simulation import (
        BASE_PROXIMITY, COLLECT_RADIUS, CRUISE_SPEED,
        OBSTACLE_ATTEMPT_FRAC, R_COLLECT_PER_PRI, R_CRASH, R_HOVER_MISS,
        R_OBSTACLE_HIT, R_PATH_BLOCKED, R_RETURN_BASE, R_RETURN_COVERAGE,
        R_STRANDED, R_TIME_PENALTY, STEP_SIZE, TASKS, WIND_PERSISTENCE,
        W_COVERAGE, W_ENERGY, W_PRIORITY, W_SAFE_RETURN,
        Coord, Node, Obstacle,
    )
    from server.simulation.deployment import deploy_nodes, deploy_obstacles
    from server.simulation.energy import energy_for_distance, energy_for_hover
    from server.simulation.path_planning import (
        blocked, point_inside_any_obstacle, shortest_obstacle_free_path,
    )
    from server.simulation.rp_selection import select_rendezvous_points
    from server.simulation.wind import WindModel


# ─── Movement directions ─────────────────────────────────────────

_SQRT2_2 = math.sqrt(2) / 2
MOVE_DIRS = {
    "N": (0, -1), "NE": (_SQRT2_2, -_SQRT2_2),
    "E": (1, 0), "SE": (_SQRT2_2, _SQRT2_2),
    "S": (0, 1), "SW": (-_SQRT2_2, _SQRT2_2),
    "W": (-1, 0), "NW": (-_SQRT2_2, -_SQRT2_2),
}


# ─── State management ────────────────────────────────────────────

def _new_state() -> dict:
    return {
        "task": "", "map_w": 0, "map_h": 0,
        "nodes": [], "obstacles": [], "rps": [], "rp_members": {},
        "base": (0, 0), "uav_x": 0, "uav_y": 0,
        "battery": 0, "battery_cap": 0,
        "visited": set(), "trail": [],
        "step": 0, "max_steps": 0, "done": True,
        "total_reward": 0, "log": [],
        "wind": WindModel(0.0, 0.0),
    }


def _reset_env(task_name: str, seed: int = 42) -> dict:
    task = TASKS[task_name]
    s = _new_state()
    s["task"] = task_name
    s["map_w"], s["map_h"] = task["map_w"], task["map_h"]
    s["max_steps"] = task["max_steps"]
    s["battery_cap"] = task["battery_capacity"]
    s["battery"] = s["battery_cap"]
    s["base"] = (task["map_w"] / 2, task["map_h"] / 2)

    s["obstacles"] = deploy_obstacles(
        task["obstacle_count"], s["map_w"], s["map_h"], seed=seed,
    )
    s["nodes"] = deploy_nodes(
        task["node_count"], s["map_w"], s["map_h"],
        seed=seed, obstacles=s["obstacles"],
    )
    s["rps"], s["rp_members"] = select_rendezvous_points(
        s["nodes"], s["obstacles"], task["rp_radius"],
    )

    s["wind"] = WindModel(
        base_speed=task.get("wind_speed", 0.0),
        variability=task.get("wind_variability", 0.0),
        persistence=WIND_PERSISTENCE,
        seed=seed * 31 + 17,
    )

    s["uav_x"], s["uav_y"] = s["base"]
    s["visited"] = set()
    s["trail"] = [s["base"]]
    s["step"] = 0
    s["done"] = False
    s["total_reward"] = 0
    s["log"] = [
        f"\U0001f680 Episode started — Task: {task_name} | "
        f"RPs: {len(s['rps'])} | Battery: 100%"
    ]
    return s


def _bat_pct(s: dict) -> float:
    return max(0, s["battery"] / s["battery_cap"]) if s["battery_cap"] > 0 else 0


def _at_base(s: dict) -> bool:
    return math.hypot(s["uav_x"] - s["base"][0], s["uav_y"] - s["base"][1]) < BASE_PROXIMITY


def _data_collected(s: dict) -> float:
    return sum(s["nodes"][i]["priority"] for i in s["visited"])


def _data_possible(s: dict) -> float:
    return sum(s["nodes"][i]["priority"] for i in s["rps"])


def _compute_score(s: dict) -> float:
    if not s["rps"]:
        return 0
    coverage = len(s["visited"]) / len(s["rps"])
    dp = _data_possible(s)
    pri = _data_collected(s) / dp if dp > 0 else 0
    energy_used_frac = 1.0 - _bat_pct(s)
    eff = min(1.0, coverage / energy_used_frac) if energy_used_frac >= 0.01 else 0.0
    safe = 1.0 if s["step"] > 0 and _at_base(s) and s["battery"] > 0 else 0.0
    return W_COVERAGE * coverage + W_PRIORITY * pri + W_ENERGY * eff + W_SAFE_RETURN * safe


def _step_env(s: dict, action: str) -> dict:
    if s["done"]:
        s["log"].append("\u26a0\ufe0f  Episode over — click Reset to start a new one.")
        return s

    s["step"] += 1
    reward = 0.0
    msg = ""

    if action in MOVE_DIRS:
        dx, dy = MOVE_DIRS[action]
        nx = s["uav_x"] + dx * STEP_SIZE
        ny = s["uav_y"] + dy * STEP_SIZE
        nx = max(0, min(s["map_w"], nx))
        ny = max(0, min(s["map_h"], ny))

        if point_inside_any_obstacle(nx, ny, s["obstacles"]):
            s["battery"] -= energy_for_distance(STEP_SIZE * OBSTACLE_ATTEMPT_FRAC)
            reward, msg = R_OBSTACLE_HIT, f"\U0001f6ab Blocked! ({nx:.0f},{ny:.0f}) inside obstacle."
        elif blocked((s["uav_x"], s["uav_y"]), (nx, ny), s["obstacles"]):
            s["battery"] -= energy_for_distance(STEP_SIZE * OBSTACLE_ATTEMPT_FRAC)
            reward, msg = R_PATH_BLOCKED, "\U0001f6ab Path blocked by obstacle."
        else:
            dist = math.hypot(nx - s["uav_x"], ny - s["uav_y"])
            s["battery"] -= energy_for_distance(dist)
            # Wind drift
            wx, wy = s["wind"].step()
            drift_time = STEP_SIZE / CRUISE_SPEED
            fx = max(0, min(s["map_w"], nx + wx * drift_time))
            fy = max(0, min(s["map_h"], ny + wy * drift_time))
            if point_inside_any_obstacle(fx, fy, s["obstacles"]):
                fx, fy = nx, ny
            s["uav_x"], s["uav_y"] = fx, fy
            s["trail"].append((fx, fy))
            wind_note = f" \U0001f4a8{s['wind'].speed:.1f}m/s" if s["wind"].speed > 0.5 else ""
            reward, msg = 0.0, f"\u27a1\ufe0f  Moved {action} \u2192 ({fx:.0f},{fy:.0f}){wind_note}"

    elif action == "hover_collect":
        s["battery"] -= energy_for_hover()
        collected = []
        for rp in s["rps"]:
            if rp in s["visited"]:
                continue
            n = s["nodes"][rp]
            if math.hypot(s["uav_x"] - n["x"], s["uav_y"] - n["y"]) <= COLLECT_RADIUS:
                s["visited"].add(rp)
                collected.append(rp)
        if collected:
            reward = sum(s["nodes"][i]["priority"] * R_COLLECT_PER_PRI for i in collected)
            names = ", ".join(
                f"RP-{i}(p={s['nodes'][i]['priority']})" for i in collected
            )
            msg = f"\U0001f4e1 Collected: {names}"
        else:
            reward, msg = R_HOVER_MISS, "\U0001f4e1 No RPs in range."

    elif action == "return_base":
        path, dist = shortest_obstacle_free_path(
            (s["uav_x"], s["uav_y"]), s["base"], s["obstacles"],
        )
        energy = energy_for_distance(dist)
        if energy > s["battery"]:
            s["battery"] = 0
            reward, msg = R_STRANDED, "\U0001f50b Not enough battery to reach base!"
        else:
            s["battery"] -= energy
            s["uav_x"], s["uav_y"] = s["base"]
            s["trail"].extend(path[1:])
            s["done"] = True
            cov = len(s["visited"]) / max(1, len(s["rps"]))
            reward = R_RETURN_COVERAGE * cov + R_RETURN_BASE
            msg = (
                f"\U0001f3e0 Returned to base! "
                f"{len(s['visited'])}/{len(s['rps'])} RPs collected."
            )

    reward += R_TIME_PENALTY
    if s["battery"] <= 0:
        s["done"] = True
        reward += R_CRASH
        msg += " \U0001f480 Battery depleted!"
    elif s["step"] >= s["max_steps"]:
        s["done"] = True
        msg += " \u23f0 Max steps reached."

    s["total_reward"] += reward
    s["log"].append(f"[{s['step']}] {msg} (r={reward:+.3f})")
    return s


# ─── SVG Rendering ───────────────────────────────────────────────

def _color_for_priority(pri: int) -> str:
    t = (pri - 1) / 9.0
    r = int(255 * min(1, 2 * t))
    g = int(255 * min(1, 2 * (1 - t)))
    return f"rgb({r},{g},40)"


def _render_svg(s: dict) -> str:
    if not s["map_w"]:
        return (
            '<div style="text-align:center;padding:100px;color:#888;'
            'font-size:1.3em;">Select a task and click <b>Reset</b> '
            "to begin.</div>"
        )

    W, H = s["map_w"], s["map_h"]
    PAD = 30
    VW, VH = W + 2 * PAD, H + 2 * PAD

    parts: list[str] = [
        f'<svg viewBox="0 0 {VW} {VH}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-height:580px;background:#0a0e17;'
        f'border-radius:12px;box-shadow:0 4px 24px rgba(0,0,0,0.4);">'
    ]

    # Background grid
    parts.append(
        f'<rect x="{PAD}" y="{PAD}" width="{W}" height="{H}" '
        f'fill="#111827" stroke="#1e293b" stroke-width="1" rx="4"/>'
    )
    for gx in range(0, W + 1, 50):
        parts.append(
            f'<line x1="{PAD+gx}" y1="{PAD}" x2="{PAD+gx}" y2="{PAD+H}" '
            f'stroke="#1e293b" stroke-width="0.5"/>'
        )
    for gy in range(0, H + 1, 50):
        parts.append(
            f'<line x1="{PAD}" y1="{PAD+gy}" x2="{PAD+W}" y2="{PAD+gy}" '
            f'stroke="#1e293b" stroke-width="0.5"/>'
        )

    # Obstacles with animated opacity
    for obs in s["obstacles"]:
        ox, oy = PAD + obs["x1"], PAD + obs["y1"]
        ow, oh = obs["x2"] - obs["x1"], obs["y2"] - obs["y1"]
        parts.append(
            f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" '
            f'fill="#991b1b" fill-opacity="0.35" stroke="#ef4444" '
            f'stroke-width="1.5" rx="3">'
            f'<animate attributeName="fill-opacity" '
            f'values="0.25;0.4;0.25" dur="3s" repeatCount="indefinite"/>'
            f"</rect>"
            f'<text x="{ox+ow/2}" y="{oy+oh/2}" text-anchor="middle" '
            f'dominant-baseline="central" fill="#fca5a5" font-size="10" '
            f'font-family="monospace">OBS-{obs["id"]}</text>'
        )

    # Collection radius rings for unvisited RPs
    for rp_idx in s["rps"]:
        if rp_idx in s["visited"]:
            continue
        n = s["nodes"][rp_idx]
        cx, cy = PAD + n["x"], PAD + n["y"]
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{COLLECT_RADIUS}" '
            f'fill="none" stroke="#38bdf8" stroke-width="0.8" '
            f'stroke-dasharray="4,4" opacity="0.3"/>'
        )

    # Flight trail
    if len(s["trail"]) > 1:
        pts = " ".join(f"{PAD+x},{PAD+y}" for x, y in s["trail"])
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#38bdf8" '
            f'stroke-width="1.5" stroke-dasharray="6,3" opacity="0.6"/>'
        )
        for i, (tx, ty) in enumerate(s["trail"][:-1]):
            op = max(0.15, 0.6 * (i / max(1, len(s["trail"]))))
            parts.append(
                f'<circle cx="{PAD+tx}" cy="{PAD+ty}" r="2" '
                f'fill="#38bdf8" opacity="{op:.2f}"/>'
            )

    # Non-RP sensor nodes (grey)
    rp_set = set(s["rps"])
    for n in s["nodes"]:
        if n["id"] in rp_set:
            continue
        cx, cy = PAD + n["x"], PAD + n["y"]
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="5" fill="#475569" '
            f'fill-opacity="0.5" stroke="#64748b" stroke-width="0.5"/>'
        )

    # RP nodes (color-coded by priority, green when visited)
    for rp_idx in s["rps"]:
        n = s["nodes"][rp_idx]
        cx, cy = PAD + n["x"], PAD + n["y"]
        visited = rp_idx in s["visited"]
        color = "#22c55e" if visited else _color_for_priority(n["priority"])
        ring = "#86efac" if visited else "#fbbf24"
        r = 7 if visited else 9

        # Pulsing ring
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r+4}" fill="none" '
            f'stroke="{ring}" stroke-width="1.5" opacity="0.5">'
            f'<animate attributeName="r" values="{r+2};{r+6};{r+2}" '
            f'dur="2s" repeatCount="indefinite"/></circle>'
        )
        # Filled circle
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" '
            f'stroke="white" stroke-width="1.5"/>'
        )
        label = "\u2713" if visited else str(n["priority"])
        parts.append(
            f'<text x="{cx}" y="{cy+1}" text-anchor="middle" '
            f'dominant-baseline="central" fill="white" font-size="9" '
            f'font-weight="bold" font-family="monospace">{label}</text>'
        )

    # Base station diamond
    bx, by = PAD + s["base"][0], PAD + s["base"][1]
    parts.append(
        f'<polygon points="{bx},{by-14} {bx+12},{by} {bx},{by+14} {bx-12},{by}" '
        f'fill="#7c3aed" stroke="#c4b5fd" stroke-width="2">'
        f'<animate attributeName="fill" values="#7c3aed;#8b5cf6;#7c3aed" '
        f'dur="2s" repeatCount="indefinite"/></polygon>'
        f'<text x="{bx}" y="{by+1}" text-anchor="middle" '
        f'dominant-baseline="central" fill="white" font-size="8" '
        f'font-weight="bold" font-family="monospace">BS</text>'
    )

    # UAV drone icon
    ux, uy = PAD + s["uav_x"], PAD + s["uav_y"]
    parts.append(
        f'<circle cx="{ux}" cy="{uy}" r="18" fill="#06b6d4" opacity="0.15">'
        f'<animate attributeName="r" values="14;22;14" dur="1.5s" '
        f'repeatCount="indefinite"/></circle>'
    )
    parts.append(
        f'<circle cx="{ux}" cy="{uy}" r="8" fill="#06b6d4" '
        f'stroke="white" stroke-width="2"/>'
    )
    for angle in [45, 135, 225, 315]:
        rad = math.radians(angle)
        ex, ey = ux + 12 * math.cos(rad), uy + 12 * math.sin(rad)
        parts.append(
            f'<line x1="{ux}" y1="{uy}" x2="{ex}" y2="{ey}" '
            f'stroke="white" stroke-width="1.5" opacity="0.7"/>'
            f'<circle cx="{ex}" cy="{ey}" r="3" fill="#0ea5e9" '
            f'stroke="white" stroke-width="0.5"/>'
        )
    parts.append(
        f'<text x="{ux}" y="{uy-20}" text-anchor="middle" fill="#67e8f9" '
        f'font-size="9" font-weight="bold" font-family="monospace">UAV</text>'
    )

    # Legend + counters
    ly = VH - 18
    parts.append(
        f'<text x="{PAD}" y="{ly}" fill="#94a3b8" font-size="9" '
        f'font-family="monospace">'
        f"\u2b24 Sensor  \u2b24 RP(priority)  \u25c6 Base  "
        f"\u2726 UAV  \u2588 Obstacle  - - Trail</text>"
    )
    parts.append(
        f'<text x="{VW-PAD}" y="{PAD-8}" text-anchor="end" fill="#64748b" '
        f'font-size="11" font-family="monospace">'
        f'Step {s["step"]}/{s["max_steps"]}</text>'
    )
    parts.append(
        f'<text x="{PAD}" y="{PAD-8}" fill="#64748b" font-size="11" '
        f'font-family="monospace">Task: {s["task"].upper()}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ─── Stats panel ─────────────────────────────────────────────────

def _render_stats(s: dict) -> str:
    if not s["map_w"]:
        return ""

    bp = _bat_pct(s)
    bp100 = bp * 100
    bat_color = "#22c55e" if bp > 0.5 else ("#eab308" if bp > 0.2 else "#ef4444")
    dc = _data_collected(s)
    dp = _data_possible(s)
    cov = len(s["visited"]) / max(1, len(s["rps"])) * 100
    score = _compute_score(s) * 100

    if not s["done"]:
        status_icon, status_text = "\U0001f7e2", "Active"
    elif _at_base(s) and s["battery"] > 0:
        status_icon, status_text = "\U0001f3c6", "Mission Complete"
    else:
        status_icon, status_text = "\U0001f534", "Mission Failed"

    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;color:#e2e8f0;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;
                  margin-bottom:16px;">
        <div style="background:#1e293b;border-radius:10px;padding:14px;
                    border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;
                      letter-spacing:1px;">Status</div>
          <div style="font-size:22px;font-weight:700;margin-top:4px;">
            {status_icon} {status_text}</div>
        </div>
        <div style="background:#1e293b;border-radius:10px;padding:14px;
                    border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;
                      letter-spacing:1px;">Score</div>
          <div style="font-size:28px;font-weight:700;color:#38bdf8;
                      margin-top:2px;">{score:.1f}<span style="font-size:14px;
                      color:#64748b;">%</span></div>
        </div>
      </div>

      <div style="background:#1e293b;border-radius:10px;padding:14px;
                  margin-bottom:12px;border:1px solid #334155;">
        <div style="display:flex;justify-content:space-between;
                    align-items:center;margin-bottom:6px;">
          <span style="color:#94a3b8;font-size:11px;text-transform:uppercase;
                       letter-spacing:1px;">Battery</span>
          <span style="color:{bat_color};font-weight:700;
                       font-size:14px;">{bp100:.1f}%</span>
        </div>
        <div style="background:#0f172a;border-radius:6px;height:12px;
                    overflow:hidden;">
          <div style="background:linear-gradient(90deg,{bat_color},
                      {bat_color}88);height:100%;width:{bp100:.1f}%;
                      border-radius:6px;transition:width 0.3s;"></div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;
                  margin-bottom:12px;">
        <div style="background:#1e293b;border-radius:10px;padding:12px;
                    text-align:center;border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:10px;
                      text-transform:uppercase;">Steps</div>
          <div style="font-size:20px;font-weight:700;color:#e2e8f0;">
            {s['step']}<span style="font-size:12px;
            color:#64748b;">/{s['max_steps']}</span></div>
        </div>
        <div style="background:#1e293b;border-radius:10px;padding:12px;
                    text-align:center;border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:10px;
                      text-transform:uppercase;">RPs</div>
          <div style="font-size:20px;font-weight:700;color:#22c55e;">
            {len(s['visited'])}<span style="font-size:12px;
            color:#64748b;">/{len(s['rps'])}</span></div>
        </div>
        <div style="background:#1e293b;border-radius:10px;padding:12px;
                    text-align:center;border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:10px;
                      text-transform:uppercase;">Coverage</div>
          <div style="font-size:20px;font-weight:700;
                      color:#fbbf24;">{cov:.0f}%</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;
                  margin-bottom:12px;">
        <div style="background:#1e293b;border-radius:10px;padding:12px;
                    text-align:center;border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:10px;
                      text-transform:uppercase;">Data Collected</div>
          <div style="font-size:16px;font-weight:700;">{dc:.0f}<span
            style="font-size:11px;color:#64748b;">/{dp:.0f} pts</span></div>
        </div>
        <div style="background:#1e293b;border-radius:10px;padding:12px;
                    text-align:center;border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:10px;
                      text-transform:uppercase;">UAV Position</div>
          <div style="font-size:16px;font-weight:700;">
            ({s['uav_x']:.0f}, {s['uav_y']:.0f})</div>
        </div>
      </div>

      <div style="background:#1e293b;border-radius:10px;padding:14px;
                  border:1px solid #334155;">
        <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:8px;">Mission Info</div>
        <div style="font-size:12px;line-height:1.6;color:#cbd5e1;">
          <b>Task:</b> {s['task'].upper()} &nbsp;|&nbsp;
          <b>Map:</b> {s['map_w']}\u00d7{s['map_h']}m &nbsp;|&nbsp;
          <b>Obstacles:</b> {len(s['obstacles'])} &nbsp;|&nbsp;
          <b>Sensors:</b> {len(s['nodes'])} &nbsp;|&nbsp;
          <b>Wind:</b> {s['wind'].speed:.1f} m/s &nbsp;|&nbsp;
          <b>Reward:</b> {s['total_reward']:+.3f}
        </div>
      </div>
    </div>"""


# ─── Mission log ─────────────────────────────────────────────────

def _render_log(s: dict) -> str:
    if not s["log"]:
        return ""
    lines = s["log"][-20:]
    items = "".join(
        f'<div style="padding:4px 8px;border-bottom:1px solid #1e293b;'
        f'font-size:12px;color:#94a3b8;font-family:monospace;">{line}</div>'
        for line in reversed(lines)
    )
    return (
        f'<div style="background:#0f172a;border-radius:10px;'
        f'border:1px solid #1e293b;max-height:300px;overflow-y:auto;">'
        f"{items}</div>"
    )


# ─── Gradio callbacks ────────────────────────────────────────────

def _do_reset(task_name, seed, _state):
    s = _reset_env(task_name, int(seed))
    return s, _render_svg(s), _render_stats(s), _render_log(s)


def _do_step(action, state):
    s = _step_env(state, action)
    return s, _render_svg(s), _render_stats(s), _render_log(s)


def _do_auto_play(task_name, seed, _state):
    s = _reset_env(task_name, int(seed))
    yield s, _render_svg(s), _render_stats(s), _render_log(s)
    time.sleep(0.3)

    _stuck_count = 0
    _last_pos = (s["uav_x"], s["uav_y"])

    while not s["done"]:
        unvisited = [i for i in s["rps"] if i not in s["visited"]]
        if not unvisited:
            s = _step_env(s, "return_base")
            yield s, _render_svg(s), _render_stats(s), _render_log(s)
            break

        # Check if any RP is in collection range
        can_collect = any(
            math.hypot(s["uav_x"] - s["nodes"][rp]["x"],
                       s["uav_y"] - s["nodes"][rp]["y"]) <= COLLECT_RADIUS
            for rp in unvisited
        )
        if can_collect:
            s = _step_env(s, "hover_collect")
        else:
            # Sort unvisited RPs by distance; skip stuck targets
            sorted_rps = sorted(
                unvisited,
                key=lambda i: math.hypot(
                    s["uav_x"] - s["nodes"][i]["x"],
                    s["uav_y"] - s["nodes"][i]["y"],
                ),
            )
            # If stuck on same position for 5+ steps, try next-closest RP
            rp_idx = min(_stuck_count // 5, len(sorted_rps) - 1)
            target = sorted_rps[rp_idx]
            n = s["nodes"][target]
            angle_deg = math.degrees(
                math.atan2(n["y"] - s["uav_y"], n["x"] - s["uav_x"])
            )
            compass = {
                "E": 0, "SE": 45, "S": 90, "SW": 135,
                "W": 180, "NW": -135, "N": -90, "NE": -45,
            }
            # Rank directions by angle proximity; try alternatives if blocked
            ranked = sorted(
                compass,
                key=lambda d: abs(((compass[d] - angle_deg + 180) % 360) - 180),
            )
            chosen = ranked[0]
            for d in ranked:
                dx, dy = MOVE_DIRS[d]
                nx = s["uav_x"] + dx * STEP_SIZE
                ny = s["uav_y"] + dy * STEP_SIZE
                nx = max(0, min(s["map_w"], nx))
                ny = max(0, min(s["map_h"], ny))
                if not point_inside_any_obstacle(nx, ny, s["obstacles"]) \
                   and not blocked((s["uav_x"], s["uav_y"]), (nx, ny), s["obstacles"]):
                    chosen = d
                    break
            s = _step_env(s, chosen)

        cur_pos = (s["uav_x"], s["uav_y"])
        if cur_pos == _last_pos:
            _stuck_count += 1
        else:
            _stuck_count = 0
        _last_pos = cur_pos

        # If hopelessly stuck, return to base early
        if _stuck_count >= 15:
            s = _step_env(s, "return_base")
            yield s, _render_svg(s), _render_stats(s), _render_log(s)
            break

        yield s, _render_svg(s), _render_stats(s), _render_log(s)
        time.sleep(0.15)

    if not s["done"]:
        s = _step_env(s, "return_base")
        yield s, _render_svg(s), _render_stats(s), _render_log(s)


# ─── Builder (OpenEnv gradio_builder interface) ───────────────────

_CSS = """
.sim-btn{min-height:52px!important;font-size:15px!important;font-weight:600!important}
.sim-btn:hover{transform:scale(1.03);transition:transform .15s}
.dir-grid{gap:4px!important}
"""


def build_uav_gradio_app(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title,
    quick_start_md,
):
    """Return a gr.Blocks with the visual simulation interface.

    Appears as the **Simulation** tab alongside the default Playground tab.
    """
    empty = _new_state()

    with gr.Blocks(title=f"{title} \u2014 Simulation") as blocks:
        state = gr.State(empty)

        gr.HTML(f"<style>{_CSS}</style>")
        gr.HTML(
            '<div style="text-align:center;padding:12px 0 4px;">'
            '<h2 style="margin:0;font-size:22px;font-weight:800;'
            "background:linear-gradient(135deg,#06b6d4,#8b5cf6);"
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
            "UAV IoT Simulation"
            "</h2>"
            '<p style="color:#64748b;margin:4px 0 0;font-size:13px;">'
            "Interactive map \u2014 fly the drone, collect sensor data, "
            "return to base</p></div>"
        )

        with gr.Row():
            # ── Left: Map + Log ──
            with gr.Column(scale=3):
                svg_out = gr.HTML(_render_svg(empty))
                log_out = gr.HTML(_render_log(empty), label="Mission Log")

            # ── Right: Stats + Controls ──
            with gr.Column(scale=1, min_width=320):
                stats_out = gr.HTML(_render_stats(empty))

                gr.HTML(
                    '<div style="color:#94a3b8;font-size:11px;'
                    "text-transform:uppercase;letter-spacing:1px;"
                    'margin:12px 0 4px;font-weight:600;">Configuration</div>'
                )
                with gr.Row():
                    task_dd = gr.Dropdown(
                        choices=["easy", "medium", "hard", "expert"],
                        value="medium", label="Task", scale=2,
                    )
                    seed_num = gr.Number(
                        value=42, label="Seed", scale=1, precision=0,
                    )
                with gr.Row():
                    reset_btn = gr.Button(
                        "\U0001f504 Reset / Start",
                        variant="primary", elem_classes=["sim-btn"],
                    )
                    auto_btn = gr.Button(
                        "\u25b6 Auto Play",
                        variant="secondary", elem_classes=["sim-btn"],
                    )

                gr.HTML(
                    '<div style="color:#94a3b8;font-size:11px;'
                    "text-transform:uppercase;letter-spacing:1px;"
                    'margin:16px 0 6px;font-weight:600;">Movement</div>'
                )

                # 3x3 direction grid
                with gr.Row(elem_classes=["dir-grid"]):
                    nw = gr.Button("\u2196 NW", size="sm", elem_classes=["sim-btn"])
                    n_ = gr.Button(
                        "\u2b06 N", size="sm",
                        variant="primary", elem_classes=["sim-btn"],
                    )
                    ne = gr.Button("\u2197 NE", size="sm", elem_classes=["sim-btn"])
                with gr.Row(elem_classes=["dir-grid"]):
                    w_ = gr.Button(
                        "\u2b05 W", size="sm",
                        variant="primary", elem_classes=["sim-btn"],
                    )
                    col = gr.Button(
                        "\U0001f4e1 Collect", size="sm",
                        variant="stop", elem_classes=["sim-btn"],
                    )
                    e_ = gr.Button(
                        "\u27a1 E", size="sm",
                        variant="primary", elem_classes=["sim-btn"],
                    )
                with gr.Row(elem_classes=["dir-grid"]):
                    sw = gr.Button("\u2199 SW", size="sm", elem_classes=["sim-btn"])
                    s_ = gr.Button(
                        "\u2b07 S", size="sm",
                        variant="primary", elem_classes=["sim-btn"],
                    )
                    se = gr.Button("\u2198 SE", size="sm", elem_classes=["sim-btn"])

                gr.HTML('<div style="height:8px;"></div>')
                home_btn = gr.Button(
                    "\U0001f3e0 Return to Base",
                    variant="huggingface", elem_classes=["sim-btn"],
                )

        # ── Wire events ──
        outs = [state, svg_out, stats_out, log_out]

        reset_btn.click(_do_reset, [task_dd, seed_num, state], outs)
        auto_btn.click(_do_auto_play, [task_dd, seed_num, state], outs)

        for btn, direction in [
            (n_, "N"), (ne, "NE"), (e_, "E"), (se, "SE"),
            (s_, "S"), (sw, "SW"), (w_, "W"), (nw, "NW"),
        ]:
            btn.click(_do_step, [gr.State(direction), state], outs)

        col.click(_do_step, [gr.State("hover_collect"), state], outs)
        home_btn.click(_do_step, [gr.State("return_base"), state], outs)

    return blocks
