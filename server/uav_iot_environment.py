"""OpenEnv server — UAV navigates obstacles, collects IoT data, returns to base."""

from __future__ import annotations

import math
import uuid
from difflib import get_close_matches
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ObstacleInfo, SensorInfo, UAVAction, UAVObservation, UAVState,
    )
    from .simulation import (
        BASE_PROXIMITY, COLLECT_RADIUS, CRUISE_SPEED, E_FLY_PER_M,
        HOVER_TIME, OBSTACLE_ATTEMPT_FRAC, P_HOVER, R_COLLECT_PER_PRI,
        R_CRASH, R_HOVER_MISS, R_OBSTACLE_HIT, R_PATH_BLOCKED,
        R_RETURN_BASE, R_RETURN_COVERAGE, R_STRANDED, R_TIME_PENALTY,
        R_UNKNOWN_ACTION, STEP_SIZE, TASKS, WIND_PERSISTENCE,
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
    from models import (
        ObstacleInfo, SensorInfo, UAVAction, UAVObservation, UAVState,
    )
    from server.simulation import (
        BASE_PROXIMITY, COLLECT_RADIUS, CRUISE_SPEED, E_FLY_PER_M,
        HOVER_TIME, OBSTACLE_ATTEMPT_FRAC, P_HOVER, R_COLLECT_PER_PRI,
        R_CRASH, R_HOVER_MISS, R_OBSTACLE_HIT, R_PATH_BLOCKED,
        R_RETURN_BASE, R_RETURN_COVERAGE, R_STRANDED, R_TIME_PENALTY,
        R_UNKNOWN_ACTION, STEP_SIZE, TASKS, WIND_PERSISTENCE,
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

# 8-direction movement deltas (unit vectors scaled by STEP_SIZE)
_SQRT2_2 = math.sqrt(2) / 2
MOVE_DIRS: dict[str, tuple[float, float]] = {
    "move_N":  (0, -1),
    "move_NE": (_SQRT2_2, -_SQRT2_2),
    "move_E":  (1, 0),
    "move_SE": (_SQRT2_2, _SQRT2_2),
    "move_S":  (0, 1),
    "move_SW": (-_SQRT2_2, _SQRT2_2),
    "move_W":  (-1, 0),
    "move_NW": (-_SQRT2_2, -_SQRT2_2),
}
ALL_ACTIONS = list(MOVE_DIRS) + ["hover_collect", "return_base"]

# Short instructions shown in the playground instead of the full README.
PLAYGROUND_INSTRUCTIONS = """\
# UAV IoT Data Collection

Control a UAV to collect data from IoT sensor clusters and return to base.

## How to Play

1. **Reset** the environment (picks task: easy / medium / hard / expert)
2. **Move** toward sensor clusters using directional actions
3. **Hover** near a cluster to collect its data
4. **Return to base** before your battery runs out

## Actions

| Action | What it does |
|--------|-------------|
| `move_N` `move_NE` `move_E` ... | Fly ~50 m in that direction (costs 850 J) |
| `hover_collect` | Hover for 10 s, collects data from RPs within 60 m (costs 1685 J) |
| `return_base` | Autopilot back to base (path avoids obstacles) |

## Wind

Medium, hard, and expert tasks have stochastic wind that drifts the UAV off \
course each step. Wind intensity increases with difficulty. Check `wind_x` \
and `wind_y` in the observation to anticipate drift and adjust your heading.

## Tips

- Check `sensors` in the observation for RP locations and distances
- Prioritise high-priority RPs (higher `priority` = more score)
- Watch `battery_pct` — if it hits 0 the UAV crashes (−1.0 reward)
- Use `return_base` when you've collected enough or battery is low
- `hover_collect` only works within 60 m of an unvisited RP
- Obstacles block movement; the UAV cannot fly through them
- On windy tasks, compensate for drift when navigating tight corridors

## Scoring

```
score = 0.35 × coverage + 0.25 × priority_ratio
      + 0.25 × energy_efficiency + 0.15 × safe_return
```

Energy efficiency = coverage / energy_fraction_used (rewarding more collection \
per unit energy spent, not idle conservation).
"""


class UAVIoTEnvironment(Environment[UAVAction, UAVObservation, UAVState]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, default_task: str = "medium", **kwargs: Any):
        super().__init__(**kwargs)
        self._default_task = default_task

        # Episode state (set in reset)
        self._task_name: str = ""
        self._episode_id: str = ""
        self._step_count: int = 0
        self._max_steps: int = 0
        self._done: bool = True

        # Map
        self._map_w: int = 0
        self._map_h: int = 0
        self._nodes: list[Node] = []
        self._obstacles: list[Obstacle] = []
        self._rps: list[int] = []
        self._rp_members: dict[int, list[int]] = {}
        self._base: Coord = (0.0, 0.0)

        # UAV
        self._uav_x: float = 0.0
        self._uav_y: float = 0.0
        self._battery: float = 0.0
        self._battery_cap: float = 0.0
        self._visited_rps: set[int] = set()
        self._total_reward: float = 0.0

        # Wind
        self._wind: WindModel = WindModel(0.0, 0.0)

    # ------------------------------------------------------------------
    #  OpenEnv interface
    # ------------------------------------------------------------------

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="UAVIoTEnvironment",
            description="UAV navigates obstacles, collects IoT sensor data, returns to base.",
            version="1.0.0",
            readme_content=PLAYGROUND_INSTRUCTIONS,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> UAVObservation:
        task_name = kwargs.get("task_id", self._default_task)
        if task_name not in TASKS:
            task_name = self._default_task
        task = TASKS[task_name]

        self._task_name = task_name
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False

        # Configure map from task
        self._map_w = task["map_w"]
        self._map_h = task["map_h"]
        self._max_steps = task["max_steps"]
        self._battery_cap = task["battery_capacity"]
        self._battery = self._battery_cap

        # Base station at centre
        self._base = (self._map_w / 2, self._map_h / 2)

        # Deploy obstacles first, then sensors (avoiding obstacle interiors)
        effective_seed = seed if seed is not None else 42
        self._obstacles = deploy_obstacles(
            task["obstacle_count"], self._map_w, self._map_h, seed=effective_seed,
        )
        self._nodes = deploy_nodes(
            task["node_count"], self._map_w, self._map_h,
            seed=effective_seed, obstacles=self._obstacles,
        )

        # Select rendezvous points
        self._rps, self._rp_members = select_rendezvous_points(
            self._nodes, self._obstacles, task["rp_radius"],
        )

        # Wind model — seeded independently from deployment
        self._wind = WindModel(
            base_speed=task.get("wind_speed", 0.0),
            variability=task.get("wind_variability", 0.0),
            persistence=WIND_PERSISTENCE,
            seed=effective_seed * 31 + 17,
        )

        # UAV starts at base
        self._uav_x, self._uav_y = self._base
        self._visited_rps = set()
        self._total_reward = 0.0

        return self._make_observation(
            reward=0.0,
            message=self._build_reset_message(task_name),
        )

    def step(
        self,
        action: UAVAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> UAVObservation:
        if self._done:
            return self._make_observation(
                reward=0.0,
                message="Episode is over. Call reset() to start a new one.",
            )

        self._step_count += 1
        act = action.action.strip()
        reward = 0.0
        msg = ""

        if act in MOVE_DIRS:
            reward, msg = self._do_move(act)
        elif act == "hover_collect":
            reward, msg = self._do_hover_collect()
        elif act == "return_base":
            reward, msg = self._do_return_base()
        else:
            reward, msg = self._handle_unknown_action(act)

        reward += R_TIME_PENALTY

        if self._battery <= 0:
            self._done = True
            reward += R_CRASH
            msg += " Battery depleted! UAV crashed."
        elif self._step_count >= self._max_steps:
            self._done = True
            msg += " Max steps reached."

        self._total_reward += reward
        msg += self._build_hint()
        return self._make_observation(reward=reward, message=msg)

    @property
    def state(self) -> UAVState:
        return UAVState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            rps_visited=len(self._visited_rps),
            rps_total=len(self._rps),
            battery_pct=self._battery_pct(),
            data_collected=self._data_collected(),
            data_possible=self._data_possible(),
            wind_speed=round(self._wind.speed, 2),
            score=self._compute_score(),
        )

    # ------------------------------------------------------------------
    #  Actions
    # ------------------------------------------------------------------

    def _do_move(self, direction: str) -> tuple[float, str]:
        dx, dy = MOVE_DIRS[direction]
        new_x = self._uav_x + dx * STEP_SIZE
        new_y = self._uav_y + dy * STEP_SIZE

        # Clamp to map bounds
        new_x = max(0.0, min(float(self._map_w), new_x))
        new_y = max(0.0, min(float(self._map_h), new_y))

        if point_inside_any_obstacle(new_x, new_y, self._obstacles):
            self._battery -= energy_for_distance(STEP_SIZE * OBSTACLE_ATTEMPT_FRAC)
            return R_OBSTACLE_HIT, f"Blocked! Position ({new_x:.0f},{new_y:.0f}) is inside an obstacle."

        if blocked((self._uav_x, self._uav_y), (new_x, new_y), self._obstacles):
            self._battery -= energy_for_distance(STEP_SIZE * OBSTACLE_ATTEMPT_FRAC)
            return R_PATH_BLOCKED, f"Path to ({new_x:.0f},{new_y:.0f}) blocked by an obstacle."

        dist = math.hypot(new_x - self._uav_x, new_y - self._uav_y)
        self._battery -= energy_for_distance(dist)

        # Apply wind drift: wind_vector * (time for this move)
        wx, wy = self._wind.step()
        drift_time = STEP_SIZE / CRUISE_SPEED
        final_x = max(0.0, min(float(self._map_w), new_x + wx * drift_time))
        final_y = max(0.0, min(float(self._map_h), new_y + wy * drift_time))

        # If wind pushes into obstacle, land at intended position instead
        if point_inside_any_obstacle(final_x, final_y, self._obstacles):
            final_x, final_y = new_x, new_y

        self._uav_x = final_x
        self._uav_y = final_y
        wind_note = f" (wind drift {self._wind.speed:.1f} m/s)" if self._wind.speed > 0.5 else ""
        return 0.0, f"Moved {direction.replace('move_', '')} to ({final_x:.0f},{final_y:.0f}){wind_note}. Battery: {self._battery_pct():.0%}."

    def _do_hover_collect(self) -> tuple[float, str]:
        self._battery -= energy_for_hover()
        collected_ids: list[int] = []

        for rp_idx in self._rps:
            if rp_idx in self._visited_rps:
                continue
            rp_node = self._nodes[rp_idx]
            dist = math.hypot(self._uav_x - rp_node["x"], self._uav_y - rp_node["y"])
            if dist <= COLLECT_RADIUS:
                self._visited_rps.add(rp_idx)
                collected_ids.append(rp_idx)

        if collected_ids:
            priorities = [self._nodes[i]["priority"] for i in collected_ids]
            reward = sum(p * R_COLLECT_PER_PRI for p in priorities)
            names = ", ".join(f"RP-{i}(pri={self._nodes[i]['priority']})" for i in collected_ids)
            remaining = len(self._rps) - len(self._visited_rps)
            return reward, (
                f"Collected data from {names}. "
                f"Progress: {len(self._visited_rps)}/{len(self._rps)} RPs"
                f"{' — all collected!' if remaining == 0 else f' ({remaining} remaining).'}"
            )

        # Find nearest unvisited RP to help user
        nearest = self._nearest_rp()
        if nearest:
            rp_idx, dist = nearest
            return R_HOVER_MISS, (
                f"No RPs within {COLLECT_RADIUS:.0f} m range. "
                f"Nearest unvisited: RP-{rp_idx} at {dist:.0f} m."
            )
        return R_HOVER_MISS, "No uncollected RPs within range."

    def _do_return_base(self) -> tuple[float, str]:
        """Fly back to base using obstacle-aware path."""
        path, dist = shortest_obstacle_free_path(
            (self._uav_x, self._uav_y), self._base, self._obstacles,
        )
        energy = energy_for_distance(dist)

        if energy > self._battery:
            fraction = self._battery / energy if energy > 0 else 0
            partial_idx = max(0, int(len(path) * fraction) - 1)
            if partial_idx > 0 and partial_idx < len(path):
                self._uav_x, self._uav_y = path[partial_idx]
            self._battery = 0.0
            return R_STRANDED, "Insufficient battery to reach base. UAV stranded."

        self._battery -= energy
        self._uav_x, self._uav_y = self._base
        self._done = True

        coverage = len(self._visited_rps) / max(1, len(self._rps))
        reward = R_RETURN_COVERAGE * coverage + R_RETURN_BASE
        return reward, (
            f"Returned to base. Collected {len(self._visited_rps)}/{len(self._rps)} RPs. "
            f"Battery remaining: {self._battery_pct():.0%}."
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _battery_pct(self) -> float:
        if self._battery_cap <= 0:
            return 0.0
        return max(0.0, self._battery / self._battery_cap)

    def _data_collected(self) -> float:
        return sum(self._nodes[i]["priority"] for i in self._visited_rps)

    def _data_possible(self) -> float:
        return sum(self._nodes[i]["priority"] for i in self._rps)

    def _compute_score(self) -> float:
        """Composite score in the open interval (0, 1) for grading."""
        if not self._rps:
            return 0.01

        coverage = len(self._visited_rps) / len(self._rps)
        dp = self._data_possible()
        priority_score = self._data_collected() / dp if dp > 0 else 0.0

        # Energy efficiency: coverage per unit energy spent (not idle conservation)
        energy_used_frac = 1.0 - self._battery_pct()
        if energy_used_frac < 0.01:
            energy_eff = 0.0
        else:
            energy_eff = min(1.0, coverage / energy_used_frac)

        # Safe return only counts if the UAV actually traveled
        safe_return = (
            1.0 if self._step_count > 0 and self._at_base() and self._battery > 0
            else 0.0
        )

        raw = (
            W_COVERAGE * coverage
            + W_PRIORITY * priority_score
            + W_ENERGY * energy_eff
            + W_SAFE_RETURN * safe_return
        )
        return max(0.01, min(0.99, raw))

    def _at_base(self) -> bool:
        return math.hypot(self._uav_x - self._base[0],
                          self._uav_y - self._base[1]) < BASE_PROXIMITY

    # ------------------------------------------------------------------
    #  UX helpers — hints, error messages, reset summary
    # ------------------------------------------------------------------

    def _build_reset_message(self, task_name: str) -> str:
        nearest = self._nearest_rp()
        task = TASKS[task_name]
        wind_info = ""
        if task.get("wind_speed", 0) > 0:
            wind_info = f" Wind: ~{task['wind_speed']:.1f} m/s (variable)."
        lines = [
            f"Episode started. Task={task_name}, "
            f"{len(self._rps)} RPs to visit, "
            f"{len(self._obstacles)} obstacles, "
            f"battery={self._battery_pct():.0%}.{wind_info}",
            "",
            "Actions: move_N, move_NE, move_E, move_SE, move_S, move_SW, "
            "move_W, move_NW, hover_collect, return_base.",
            "",
            f"Goal: visit all RPs, collect data, return to base.",
        ]
        if nearest:
            rp_idx, dist = nearest
            pri = self._nodes[rp_idx]["priority"]
            lines.append(
                f"Nearest RP: id={rp_idx} (priority={pri}) at {dist:.0f} m "
                f"— try moving toward it first."
            )
        return " ".join(lines)

    def _handle_unknown_action(self, act: str) -> tuple[float, str]:
        matches = get_close_matches(act, ALL_ACTIONS, n=2, cutoff=0.4)
        if matches:
            suggestion = " or ".join(f"'{m}'" for m in matches)
            msg = f"Unknown action '{act}'. Did you mean {suggestion}?"
        else:
            msg = (
                f"Unknown action '{act}'. "
                f"Valid actions: {', '.join(ALL_ACTIONS)}."
            )
        return R_UNKNOWN_ACTION, msg

    def _build_hint(self) -> str:
        if self._done:
            return ""
        parts = []

        # Battery warning
        batt = self._battery_pct()
        if batt < 0.15:
            parts.append("CRITICAL: battery below 15%! Use return_base now.")
        elif batt < 0.30:
            parts.append("Warning: battery below 30%, consider returning soon.")

        # Suggest nearest unvisited RP
        nearest = self._nearest_rp()
        if nearest and not parts:
            rp_idx, dist = nearest
            pri = self._nodes[rp_idx]["priority"]
            if dist <= COLLECT_RADIUS:
                parts.append(
                    f"RP-{rp_idx} (priority={pri}) is within collect range "
                    f"— try hover_collect."
                )
            else:
                # Suggest direction
                node = self._nodes[rp_idx]
                direction = self._suggest_direction(node["x"], node["y"])
                parts.append(
                    f"Nearest unvisited RP-{rp_idx} (priority={pri}) is {dist:.0f} m away "
                    f"— try {direction}."
                )

        # All collected
        if len(self._visited_rps) == len(self._rps) and len(self._rps) > 0:
            parts.append("All RPs collected! Use return_base to finish.")

        if not parts:
            return ""
        return " [Hint: " + " ".join(parts) + "]"

    def _nearest_rp(self) -> Optional[tuple[int, float]]:
        best_idx, best_dist = None, float("inf")
        for rp_idx in self._rps:
            if rp_idx in self._visited_rps:
                continue
            node = self._nodes[rp_idx]
            d = math.hypot(self._uav_x - node["x"], self._uav_y - node["y"])
            if d < best_dist:
                best_idx, best_dist = rp_idx, d
        if best_idx is not None:
            return best_idx, best_dist
        return None

    def _suggest_direction(self, target_x: float, target_y: float) -> str:
        dx = target_x - self._uav_x
        dy = target_y - self._uav_y
        angle = math.atan2(-dy, dx)  # negative because y increases downward
        # Map angle to 8 compass directions
        dirs = ["move_E", "move_NE", "move_N", "move_NW",
                "move_W", "move_SW", "move_S", "move_SE"]
        idx = round(angle / (math.pi / 4)) % 8
        return dirs[idx]

    def _make_observation(self, reward: float, message: str) -> UAVObservation:
        sensors = []
        for rp_idx in self._rps:
            node = self._nodes[rp_idx]
            dist = math.hypot(self._uav_x - node["x"], self._uav_y - node["y"])
            sensors.append(SensorInfo(
                id=rp_idx,
                x=node["x"],
                y=node["y"],
                priority=node["priority"],
                distance=round(dist, 1),
                visited=rp_idx in self._visited_rps,
            ))

        obstacles = [
            ObstacleInfo(id=o["id"], x1=o["x1"], y1=o["y1"], x2=o["x2"], y2=o["y2"])
            for o in self._obstacles
        ]

        wx, wy = self._wind.vector
        return UAVObservation(
            done=self._done,
            reward=reward,
            uav_x=round(self._uav_x, 1),
            uav_y=round(self._uav_y, 1),
            battery_pct=round(self._battery_pct(), 4),
            rps_visited=len(self._visited_rps),
            rps_total=len(self._rps),
            data_collected=self._data_collected(),
            data_possible=self._data_possible(),
            sensors=sensors,
            obstacles=obstacles,
            base_x=self._base[0],
            base_y=self._base[1],
            dist_to_base=round(
                math.hypot(self._uav_x - self._base[0], self._uav_y - self._base[1]), 1,
            ),
            map_width=float(self._map_w),
            map_height=float(self._map_h),
            wind_x=round(wx, 2),
            wind_y=round(wy, 2),
            message=message,
            step_num=self._step_count,
            task_name=self._task_name,
            energy_per_move=E_FLY_PER_M * STEP_SIZE,
        )
