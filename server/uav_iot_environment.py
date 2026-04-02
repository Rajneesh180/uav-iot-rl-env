"""OpenEnv server — UAV navigates obstacles, collects IoT data, returns to base."""

from __future__ import annotations

import math
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import UAVAction, UAVObservation, UAVState
    from .simulation import (
        COLLECT_RADIUS, CRUISE_SPEED, E_FLY_PER_M, HOVER_TIME, P_HOVER,
        STEP_SIZE, TASKS, Coord, Node, Obstacle,
    )
    from .simulation.deployment import deploy_nodes, deploy_obstacles
    from .simulation.energy import energy_for_distance, energy_for_hover
    from .simulation.path_planning import (
        blocked, point_inside_any_obstacle, shortest_obstacle_free_path,
    )
    from .simulation.rp_selection import select_rendezvous_points
except ImportError:
    from models import UAVAction, UAVObservation, UAVState
    from server.simulation import (
        COLLECT_RADIUS, CRUISE_SPEED, E_FLY_PER_M, HOVER_TIME, P_HOVER,
        STEP_SIZE, TASKS, Coord, Node, Obstacle,
    )
    from server.simulation.deployment import deploy_nodes, deploy_obstacles
    from server.simulation.energy import energy_for_distance, energy_for_hover
    from server.simulation.path_planning import (
        blocked, point_inside_any_obstacle, shortest_obstacle_free_path,
    )
    from server.simulation.rp_selection import select_rendezvous_points

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


class UAVIoTEnvironment(Environment[UAVAction, UAVObservation, UAVState]):
    """UAV IoT data-collection environment with 3 difficulty levels."""

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
        self._battery: float = 0.0       # Joules remaining
        self._battery_cap: float = 0.0    # Joules total
        self._visited_rps: set[int] = set()
        self._total_reward: float = 0.0

    # ------------------------------------------------------------------
    #  OpenEnv interface
    # ------------------------------------------------------------------

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

        # Deploy sensors and obstacles
        effective_seed = seed if seed is not None else 42
        self._nodes = deploy_nodes(
            task["node_count"], self._map_w, self._map_h, seed=effective_seed,
        )
        self._obstacles = deploy_obstacles(
            task["obstacle_count"], self._map_w, self._map_h, seed=effective_seed,
        )

        # Select rendezvous points
        self._rps, self._rp_members = select_rendezvous_points(
            self._nodes, self._obstacles, task["rp_radius"],
        )

        # UAV starts at base
        self._uav_x, self._uav_y = self._base
        self._visited_rps = set()
        self._total_reward = 0.0

        return self._make_observation(
            reward=0.0,
            message=f"Episode started. Task={task_name}. "
                    f"Collect data from {len(self._rps)} RPs and return to base. "
                    f"Battery: {self._battery_pct():.0%}.",
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
            msg = f"Unknown action '{act}'. Valid: {', '.join(list(MOVE_DIRS) + ['hover_collect', 'return_base'])}"
            reward = -0.01

        # Small time penalty to encourage efficiency
        reward -= 0.005

        # Check termination conditions
        if self._battery <= 0:
            self._done = True
            reward -= 1.0
            msg += " Battery depleted! UAV crashed."
        elif self._step_count >= self._max_steps:
            self._done = True
            msg += " Max steps reached."

        self._total_reward += reward
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

        # Check obstacle collision
        if point_inside_any_obstacle(new_x, new_y, self._obstacles):
            self._battery -= energy_for_distance(STEP_SIZE * 0.1)  # small penalty for attempt
            return -0.05, f"Blocked! Position ({new_x:.0f},{new_y:.0f}) is inside an obstacle."

        # Check if path crosses obstacle
        if blocked((self._uav_x, self._uav_y), (new_x, new_y), self._obstacles):
            self._battery -= energy_for_distance(STEP_SIZE * 0.1)
            return -0.03, f"Path to ({new_x:.0f},{new_y:.0f}) blocked by an obstacle."

        dist = math.hypot(new_x - self._uav_x, new_y - self._uav_y)
        self._battery -= energy_for_distance(dist)
        self._uav_x = new_x
        self._uav_y = new_y
        return 0.0, f"Moved {direction.replace('move_', '')} to ({new_x:.0f},{new_y:.0f})."

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
            reward = sum(p / 10.0 for p in priorities) * 0.3
            names = ", ".join(f"RP-{i}(pri={self._nodes[i]['priority']})" for i in collected_ids)
            return reward, f"Collected data from {names}."
        return -0.01, "No uncollected RPs within collection range."

    def _do_return_base(self) -> tuple[float, str]:
        """Fly back to base using obstacle-aware path."""
        path, dist = shortest_obstacle_free_path(
            (self._uav_x, self._uav_y), self._base, self._obstacles,
        )
        energy = energy_for_distance(dist)

        if energy > self._battery:
            # Not enough battery to return — still try but will crash
            fraction = self._battery / energy if energy > 0 else 0
            partial_idx = max(0, int(len(path) * fraction) - 1)
            if partial_idx > 0 and partial_idx < len(path):
                self._uav_x, self._uav_y = path[partial_idx]
            self._battery = 0.0
            return -0.5, "Insufficient battery to reach base. UAV stranded."

        self._battery -= energy
        self._uav_x, self._uav_y = self._base
        self._done = True

        # Reward for safe return scales with data collected
        coverage = len(self._visited_rps) / max(1, len(self._rps))
        reward = 0.5 * coverage + 0.2  # base reward for returning safely
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
        """Compute final score (0.0–1.0) for grading."""
        if not self._rps:
            return 0.0

        coverage = len(self._visited_rps) / len(self._rps)
        dp = self._data_possible()
        priority_score = self._data_collected() / dp if dp > 0 else 0.0
        energy_eff = self._battery_pct()  # higher remaining = more efficient
        safe_return = 1.0 if self._at_base() and self._battery > 0 else 0.0

        return (
            0.35 * coverage
            + 0.25 * priority_score
            + 0.25 * energy_eff
            + 0.15 * safe_return
        )

    def _at_base(self) -> bool:
        return math.hypot(self._uav_x - self._base[0],
                          self._uav_y - self._base[1]) < 10.0

    def _make_observation(self, reward: float, message: str) -> UAVObservation:
        sensors = []
        for rp_idx in self._rps:
            node = self._nodes[rp_idx]
            dist = math.hypot(self._uav_x - node["x"], self._uav_y - node["y"])
            sensors.append({
                "id": rp_idx,
                "x": node["x"],
                "y": node["y"],
                "priority": node["priority"],
                "distance": round(dist, 1),
                "visited": rp_idx in self._visited_rps,
            })

        obstacles = [
            {"id": o["id"], "x1": o["x1"], "y1": o["y1"], "x2": o["x2"], "y2": o["y2"]}
            for o in self._obstacles
        ]

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
            message=message,
            step_num=self._step_count,
            task_name=self._task_name,
            energy_per_move=E_FLY_PER_M * STEP_SIZE,
        )
