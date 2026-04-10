"""Simulation constants, task configurations, and reward parameters."""

from __future__ import annotations

from typing import Any

# Type aliases
Node = dict[str, Any]
Obstacle = dict[str, Any]
Coord = tuple[float, float]

# ── Physics ──────────────────────────────────────────────────────
STEP_SIZE: float = 50.0
E_FLY_PER_M: float = 17.0        # J/m at cruise speed (Zeng & Zhang 2017)
P_HOVER: float = 168.483         # W — blade profile + induced power
CRUISE_SPEED: float = 10.0       # m/s
HOVER_TIME: float = 10.0         # seconds per hover-collect
COLLECT_RADIUS: float = 60.0     # metres to collect from an RP
NODE_MARGIN: int = 40
UAV_ALTITUDE: int = 100
BASE_PROXIMITY: float = 10.0     # metres — threshold for "at base"
WIND_PERSISTENCE: float = 0.8    # AR(1) coefficient for wind model
OBSTACLE_ATTEMPT_FRAC: float = 0.1  # energy fraction wasted on blocked move

# ── Reward signals ───────────────────────────────────────────────
R_COLLECT_PER_PRI: float = 0.03  # per priority point collected
R_OBSTACLE_HIT: float = -0.05
R_PATH_BLOCKED: float = -0.03
R_HOVER_MISS: float = -0.01
R_TIME_PENALTY: float = -0.005
R_CRASH: float = -1.0
R_STRANDED: float = -0.5
R_RETURN_BASE: float = 0.2
R_RETURN_COVERAGE: float = 0.5
R_UNKNOWN_ACTION: float = -0.01

# ── Scoring weights ──────────────────────────────────────────────
W_COVERAGE: float = 0.35
W_PRIORITY: float = 0.25
W_ENERGY: float = 0.25
W_SAFE_RETURN: float = 0.15

# ── Task configurations ─────────────────────────────────────────
TASKS = {
    "easy": {
        "map_w": 400,
        "map_h": 300,
        "node_count": 8,
        "obstacle_count": 0,
        "battery_capacity": 600_000.0,
        "rp_radius": 120.0,
        "max_steps": 80,
        "wind_speed": 0.0,
        "wind_variability": 0.0,
        "description": "Small area, calm conditions, no obstacles.",
    },
    "medium": {
        "map_w": 600,
        "map_h": 450,
        "node_count": 14,
        "obstacle_count": 3,
        "battery_capacity": 480_000.0,
        "rp_radius": 120.0,
        "max_steps": 120,
        "wind_speed": 1.0,
        "wind_variability": 0.5,
        "description": "Mid-size area, light wind, moderate obstacles.",
    },
    "hard": {
        "map_w": 800,
        "map_h": 600,
        "node_count": 20,
        "obstacle_count": 5,
        "battery_capacity": 360_000.0,
        "rp_radius": 120.0,
        "max_steps": 160,
        "wind_speed": 2.0,
        "wind_variability": 1.0,
        "description": "Large area, moderate wind, dense obstacles.",
    },
    "expert": {
        "map_w": 1000,
        "map_h": 750,
        "node_count": 28,
        "obstacle_count": 7,
        "battery_capacity": 300_000.0,
        "rp_radius": 120.0,
        "max_steps": 200,
        "wind_speed": 3.5,
        "wind_variability": 2.5,
        "description": "Full-scale area, gusty wind, tight energy budget.",
    },
}
