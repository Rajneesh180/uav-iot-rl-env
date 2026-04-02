# Simulation constants and task configurations

from __future__ import annotations

from typing import Any

# Type aliases
Node = dict[str, Any]
Obstacle = dict[str, Any]
Coord = tuple[float, float]

# Task configurations
TASKS = {
    "easy": {
        "map_w": 400,
        "map_h": 300,
        "node_count": 8,
        "obstacle_count": 0,
        "battery_capacity": 600_000.0,
        "rp_radius": 120.0,
        "max_steps": 80,
        "description": "Small area, no obstacles, few sensors, full battery.",
    },
    "medium": {
        "map_w": 600,
        "map_h": 450,
        "node_count": 14,
        "obstacle_count": 3,
        "battery_capacity": 480_000.0,
        "rp_radius": 120.0,
        "max_steps": 120,
        "description": "Mid-size area, 3 obstacles, moderate sensors, 80% battery.",
    },
    "hard": {
        "map_w": 800,
        "map_h": 600,
        "node_count": 20,
        "obstacle_count": 5,
        "battery_capacity": 360_000.0,
        "rp_radius": 120.0,
        "max_steps": 160,
        "description": "Full area, 5 obstacles, many sensors, 60% battery.",
    },
}

# Common constants
NODE_MARGIN: int = 40
STEP_SIZE: float = 50.0       # metres per move action
UAV_ALTITUDE: int = 100
E_FLY_PER_M: float = 17.0    # J/m at cruise speed
P_HOVER: float = 168.483     # Watts (P_0 + P_i)
CRUISE_SPEED: float = 10.0   # m/s
HOVER_TIME: float = 10.0     # seconds per hover-collect action
COLLECT_RADIUS: float = 60.0 # metres — how close UAV must be to collect from an RP
