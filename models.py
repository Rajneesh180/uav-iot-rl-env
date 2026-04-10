"""Action / Observation / State models for the UAV-IoT environment."""

from typing import List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class SensorInfo(BaseModel):
    id: int
    x: float
    y: float
    priority: int
    distance: float
    visited: bool = False


class ObstacleInfo(BaseModel):
    id: int
    x1: float
    y1: float
    x2: float
    y2: float


class UAVAction(Action):
    action: str = Field(
        ...,
        description=(
            "One of: move_N, move_NE, move_E, move_SE, move_S, move_SW, "
            "move_W, move_NW, hover_collect, return_base"
        ),
    )


class UAVObservation(Observation):
    uav_x: float = Field(..., description="UAV X position (metres)")
    uav_y: float = Field(..., description="UAV Y position (metres)")
    battery_pct: float = Field(..., description="Battery remaining (0.0-1.0)")

    rps_visited: int = Field(0)
    rps_total: int = Field(0)
    data_collected: float = Field(0.0)
    data_possible: float = Field(0.0)

    sensors: List[SensorInfo] = Field(default_factory=list)
    obstacles: List[ObstacleInfo] = Field(default_factory=list)
    base_x: float = Field(0.0)
    base_y: float = Field(0.0)
    dist_to_base: float = Field(0.0)
    map_width: float = Field(0.0)
    map_height: float = Field(0.0)

    wind_x: float = Field(0.0, description="Wind x-component (m/s)")
    wind_y: float = Field(0.0, description="Wind y-component (m/s)")

    message: str = Field("")
    step_num: int = Field(0)
    task_name: str = Field("")
    energy_per_move: float = Field(0.0)


class UAVState(State):
    task_name: str = ""
    rps_visited: int = 0
    rps_total: int = 0
    battery_pct: float = 1.0
    data_collected: float = 0.0
    data_possible: float = 0.0
    wind_speed: float = 0.0
    score: float = 0.0
