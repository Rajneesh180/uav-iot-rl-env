# Random sensor / obstacle placement

from __future__ import annotations

import random

from . import Node, Obstacle

_BASE_CLEARANCE = 80  # min distance from map-centre base to any obstacle edge


def _overlaps_base(x1: int, y1: int, x2: int, y2: int,
                   bx: float, by: float) -> bool:
    """True if rectangle [x1,y1,x2,y2] is too close to the base station."""
    return (x1 - _BASE_CLEARANCE < bx < x2 + _BASE_CLEARANCE and
            y1 - _BASE_CLEARANCE < by < y2 + _BASE_CLEARANCE)


def _inside_any(x: float, y: float, obstacles: list[Obstacle]) -> bool:
    for o in obstacles:
        if o["x1"] <= x <= o["x2"] and o["y1"] <= y <= o["y2"]:
            return True
    return False


def deploy_nodes(
    n: int,
    map_w: int,
    map_h: int,
    margin: int = 40,
    seed: int = 42,
    obstacles: list[Obstacle] | None = None,
) -> list[Node]:
    """Place n sensors at random positions, avoiding obstacle interiors."""
    rng = random.Random(seed)
    obs = obstacles or []
    nodes: list[Node] = []
    for i in range(n):
        for _ in range(200):
            x = rng.randint(margin, map_w - margin)
            y = rng.randint(margin, map_h - margin)
            if not _inside_any(x, y, obs):
                break
        nodes.append({
            "id": i,
            "x": x, "y": y,
            "priority": rng.randint(1, 10),
        })
    return nodes


def deploy_obstacles(
    k: int,
    map_w: int,
    map_h: int,
    seed: int = 42,
) -> list[Obstacle]:
    """Place k rectangular obstacles, never covering the base station."""
    rng = random.Random(seed + 100)
    bx, by = map_w / 2, map_h / 2
    obstacles: list[Obstacle] = []
    for i in range(k):
        for _ in range(200):
            x1 = rng.randint(60, max(61, map_w - 200))
            y1 = rng.randint(60, max(61, map_h - 180))
            w = rng.randint(60, 120)
            h = rng.randint(50, 100)
            if not _overlaps_base(x1, y1, x1 + w, y1 + h, bx, by):
                break
        obstacles.append({
            "id": i,
            "x1": x1, "y1": y1,
            "x2": x1 + w, "y2": y1 + h,
            "cx": x1 + w / 2, "cy": y1 + h / 2,
            "width": w, "depth": h,
        })
    return obstacles
