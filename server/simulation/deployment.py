# Random sensor / obstacle placement

from __future__ import annotations

import random

from . import Node, Obstacle


def deploy_nodes(
    n: int,
    map_w: int,
    map_h: int,
    margin: int = 40,
    seed: int = 42,
) -> list[Node]:
    """Place n sensors at random positions with priorities 1-10."""
    rng = random.Random(seed)
    nodes: list[Node] = []
    for i in range(n):
        nodes.append({
            "id": i,
            "x": rng.randint(margin, map_w - margin),
            "y": rng.randint(margin, map_h - margin),
            "priority": rng.randint(1, 10),
            "data_level": round(rng.uniform(0.1, 1.0), 2),
        })
    return nodes


def deploy_obstacles(
    k: int,
    map_w: int,
    map_h: int,
    seed: int = 42,
) -> list[Obstacle]:
    """Place k rectangular obstacles."""
    rng = random.Random(seed + 100)
    obstacles: list[Obstacle] = []
    for i in range(k):
        x1 = rng.randint(60, max(61, map_w - 200))
        y1 = rng.randint(60, max(61, map_h - 180))
        w = rng.randint(60, 120)
        h = rng.randint(50, 100)
        obstacles.append({
            "id": i,
            "x1": x1, "y1": y1,
            "x2": x1 + w, "y2": y1 + h,
            "cx": x1 + w / 2, "cy": y1 + h / 2,
            "width": w, "depth": h,
        })
    return obstacles
