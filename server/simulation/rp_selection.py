# Greedy dominating-set for rendezvous-point selection

from __future__ import annotations

import math

from . import Node, Obstacle


def inside_obstacle(nx: float, ny: float, obstacles: list[Obstacle]) -> bool:
    for o in obstacles:
        if o["x1"] <= nx <= o["x2"] and o["y1"] <= ny <= o["y2"]:
            return True
    return False


def select_rendezvous_points(
    nodes: list[Node],
    obstacles: list[Obstacle],
    r_max: float = 120.0,
) -> tuple[list[int], dict[int, list[int]]]:
    """Pick RPs using greedy dominating set, skip nodes inside obstacles."""
    n = len(nodes)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(nodes[i]["x"] - nodes[j]["x"],
                          nodes[i]["y"] - nodes[j]["y"]) <= r_max:
                adj[i].add(j)
                adj[j].add(i)

    uncovered: set[int] = set(range(n))
    rps: list[int] = []
    members: dict[int, list[int]] = {}

    while uncovered:
        best = max(uncovered, key=lambda i: len(adj[i] & uncovered))
        if inside_obstacle(nodes[best]["x"], nodes[best]["y"], obstacles):
            uncovered.discard(best)
            continue
        rps.append(best)
        covered = {best} | (adj[best] & uncovered)
        members[best] = sorted(covered - {best})
        uncovered -= covered

    # Assign orphan nodes to nearest RP
    all_covered: set[int] = set()
    for rp in rps:
        all_covered.add(rp)
        all_covered.update(members.get(rp, []))

    for orphan in set(range(n)) - all_covered:
        closest = min(
            rps,
            key=lambda r: math.hypot(
                nodes[r]["x"] - nodes[orphan]["x"],
                nodes[r]["y"] - nodes[orphan]["y"],
            ),
        )
        members.setdefault(closest, []).append(orphan)

    return rps, members
