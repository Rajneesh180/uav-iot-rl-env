# Visibility-graph Dijkstra pathfinding with rectangular obstacles

from __future__ import annotations

import math

from . import Coord, Obstacle

_MARGIN: float = 20.0


def _cross2d(ox: float, oy: float, ax: float, ay: float,
             bx: float, by: float) -> float:
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _segs_cross(ax: float, ay: float, bx: float, by: float,
                cx: float, cy: float, dx: float, dy: float) -> bool:
    d1 = _cross2d(cx, cy, dx, dy, ax, ay)
    d2 = _cross2d(cx, cy, dx, dy, bx, by)
    d3 = _cross2d(ax, ay, bx, by, cx, cy)
    d4 = _cross2d(ax, ay, bx, by, dx, dy)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _point_strictly_inside(px: float, py: float,
                           x1: float, y1: float,
                           x2: float, y2: float) -> bool:
    return x1 < px < x2 and y1 < py < y2


def _segment_crosses_rect(p1: Coord, p2: Coord,
                          x1: float, y1: float,
                          x2: float, y2: float) -> bool:
    if _point_strictly_inside(p1[0], p1[1], x1, y1, x2, y2):
        return True
    if _point_strictly_inside(p2[0], p2[1], x1, y1, x2, y2):
        return True
    edges = [
        (x1, y1, x2, y1),
        (x2, y1, x2, y2),
        (x2, y2, x1, y2),
        (x1, y2, x1, y1),
    ]
    return any(
        _segs_cross(p1[0], p1[1], p2[0], p2[1], ex1, ey1, ex2, ey2)
        for ex1, ey1, ex2, ey2 in edges
    )


def blocked(p1: Coord, p2: Coord, obstacles: list[Obstacle]) -> bool:
    """True if the line segment p1->p2 passes through any obstacle."""
    for obs in obstacles:
        if _segment_crosses_rect(p1, p2,
                                 obs["x1"], obs["y1"],
                                 obs["x2"], obs["y2"]):
            return True
    return False


def point_inside_any_obstacle(x: float, y: float, obstacles: list[Obstacle]) -> bool:
    """True if the point (x,y) is inside any obstacle."""
    for obs in obstacles:
        if obs["x1"] <= x <= obs["x2"] and obs["y1"] <= y <= obs["y2"]:
            return True
    return False


def obstacle_corners(obstacles: list[Obstacle]) -> list[Coord]:
    """Waypoint candidates: corners at MARGIN distance from each obstacle."""
    pts: list[Coord] = []
    for obs in obstacles:
        x1 = obs["x1"] - _MARGIN
        y1 = obs["y1"] - _MARGIN
        x2 = obs["x2"] + _MARGIN
        y2 = obs["y2"] + _MARGIN
        pts.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return pts


def shortest_obstacle_free_path(
    start: Coord, end: Coord, obstacles: list[Obstacle],
) -> tuple[list[Coord], float]:
    """Visibility-graph Dijkstra returning (path, distance)."""
    if not blocked(start, end, obstacles):
        d = math.hypot(end[0] - start[0], end[1] - start[1])
        return [start, end], d

    corners = obstacle_corners(obstacles)
    verts = [start] + corners + [end]
    n = len(verts)

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not blocked(verts[i], verts[j], obstacles):
                d = math.hypot(verts[j][0] - verts[i][0],
                               verts[j][1] - verts[i][1])
                adj[i].append((j, d))
                adj[j].append((i, d))

    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    dist[0] = 0.0
    visited = [False] * n

    for _ in range(n):
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or dist[v] < dist[u]):
                u = v
        if u == -1 or dist[u] == INF:
            break
        visited[u] = True
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    end_idx = n - 1
    if dist[end_idx] == INF:
        d = math.hypot(end[0] - start[0], end[1] - start[1])
        return [start, end], d

    path: list[Coord] = []
    cur = end_idx
    while cur != -1:
        path.append(verts[cur])
        cur = prev[cur]
    path.reverse()
    return path, dist[end_idx]
