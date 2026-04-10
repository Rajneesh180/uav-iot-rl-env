"""Rotary-wing UAV energy model (Zeng & Zhang 2017)."""

from __future__ import annotations

from . import E_FLY_PER_M, HOVER_TIME, P_HOVER


def energy_for_distance(dist: float) -> float:
    """Joules consumed flying `dist` metres."""
    return dist * E_FLY_PER_M


def energy_for_hover(seconds: float = HOVER_TIME) -> float:
    """Joules consumed hovering for `seconds`."""
    return seconds * P_HOVER
