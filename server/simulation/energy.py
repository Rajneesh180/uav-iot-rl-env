# Rotary-wing UAV energy model

from __future__ import annotations

from . import E_FLY_PER_M, P_HOVER, HOVER_TIME, CRUISE_SPEED


def energy_for_distance(dist: float) -> float:
    """Joules consumed flying `dist` metres."""
    return dist * E_FLY_PER_M


def energy_for_hover(seconds: float = HOVER_TIME) -> float:
    """Joules consumed hovering for `seconds`."""
    return seconds * P_HOVER


def estimate_energy(
    total_dist: float,
    n_rps: int,
    hover_time: float = HOVER_TIME,
    speed: float = CRUISE_SPEED,
) -> tuple[float, float]:
    """Return (total energy in joules, fraction of 600kJ battery used)."""
    e_flight = total_dist * E_FLY_PER_M
    e_hover = n_rps * hover_time * P_HOVER
    e_total = e_flight + e_hover
    pct = e_total / 600_000.0
    return e_total, pct
