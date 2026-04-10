"""Autoregressive wind model for UAV flight simulation.

Models atmospheric turbulence as a correlated random process (AR(1)).
Wind vector persists across timesteps with gradual drift, producing
realistic gusts rather than white-noise jitter.
"""

from __future__ import annotations

import math
import random


class WindModel:

    def __init__(
        self,
        base_speed: float,
        variability: float,
        persistence: float = 0.8,
        seed: int | None = None,
    ):
        self._rng = random.Random(seed)
        self._var = variability
        self._alpha = persistence
        self._wx = 0.0
        self._wy = 0.0
        if base_speed > 0:
            angle = self._rng.uniform(0, 2 * math.pi)
            self._wx = base_speed * math.cos(angle)
            self._wy = base_speed * math.sin(angle)

    def step(self) -> tuple[float, float]:
        """Advance wind state by one timestep, return (wx, wy) in m/s."""
        if self._var <= 0:
            return self._wx, self._wy
        self._wx = (
            self._alpha * self._wx
            + (1 - self._alpha) * self._rng.gauss(0, self._var)
        )
        self._wy = (
            self._alpha * self._wy
            + (1 - self._alpha) * self._rng.gauss(0, self._var)
        )
        return self._wx, self._wy

    @property
    def speed(self) -> float:
        return math.hypot(self._wx, self._wy)

    @property
    def vector(self) -> tuple[float, float]:
        return self._wx, self._wy
