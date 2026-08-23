"""Exit-time distributions and the temporal quantities derived from them.

Every temporal quantity in the study (hazard, expected remaining time, the
ground-truth beliefs the agent is scored against) is derived here from a single
pmf, so a new exit distribution only ever needs to supply P(T = t).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExitDistribution:
    """A distribution over the forced-liquidation day T, supported on 1..N.

    pmf is length N+1; index 0 is unused padding so that pmf[t] reads naturally
    as P(T = t) for the 1-indexed day t used throughout the study.
    """

    name: str
    pmf: np.ndarray

    def __post_init__(self) -> None:
        pmf = np.asarray(self.pmf, dtype=float)
        object.__setattr__(self, "pmf", pmf)
        if pmf.ndim != 1 or pmf.size < 2:
            raise ValueError("pmf must be a 1-D array of length N+1 with N >= 1")
        if pmf[0] != 0.0:
            raise ValueError("pmf index 0 is unused padding and must be 0.0")
        if (pmf < 0).any():
            raise ValueError("pmf must be non-negative")
        total = pmf[1:].sum()
        if not np.isclose(total, 1.0):
            raise ValueError(f"pmf must sum to 1 over days 1..N, got {total}")

    @property
    def N(self) -> int:
        return int(self.pmf.size - 1)

    def survival(self, t: int) -> float:
        """S(t) = P(T > t). S(0) = 1, S(N) = 0."""
        return float(self.pmf[t + 1 :].sum())

    def hazard(self, t: int) -> float:
        """h(t) = P(T = t | T >= t). Zero where the conditioning event is empty."""
        prior = self.survival(t - 1)
        if prior <= 0.0:
            return 0.0
        return float(self.pmf[t] / prior)

    def mean_remaining(self, t: int) -> float:
        """m(t) = E[T - t | T > t], the true belief the agent's probe is scored against."""
        s = self.survival(t)
        if s <= 0.0:
            return 0.0
        days = np.arange(t + 1, self.N + 1)
        return float(((days - t) * self.pmf[t + 1 :]).sum() / s)
