"""The synthetic market.

Returns are a function of how long the position has been held, not of calendar
time. This is a deliberate identification device, not a claim about real assets:
it is what forces the agent to reason about holding length (README section 11, T1).
"""
from dataclasses import dataclass

HOLDINGS = ("A", "B", "C", "CASH")
CASH = 3


@dataclass(frozen=True)
class AssetSpec:
    """A two-regime return schedule keyed on the per-position age clock."""

    name: str
    early_rate: float
    late_rate: float
    breakpoint: int

    def daily_return(self, age: int) -> float:
        """Return earned on a day when this position has been held `age` days (1-indexed)."""
        if age < 1:
            raise ValueError(f"age must be >= 1, got {age}")
        return self.early_rate if age <= self.breakpoint else self.late_rate


@dataclass(frozen=True)
class Market:
    assets: tuple
    switch_cost: float = 0.001

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def index(self, name: str) -> int:
        for i, a in enumerate(self.assets):
            if a.name == name:
                return i
        raise KeyError(f"unknown holding {name!r}")

    def growth(self, holding: int, age: int) -> float:
        """Wealth multiplier for holding `holding` on a day at age `age`."""
        return 1.0 + self.assets[holding].daily_return(age)


def default_market(switch_cost: float = 0.001) -> Market:
    """Frozen calibrated parameters (2026-08-27).

    Chosen by automated search over 9,216 candidates against the calibration
    targets, never hand-tuned. 1,386 combinations were feasible; the winner is
    the highest value-separation margin among those that also pass the
    non-collapse check on the uniform condition. The full feasible set is
    committed in data/calibration.json for the appendix, so the study's
    conclusions can be shown not to hinge on one lucky parameter choice.

    A "Sprinter" pays a high rate while freshly held and then stops, B
    "Compounder" is flat, C "Late bloomer" is weak early and strong late, so the
    best holding depends on how long it will be held.

    A decaying asset is only safe because re-entry is forbidden
    (horizon.env.rules). Without that rule the optimum farms A's clock reset by
    bouncing A -> B -> A forever, and the policy stops depending on t entirely.
    """
    return Market(
        assets=(
            AssetSpec("A", early_rate=0.012, late_rate=0.0, breakpoint=15),
            AssetSpec("B", early_rate=0.003, late_rate=0.003, breakpoint=1),
            AssetSpec("C", early_rate=0.0005, late_rate=0.008, breakpoint=50),
            AssetSpec("CASH", early_rate=0.0, late_rate=0.0, breakpoint=1),
        ),
        switch_cost=switch_cost,
    )
