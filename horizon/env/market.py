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
    """Provisional parameters from README section 4; frozen by calibration (Task 9)."""
    return Market(
        assets=(
            AssetSpec("A", early_rate=0.008, late_rate=-0.002, breakpoint=10),
            AssetSpec("B", early_rate=0.0025, late_rate=0.0025, breakpoint=1),
            AssetSpec("C", early_rate=0.0005, late_rate=0.006, breakpoint=40),
            AssetSpec("CASH", early_rate=0.0, late_rate=0.0, breakpoint=1),
        ),
        switch_cost=switch_cost,
    )
