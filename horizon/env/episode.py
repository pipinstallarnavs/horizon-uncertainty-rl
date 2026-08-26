"""Wealth accounting for a sequence of daily holdings.

Timing convention (locked): on each day the agent acts, the day's return
accrues, and only then may the exit fire. A liquidated agent therefore receives
that day's return, because liquidation happens at that day's prices
(README section 2.1).
"""
import numpy as np

from horizon.env.market import CASH, Market
from horizon.env.rules import burn, is_available


def ages_for(holdings) -> list:
    """Per-day age index. The agent starts in CASH with age 0 before day 1."""
    ages = []
    previous = CASH
    age = 0
    for holding in holdings:
        age = age + 1 if holding == previous else 1
        ages.append(age)
        previous = holding
    return ages


def is_legal(holdings) -> bool:
    """True if the sequence never re-enters a risky asset it has left."""
    previous, mask = CASH, 0
    for holding in holdings:
        if holding != previous:
            if not is_available(mask, holding):
                return False
            mask = burn(mask, previous)
        previous = holding
    return True


def wealth_path(market: Market, holdings) -> np.ndarray:
    """Wealth at the end of each day, starting from W[0] = 1.0.

    Rejects sequences that violate the no-re-entry rule rather than silently
    pricing them, so an illegal policy can never contribute to a reported value.
    """
    wealth = np.ones(len(holdings) + 1)
    previous, mask, age = CASH, 0, 0
    for day, holding in enumerate(holdings, start=1):
        if holding == previous:
            age += 1
            cost = 1.0
        else:
            if not is_available(mask, holding):
                raise ValueError(
                    f"illegal re-entry into asset {holding} on day {day}: "
                    "an asset that has been left may not be held again"
                )
            mask = burn(mask, previous)
            age = 1
            cost = 1.0 - market.switch_cost
        wealth[day] = wealth[day - 1] * cost * market.growth(holding, age)
        previous = holding
    return wealth


def expected_wealth(market: Market, holdings, pmf: np.ndarray) -> float:
    """E[W_T] = sum_t P(T = t) * W_t for a fixed holding sequence."""
    wealth = wealth_path(market, holdings)
    horizon = min(len(wealth) - 1, len(pmf) - 1)
    return float(sum(pmf[t] * wealth[t] for t in range(1, horizon + 1)))
