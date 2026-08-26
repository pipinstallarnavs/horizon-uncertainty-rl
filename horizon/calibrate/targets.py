"""Calibration targets from DESIGN.md P3, as executable predicates.

Asset parameters are selected by search against these, never hand-tuned, so that
"time matters in this environment" is a property the code proves rather than a
property the designer asserts.
"""
from dataclasses import dataclass

from horizon.env.episode import expected_wealth
from horizon.env.exit_dist import ExitDistribution
from horizon.env.market import CASH, Market
from horizon.solver.backward import Solution, optimal_holdings, solve

MIN_CROSSOVER_DAY = 15
MAX_CROSSOVER_DAY = 85
MIN_VALUE_MARGIN = 0.05
MIN_CASH_MARGIN = 0.15
MIN_SWITCHES = 1.0
MAX_SWITCHES = 4.0


def fixed_asset_values(market: Market, dist: ExitDistribution) -> dict:
    """Expected wealth of each always-hold-one-thing baseline."""
    return {
        asset.name: expected_wealth(market, [k] * dist.N, dist.pmf)
        for k, asset in enumerate(market.assets)
    }


def crossover_day(sol: Solution, market: Market, dist: ExitDistribution):
    """First day the optimal holding changes after the initial entry, or None.

    The day-1 move out of CASH is an entry, not a crossover, so it is skipped:
    what the target cares about is the horizon-driven switch between risky assets.
    """
    holdings = optimal_holdings(sol, market, dist.N)
    for day in range(1, len(holdings)):
        if holdings[day] != holdings[day - 1]:
            return day + 1
    return None


def expected_switches(sol: Solution, market: Market, dist: ExitDistribution) -> float:
    """E[number of switches before exit], weighted by the exit distribution."""
    holdings = optimal_holdings(sol, market, dist.N)
    previous = CASH
    count = 0
    switches_by_day = []
    for holding in holdings:
        if holding != previous:
            count += 1
        switches_by_day.append(count)
        previous = holding
    return float(sum(dist.pmf[t] * switches_by_day[t - 1] for t in range(1, dist.N + 1)))


@dataclass(frozen=True)
class TargetReport:
    interior_crossover: bool
    value_separation: bool
    cash_separation: bool
    no_dominant_asset: bool
    cost_relevance: bool

    @property
    def all_pass(self) -> bool:
        return all(
            (
                self.interior_crossover,
                self.value_separation,
                self.cash_separation,
                self.no_dominant_asset,
                self.cost_relevance,
            )
        )


def evaluate_targets(
    market: Market, dist_uniform: ExitDistribution, dist_bimodal: ExitDistribution
) -> TargetReport:
    """Score a candidate market against all five calibration targets."""
    sol_u = solve(market, dist_uniform)
    sol_b = solve(market, dist_bimodal)

    day = crossover_day(sol_u, market, dist_uniform)
    interior = day is not None and MIN_CROSSOVER_DAY <= day <= MAX_CROSSOVER_DAY

    fixed = fixed_asset_values(market, dist_uniform)
    best_fixed = max(fixed.values())
    value_sep = sol_u.v_star >= best_fixed * (1.0 + MIN_VALUE_MARGIN)
    cash_sep = sol_u.v_star >= fixed["CASH"] * (1.0 + MIN_CASH_MARGIN)

    # No single asset should be optimal in every regime we care about: the
    # short-horizon uniform start, the bimodal start, and the post-resolution
    # bimodal regime must not all pick the same holding.
    regimes = {
        optimal_holdings(sol_u, market, dist_uniform.N)[0],
        optimal_holdings(sol_b, market, dist_bimodal.N)[0],
        optimal_holdings(sol_b, market, dist_bimodal.N)[-1],
    }
    no_dominant = len(regimes) > 1

    switches = expected_switches(sol_u, market, dist_uniform)
    cost_relevant = MIN_SWITCHES <= switches <= MAX_SWITCHES

    return TargetReport(interior, value_sep, cash_sep, no_dominant, cost_relevant)
