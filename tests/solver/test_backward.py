import pytest

from horizon.env.distributions import bimodal_exit, geometric_exit, uniform_exit
from horizon.env.market import CASH, AssetSpec, Market, default_market
from horizon.solver.backward import optimal_holdings, solve


def one_asset_market(rate):
    return Market(
        assets=(
            AssetSpec("A", rate, rate, 1),
            AssetSpec("B", rate, rate, 1),
            AssetSpec("C", rate, rate, 1),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=0.001,
    )


def test_value_of_a_flat_market_is_one():
    # If nothing grows, the best you can do is hold cash and never pay a cost.
    m = one_asset_market(0.0)
    sol = solve(m, uniform_exit(20))
    assert sol.v_star == pytest.approx(1.0)
    assert optimal_holdings(sol, m, 20) == [CASH] * 20


def test_a_strictly_profitable_asset_is_entered_on_day_one():
    m = one_asset_market(0.01)
    sol = solve(m, uniform_exit(20))
    assert optimal_holdings(sol, m, 20)[0] != CASH
    assert sol.v_star > 1.0


def test_value_matches_hand_computed_two_day_case():
    # N=2, uniform: T is day 1 or day 2 with prob 1/2 each.
    # Holding A both days: W1 = 0.999*1.01, W2 = W1*1.01.
    m = one_asset_market(0.01)
    sol = solve(m, uniform_exit(2))
    w1 = 0.999 * 1.01
    expected = 0.5 * w1 + 0.5 * (w1 * 1.01)
    assert sol.v_star == pytest.approx(expected)


def test_switch_cost_deters_a_marginal_switch():
    # A gain smaller than the 0.1% switch cost is not worth taking.
    m = Market(
        assets=(
            AssetSpec("A", 0.0, 0.0, 1),
            AssetSpec("B", 0.0001, 0.0001, 1),
            AssetSpec("C", 0.0, 0.0, 1),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=0.001,
    )
    sol = solve(m, uniform_exit(5))
    assert optimal_holdings(sol, m, 5) == [CASH] * 5


def test_policy_is_stationary_under_constant_hazard():
    # The control condition: geometric exit collapses to ordinary discounting,
    # so the optimal action must not depend on t except near truncation.
    m = default_market()
    sol = solve(m, geometric_exit(60, p=0.02))
    actions = [sol.pi[t, CASH, 0] for t in range(1, 40)]
    assert len(set(actions)) == 1


def _action_change_fraction(sol, t):
    """Fraction of reachable (holding, age) states whose action differs from day t-1."""
    changed = total = 0
    for k in range(4):
        for age in range(0, t - 1):
            total += 1
            if sol.pi[t - 1, k, age] != sol.pi[t, k, age]:
                changed += 1
    return changed / total


def test_a_concentrated_early_mode_produces_a_bright_line():
    # DESIGN.md P3 target 5 wants the optimal action to change at the mode
    # boundary "for the majority of reachable states". That requires the early
    # mode to be CONCENTRATED: mass spread over days 1-10 resolves the posterior
    # gradually, so by day 10 the agent already knows it is in the late mode and
    # crossing into day 11 changes almost nothing. See the docs amendment.
    m = Market(
        assets=(
            AssetSpec("A", 0.0, 0.0, 1),
            AssetSpec("B", 0.006, 0.006, 1),
            AssetSpec("C", 0.0, 0.008, 20),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=0.001,
    )
    concentrated = bimodal_exit(early=(10, 10), late=(91, 100))
    assert _action_change_fraction(solve(m, concentrated), 11) > 0.5


def test_a_spread_early_mode_produces_no_bright_line():
    # The negative control for the test above, and the reason the bimodal
    # condition as originally specified cannot be scored on day-11 timing.
    m = Market(
        assets=(
            AssetSpec("A", 0.0, 0.0, 1),
            AssetSpec("B", 0.006, 0.006, 1),
            AssetSpec("C", 0.0, 0.008, 20),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=0.001,
    )
    spread = bimodal_exit(early=(1, 10), late=(91, 100))
    assert _action_change_fraction(solve(m, spread), 11) == 0.0


def test_readme_default_parameters_are_degenerate():
    # The provisional parameters in README section 4 admit an age-clock exploit:
    # A's boost is worth 8% while a switch costs 0.1%, so the optimum cycles
    # A -> B -> A to reset A's clock forever. The resulting policy is identical
    # at every t, which is exactly the collapse DESIGN.md P1 forbids.
    m = default_market()
    sol = solve(m, bimodal_exit())
    assert all(_action_change_fraction(sol, t) == 0.0 for t in range(2, 101))


def test_solution_shapes_are_consistent():
    m = default_market()
    dist = uniform_exit(30)
    sol = solve(m, dist)
    assert sol.V.shape == (dist.N + 2, m.n_assets, dist.N + 1)
    assert sol.pi.shape == (dist.N + 1, m.n_assets, dist.N + 1)
