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
    actions = [sol.pi[t, CASH, 0, 0] for t in range(1, 40)]
    assert len(set(actions)) == 1


def _action_change_fraction(sol, t):
    """Fraction of (holding, age, mask) states whose action differs from day t-1."""
    changed = total = 0
    for k in range(4):
        for mask in range(8):
            for age in range(0, t - 1):
                total += 1
                if sol.pi[t - 1, k, age, mask] != sol.pi[t, k, age, mask]:
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
    spread = bimodal_exit(early=(1, 10), late=(91, 100))
    assert _action_change_fraction(solve(m, concentrated), 11) > 0.2
    assert _action_change_fraction(solve(m, spread), 11) == 0.0


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


def test_the_optimal_path_never_re_enters_an_asset():
    # The no-re-entry rule is what makes the decaying Sprinter safe. Without it
    # the optimum farms A's clock reset (A -> B -> A forever) and the policy
    # stops depending on t at all.
    m = default_market()
    for dist in (uniform_exit(100), bimodal_exit(), geometric_exit(100)):
        holdings = optimal_holdings(solve(m, dist), m, 100)
        left = set()
        previous = CASH
        for holding in holdings:
            if holding != previous:
                assert holding not in left
                if previous != CASH:
                    left.add(previous)
                previous = holding


def test_the_default_market_is_not_degenerate():
    sol = solve(default_market(), bimodal_exit())
    assert any(_action_change_fraction(sol, t) > 0.0 for t in range(2, 101))


def test_the_memoryless_control_holds_something_different():
    # Under uniform and bimodal exit there is time for the late bloomer to pay
    # off, so the optimum rides the Sprinter and then switches into C. Under the
    # geometric control the expected horizon is far shorter, so the steady
    # compounder wins instead. The exit distribution changes what is held, which
    # is the whole premise of the study.
    m = default_market()
    uniform_path = optimal_holdings(solve(m, uniform_exit(100)), m, 100)
    bimodal_path = optimal_holdings(solve(m, bimodal_exit()), m, 100)
    geometric_path = optimal_holdings(solve(m, geometric_exit(100)), m, 100)

    assert uniform_path[-1] == m.index("C")
    assert bimodal_path[-1] == m.index("C")
    assert geometric_path[-1] == m.index("B")


def test_every_condition_rides_the_sprinter_through_its_boost_window():
    m = default_market()
    boost = m.assets[0].breakpoint
    for dist in (uniform_exit(100), bimodal_exit(), geometric_exit(100)):
        path = optimal_holdings(solve(m, dist), m, 100)
        assert path[: boost] == [0] * boost
        assert path[boost] != 0


def test_solution_shapes_are_consistent():
    m = default_market()
    dist = uniform_exit(30)
    sol = solve(m, dist)
    assert sol.V.shape == (dist.N + 2, m.n_assets, dist.N + 1, 8)
    assert sol.pi.shape == (dist.N + 1, m.n_assets, dist.N + 1, 8)
