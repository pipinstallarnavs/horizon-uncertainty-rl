import pytest

from horizon.env.distributions import bimodal_exit, geometric_exit, uniform_exit
from horizon.env.market import default_market
from horizon.solver.backward import solve
from horizon.solver.verify import (
    assert_non_collapse,
    brute_force_value,
    monte_carlo_value,
    time_varying_fraction,
)


def test_brute_force_matches_the_solver_exactly_at_n_eight():
    # The decisive correctness check: an independent exhaustive search.
    m = default_market()
    dist = uniform_exit(8)
    sol = solve(m, dist)
    best, _ = brute_force_value(m, dist)
    assert sol.v_star == pytest.approx(best, rel=1e-12)


def test_brute_force_matches_the_solver_on_a_bimodal_case():
    m = default_market()
    dist = bimodal_exit(N=8, early=(2, 2), late=(7, 8))
    sol = solve(m, dist)
    best, _ = brute_force_value(m, dist)
    assert sol.v_star == pytest.approx(best, rel=1e-12)


def test_brute_force_refuses_intractable_sizes():
    m = default_market()
    with pytest.raises(ValueError, match="brute force"):
        brute_force_value(m, uniform_exit(20))


def test_monte_carlo_rollout_matches_the_analytic_value():
    m = default_market()
    dist = uniform_exit(100)
    sol = solve(m, dist)
    mean, stderr = monte_carlo_value(m, dist, sol, n=20000, seed=20260814)
    assert abs(mean - sol.v_star) < 4 * stderr


def test_the_checker_separates_time_varying_policies_from_stationary_ones():
    # DESIGN.md P1 validates the checker by requiring it to (nearly) fail on the
    # geometric control, whose constant hazard makes the optimum stationary.
    m = default_market()
    bimodal = bimodal_exit()
    geometric = geometric_exit(100, p=0.02)
    varying = time_varying_fraction(solve(m, bimodal), bimodal)
    stationary = time_varying_fraction(solve(m, geometric), geometric)
    assert stationary < 0.05
    assert varying > 3 * stationary


def test_non_collapse_raises_on_a_stationary_policy():
    m = default_market()
    dist = geometric_exit(100, p=0.02)
    with pytest.raises(AssertionError, match="collapse"):
        assert_non_collapse(solve(m, dist), dist)


def test_non_collapse_passes_when_the_threshold_is_met():
    m = default_market()
    dist = bimodal_exit()
    assert_non_collapse(solve(m, dist), dist, threshold=0.10)


def test_uncalibrated_uniform_condition_has_no_crossover_yet():
    # Records the state of play: with PROVISIONAL parameters the optimum under
    # uniform exit is buy-and-hold B for all 100 days, so the uniform condition
    # tests nothing temporal. Calibration (targets 1 and 4) must fix this before
    # any LLM run; this test is expected to be replaced once parameters freeze.
    m = default_market()
    dist = uniform_exit(100)
    assert time_varying_fraction(solve(m, dist), dist) < 0.05
