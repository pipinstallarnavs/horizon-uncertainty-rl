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


def test_headline_distributions_pass_the_non_collapse_check():
    # DESIGN.md P1: if the optimal policy does not depend on t, the environment
    # tests nothing temporal and no API budget should be spent on it.
    m = default_market()
    for dist in (uniform_exit(100), bimodal_exit()):
        sol = solve(m, dist)
        assert time_varying_fraction(sol, dist) >= 0.20
        assert_non_collapse(sol, dist)


def test_the_geometric_control_nearly_fails_the_check():
    # P1 validates the checker itself by requiring the memoryless control to fall
    # short: constant hazard makes the optimum stationary, so it must NOT clear
    # the same bar the headline distributions clear.
    m = default_market()
    dist = geometric_exit(100)
    sol = solve(m, dist)
    assert time_varying_fraction(sol, dist) < 0.20
    with pytest.raises(AssertionError, match="collapse"):
        assert_non_collapse(sol, dist)


def test_the_geometric_rate_must_avoid_truncation_mass():
    # At p=0.02 the residual atom on day N holds 13.5% of the mass and the
    # "memoryless" control becomes strongly time-dependent, varying MORE than the
    # uniform condition it is meant to contrast with. The default rate avoids it.
    m = default_market()
    truncated = geometric_exit(100, p=0.02)
    clean = geometric_exit(100)
    assert truncated.pmf[100] > 0.10
    assert clean.pmf[100] < 0.001
    assert time_varying_fraction(solve(m, clean), clean) < time_varying_fraction(
        solve(m, truncated), truncated
    )
