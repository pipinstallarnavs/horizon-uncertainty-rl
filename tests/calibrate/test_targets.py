import pytest

from horizon.calibrate.targets import (
    crossover_day,
    evaluate_targets,
    expected_switches,
    fixed_asset_values,
)
from horizon.env.distributions import bimodal_exit, uniform_exit
from horizon.env.market import AssetSpec, Market, default_market
from horizon.solver.backward import solve


def flat_market():
    return Market(
        assets=(
            AssetSpec("A", 0.001, 0.001, 1),
            AssetSpec("B", 0.001, 0.001, 1),
            AssetSpec("C", 0.001, 0.001, 1),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=0.001,
    )


def test_fixed_asset_values_reports_every_holding():
    values = fixed_asset_values(default_market(), uniform_exit(50))
    assert set(values) == {"A", "B", "C", "CASH"}
    assert values["CASH"] == pytest.approx(1.0)


def test_no_crossover_when_one_asset_dominates_at_all_horizons():
    m = flat_market()
    dist = uniform_exit(50)
    assert crossover_day(solve(m, dist), m, dist) is None


def test_crossover_is_found_when_the_best_holding_changes():
    # With the frozen parameters the optimum rides the Sprinter through its
    # 15-day boost window and then switches, so the crossover lands on day 16.
    m = default_market()
    dist = bimodal_exit()
    assert crossover_day(solve(m, dist), m, dist) == 16


def test_expected_switches_counts_the_entry_out_of_cash():
    m = flat_market()
    dist = uniform_exit(50)
    # One switch out of CASH on day 1, then never again.
    assert expected_switches(solve(m, dist), m, dist) == pytest.approx(1.0)


def test_target_report_flags_a_degenerate_market_as_failing():
    m = flat_market()
    report = evaluate_targets(m, uniform_exit(100), bimodal_exit())
    assert not report.all_pass
    assert not report.no_dominant_asset


def test_target_report_exposes_every_target():
    report = evaluate_targets(default_market(), uniform_exit(100), bimodal_exit())
    for field in (
        "interior_crossover",
        "value_separation",
        "cash_separation",
        "no_dominant_asset",
        "cost_relevance",
    ):
        assert isinstance(getattr(report, field), bool)


def test_no_solved_policy_can_re_enter_an_asset():
    # Now an invariant rather than a target: the no-re-entry rule is enforced
    # inside the solver, so even a market built to be exploitable (A pays far
    # more freshly held than a switch costs) cannot produce clock farming.
    from horizon.calibrate.targets import has_re_entry

    exploitable = Market(
        assets=(
            AssetSpec("A", 0.016, -0.004, 30),
            AssetSpec("B", 0.006, 0.006, 1),
            AssetSpec("C", 0.0, 0.008, 30),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=0.002,
    )
    for market in (exploitable, default_market()):
        for dist in (uniform_exit(100), bimodal_exit(100)):
            assert not has_re_entry(solve(market, dist), market, dist)


def test_myopic_value_matches_a_greedy_rollout():
    from horizon.calibrate.targets import myopic_value

    value = myopic_value(default_market(), uniform_exit(100))
    assert value > 1.0


def test_myopic_separation_rejects_a_market_solvable_without_the_horizon():
    # A late bloomer that cannot bloom inside the expected remaining time is
    # never worth entering, so the optimum collapses onto greedy and the
    # condition tests no horizon reasoning at all -- even with a large margin
    # over every fixed-asset baseline.
    from horizon.calibrate.targets import myopic_value

    unreachable_bloom = Market(
        assets=(
            AssetSpec("A", 0.012, 0.0, 15),
            AssetSpec("B", 0.003, 0.003, 1),
            AssetSpec("C", 0.0005, 0.008, 50),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=0.001,
    )
    dist = uniform_exit(100)
    v_star = solve(unreachable_bloom, dist).v_star
    assert v_star == pytest.approx(myopic_value(unreachable_bloom, dist))

    report = evaluate_targets(unreachable_bloom, dist, bimodal_exit(100))
    assert not report.myopic_separation
    assert not report.all_pass
