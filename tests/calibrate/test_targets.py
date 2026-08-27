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
    # Under the bimodal exit the optimum is B then C, switching at day 11.
    m = default_market()
    dist = bimodal_exit()
    assert crossover_day(solve(m, dist), m, dist) == 11


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


def test_re_entry_detects_the_age_clock_exploit():
    # A market where A pays far more when freshly held than a switch costs. The
    # optimum bounces A -> B -> A to restart the boost, which expected-switch
    # counts miss because uniform exit halves the visible count.
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
    dist = uniform_exit(100)
    sol = solve(exploitable, dist)
    assert has_re_entry(sol, exploitable, dist)
    assert expected_switches(sol, exploitable, dist) <= 4.0  # target 4 is fooled


def test_no_re_entry_on_a_monotone_market():
    from horizon.calibrate.targets import has_re_entry

    m = default_market()
    dist = uniform_exit(100)
    assert not has_re_entry(solve(m, dist), m, dist)
