import pytest

from horizon.env.market import default_market
from horizon.report import describe_path, environment_report, verify_all


def test_report_covers_every_distribution():
    report = environment_report(default_market(), N=40)
    assert set(report) == {"uniform", "bimodal", "geometric", "sawtooth"}


def test_report_includes_the_curves_needed_for_figure_one():
    report = environment_report(default_market(), N=40)
    uniform = report["uniform"]
    assert len(uniform["hazard"]) == 40
    assert len(uniform["mean_remaining"]) == 40
    assert uniform["hazard"][-1] == pytest.approx(1.0)


def test_report_records_the_non_collapse_fraction():
    report = environment_report(default_market(), N=40)
    assert report["uniform"]["time_varying_fraction"] > report["geometric"]["time_varying_fraction"]


def test_describe_path_compresses_runs():
    assert describe_path([0, 0, 0, 1, 1]) == "d1-3:A  d4-5:B"


def test_verify_all_passes_on_the_default_market():
    checks = verify_all(default_market())
    assert any("brute force" in c for c in checks)
    assert any("Monte Carlo" in c for c in checks)
