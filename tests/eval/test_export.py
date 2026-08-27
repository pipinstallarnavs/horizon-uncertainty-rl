import io
import json
import zipfile

import pytest

from horizon.env.distributions import bimodal_exit
from horizon.eval.export import compute_tables, export_bundle, html_report, records_from_rows
from horizon.eval.metrics import EpisodeRecord


def sample_records(n=30):
    return [
        EpisodeRecord(
            episode=i,
            true_exit_day=95,
            terminal_wealth=1.2 + 0.001 * i,
            holdings=tuple([0] * 15 + [2] * 80),
            beliefs=tuple(float(85 - d) for d in range(95)),
            switch_days=(1, 16),
        )
        for i in range(n)
    ]


def test_records_are_rebuilt_from_database_rows():
    rows = [(0, 50, 1.1, [0, 1], [40.0, 39.0], [1])]
    records = records_from_rows(rows)
    assert records[0].episode == 0
    assert records[0].holdings == (0, 1)


def test_records_tolerate_null_arrays():
    records = records_from_rows([(0, 50, 1.1, [0], None, None)])
    assert records[0].beliefs == ()
    assert records[0].switch_days == ()


def test_compute_tables_reports_every_headline_number():
    tables = compute_tables(sample_records(), bimodal_exit(), v_star=1.29321)
    for key in (
        "regret",
        "clock_coefficient",
        "mean_switches",
        "trade_alignment",
        "event_timing",
        "mean_belief_error",
    ):
        assert key in tables
    assert tables["episodes"] == 30


def test_regret_matches_v_star_minus_mean_wealth():
    records = sample_records(10)
    tables = compute_tables(records, bimodal_exit(), v_star=1.5)
    expected = 1.5 - sum(r.terminal_wealth for r in records) / 10
    assert tables["regret"] == pytest.approx(expected)


def test_html_report_is_self_contained():
    tables = compute_tables(sample_records(), bimodal_exit(), v_star=1.29)
    html = html_report("t1-4", {"exit": "bimodal"}, tables, bimodal_exit())
    assert "<svg" in html
    assert "cdn" not in html.lower()
    assert "src=" not in html


def test_html_report_states_the_single_model_limitation():
    tables = compute_tables(sample_records(), bimodal_exit(), v_star=1.29)
    html = html_report("t1-4", {}, tables, bimodal_exit())
    assert "Cross-provider generality is" in html


def test_bundle_contains_report_csvs_and_summary(tmp_path):
    path = tmp_path / "bundle.zip"
    export_bundle(str(path), "t1-4", {"exit": "bimodal"}, sample_records(),
                  bimodal_exit(), v_star=1.29321)
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        assert "t1-4-report.html" in names
        assert "t1-4-episodes.csv" in names
        assert "t1-4-belief-calibration.csv" in names
        summary = json.loads(bundle.read("t1-4-summary.json"))
    assert summary["episodes"] == 30


def test_bundle_csv_lets_regret_be_recomputed(tmp_path):
    # The reproducibility claim: every reported number recomputable from the
    # bundle alone, without the database.
    import csv

    path = tmp_path / "bundle.zip"
    records = sample_records(20)
    export_bundle(str(path), "r", {}, records, bimodal_exit(), v_star=1.5)
    with zipfile.ZipFile(path) as bundle:
        rows = list(csv.DictReader(io.StringIO(bundle.read("r-episodes.csv").decode())))
    recomputed = sum(float(r["regret"]) for r in rows) / len(rows)
    assert recomputed == pytest.approx(1.5 - sum(r.terminal_wealth for r in records) / 20)
