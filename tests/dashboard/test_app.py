from fastapi.testclient import TestClient

from horizon.dashboard.app import app
from horizon.dashboard.charts import (
    bar_chart,
    empty_chart,
    equity_chart,
    hazard_ribbon,
    holding_bands,
    line_chart,
)
from horizon.env.distributions import bimodal_exit, uniform_exit

client = TestClient(app)


def test_health_endpoint_reports_the_run():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "run_id" in body


def test_index_renders_without_a_database():
    # The dashboard has to come up before any worker has written a row, and must
    # not 500 when Postgres is unreachable.
    response = client.get("/")
    assert response.status_code == 200
    assert "Random-Horizon Trading" in response.text


def test_index_carries_every_panel():
    text = client.get("/").text
    for heading in (
        "Token throughput",
        "Deliberation by day",
        "When the agent traded",
        "Belief calibration",
        "Data quality",
        "Export",
    ):
        assert heading in text


def test_live_fragment_is_served_separately_from_the_shell():
    # The page polls this and swaps it in, so charts never blink and scroll
    # position survives an update.
    shell = client.get("/").text
    fragment = client.get("/live")
    assert fragment.status_code == 200
    assert 'class="tabpanel" data-tab="overview"' in shell
    assert "<html" not in fragment.text
    assert len(fragment.text) < len(shell)


def test_summary_api_degrades_gracefully_without_a_database():
    body = client.get("/api/summary").json()
    assert body["progress"] == {}
    assert body["errors"] == {}


def test_export_refuses_before_any_episode_completes():
    assert client.get("/export").status_code == 409


def test_page_loads_no_third_party_scripts():
    # Google Fonts is the single allowed external origin; nothing executable.
    text = client.get("/").text
    assert "cdn" not in text.lower()
    assert "<script src=" not in text


def test_page_defines_both_light_and_dark_palettes():
    text = client.get("/").text
    assert "prefers-color-scheme: dark" in text
    assert '[data-theme="dark"]' in text


def test_page_respects_reduced_motion():
    assert "prefers-reduced-motion" in client.get("/").text


def test_hazard_ribbon_draws_a_bar_per_hazardous_day():
    dist = bimodal_exit()
    hazard = [dist.hazard(t) for t in range(1, dist.N + 1)]
    svg = hazard_ribbon(hazard)
    # Bimodal is zero everywhere except day 10 and the late window.
    assert svg.count("<rect") == 11


def test_hazard_ribbon_places_the_fleet_marker():
    dist = uniform_exit(100)
    hazard = [dist.hazard(t) for t in range(1, 101)]
    assert "<circle" in hazard_ribbon(hazard, marker_day=37)
    assert "<circle" not in hazard_ribbon(hazard)


def test_hazard_ribbon_survives_no_data():
    assert "svg" in hazard_ribbon([])


def test_line_chart_handles_no_data():
    assert "waiting for data" in line_chart([])


def test_line_chart_skips_nan_points():
    svg = line_chart([("s", "--signal", [(1, 1.0), (2, float("nan")), (3, 2.0)])])
    assert "polyline" in svg


def test_line_chart_draws_event_markers():
    svg = line_chart([("s", "--signal", [(1, 1.0), (20, 2.0)])], markers=(11,))
    assert "d11" in svg


def test_bar_chart_renders_one_rect_per_nonzero_value():
    svg = bar_chart([4.0, 0.0, 8.0, 2.0])
    assert svg.count("<rect") == 3


def test_bar_chart_highlights_the_optimal_day():
    svg = bar_chart([1.0, 5.0, 1.0], highlight=1)
    assert "var(--signal)" in svg


def test_bar_chart_handles_no_data():
    assert "waiting for data" in bar_chart([])
    assert "waiting for data" in bar_chart([0, 0, 0])


def test_empty_chart_escapes_its_message():
    assert "&lt;script&gt;" in empty_chart(100, 50, "<script>")


# --- equity curves --------------------------------------------------------- #


def test_holding_bands_collapse_a_day_sequence_into_runs():
    bands, entries = holding_bands([(1, 3), (2, 3), (3, 0), (4, 0), (5, 2)])
    assert bands == [(1, 3, 3), (3, 5, 0), (5, 5, 2)]
    assert entries == [(1, 3), (3, 0), (5, 2)]


def test_holding_bands_report_a_single_unchanged_run():
    bands, entries = holding_bands([(1, 0), (2, 0), (3, 0)])
    assert bands == [(1, 3, 0)]
    assert entries == [(1, 0)]


def test_equity_chart_marks_every_entry_on_the_curve():
    wealth = [(d, 1.0 + d / 100) for d in range(1, 21)]
    svg = equity_chart(wealth, entries=[(1, 3), (5, 0), (16, 2)])
    # One flag per entry, each labelled with the day and what was entered.
    assert "d1&#8594;CASH" in svg
    assert "d5&#8594;A" in svg
    assert "d16&#8594;C" in svg
    assert svg.count("<circle") == 3


def test_equity_chart_draws_the_optimal_policy_behind_the_agent():
    svg = equity_chart([(1, 1.0), (2, 1.02)], reference=[(1, 1.01), (2, 1.03)])
    assert 'class="reference"' in svg
    assert "optimal policy" in svg


def test_equity_chart_omits_the_reference_legend_when_there_is_none():
    assert "optimal policy" not in equity_chart([(1, 1.0), (2, 1.02)])


def test_equity_chart_marks_the_day_the_exit_landed():
    svg = equity_chart([(d, 1.0) for d in range(1, 31)], exit_day=22)
    assert "liquidated d22" in svg


def test_equity_chart_ignores_an_exit_beyond_the_plotted_days():
    assert "liquidated" not in equity_chart([(1, 1.0), (2, 1.0)], exit_day=90)


def test_equity_chart_draws_the_holding_strip():
    svg = equity_chart([(d, 1.0 + d / 50) for d in range(1, 41)],
                       bands=[(1, 16, 0), (16, 40, 2)])
    assert "var(--asset-a)" in svg
    assert "var(--asset-c)" in svg


def test_equity_chart_shows_break_even_when_the_curve_crosses_it():
    below = equity_chart([(1, 1.02), (2, 0.94)])
    assert "break-even" in below


def test_equity_chart_handles_no_data():
    assert "waiting for" in equity_chart([])


def test_equity_chart_survives_a_flat_curve():
    # Every day identical would make the y-range zero and divide by nothing.
    assert "polyline" in equity_chart([(d, 1.0) for d in range(1, 10)])


def test_overview_carries_the_fleet_equity_panel():
    assert "Equity curve" in client.get("/").text


def test_episode_view_leads_with_the_pnl_curve():
    text = client.get("/decisions?episode=1").text
    assert "P&amp;L and entries" in text


def test_the_optimal_reference_curve_is_solvable_without_a_database():
    # Drawn behind every equity curve, so it must not depend on the run's rows.
    from horizon.dashboard.app import _optimal_curve

    curve = _optimal_curve("uniform", 100)
    assert len(curve) == 100
    assert curve[0][0] == 1
    # The optimum enters A, pays the switch cost on the crossover day, then
    # compounds in C: terminal wealth must clear the day-1 entry comfortably.
    assert curve[-1][1] > curve[0][1] * 1.5


def test_the_reference_curve_is_clipped_to_the_days_actually_played():
    from horizon.dashboard.app import _reference

    assert [d for d, _ in _reference("uniform", 100, last_day=12)] == list(range(1, 13))


def test_the_crossover_marker_is_read_off_the_solved_policy():
    # Hardcoding it was right for the uniform cell and would be wrong for any
    # other: the marker has to follow the policy, not a constant.
    from horizon.dashboard.app import _crossover, _optimal_path

    day = _crossover({"exit": "uniform"}, 100)
    holdings = _optimal_path("uniform", 100)[0]
    assert holdings[day - 1] != holdings[day - 2]
    assert all(holdings[d] == holdings[0] for d in range(day - 1))


# --- cell selection -------------------------------------------------------- #


def test_an_unknown_run_falls_back_to_the_default_cell():
    # ?run= comes straight off the URL bar; it must not be able to point the
    # dashboard at a run that does not exist.
    from horizon.dashboard.app import DEFAULT_RUN, _resolve

    assert _resolve("../../etc/passwd") == DEFAULT_RUN
    assert _resolve(None) == DEFAULT_RUN
    assert _resolve("") == DEFAULT_RUN


def test_every_fragment_endpoint_accepts_a_run():
    for path in ("/live?run=t1-1", "/decisions?run=t1-1", "/decisions?episode=1&run=t1-1"):
        assert client.get(path).status_code == 200


def test_the_shell_tells_the_script_which_cell_it_is_showing():
    # Without this the five-second poll would drift back to the default run.
    assert 'data-run=' in client.get("/").text


def test_the_selector_is_hidden_until_there_is_more_than_one_cell():
    from horizon.dashboard.render import _cell_chips

    one = [{"run_id": "t1-1", "cell": {}, "done": 200, "total": 200}]
    assert _cell_chips(one, "t1-1") == ""
    two = one + [{"run_id": "t1-2", "cell": {"exit": "bimodal", "info": "K"},
                  "done": 5, "total": 200}]
    chips = _cell_chips(two, "t1-1")
    assert "t1-2" in chips and "bimodal" in chips
    assert chips.count("cell-chip active") == 1


def test_the_selector_reports_progress_per_cell():
    from horizon.dashboard.render import _cell_chips

    runs = [{"run_id": "a", "cell": {}, "done": 50, "total": 200},
            {"run_id": "b", "cell": {}, "done": 0, "total": 0}]
    chips = _cell_chips(runs, "a")
    assert "25%" in chips
    assert "queued" in chips


def test_the_export_link_carries_the_cell_being_viewed():
    assert "/export?run=" in client.get("/").text
