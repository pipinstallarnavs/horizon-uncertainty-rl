from fastapi.testclient import TestClient

from horizon.dashboard.app import app
from horizon.dashboard.charts import bar_chart, empty_chart, hazard_ribbon, line_chart
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
    assert 'id="live"' in shell
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
