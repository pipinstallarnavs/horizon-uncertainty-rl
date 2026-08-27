from fastapi.testclient import TestClient

from horizon.dashboard.app import app
from horizon.dashboard.charts import bar_chart, empty_chart, line_chart

client = TestClient(app)


def test_health_endpoint_reports_the_run():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "run_id" in body


def test_index_renders_without_a_database():
    # The dashboard has to come up before any worker has written a row, and it
    # must not 500 if Postgres is unreachable.
    response = client.get("/")
    assert response.status_code == 200
    assert "Random-Horizon Trading" in response.text


def test_index_includes_the_token_telemetry_panels():
    text = client.get("/").text
    assert "Token throughput" in text
    assert "thought tokens" in text
    assert "Deliberation effort by day" in text


def test_summary_api_degrades_gracefully_without_a_database():
    body = client.get("/api/summary").json()
    assert body["progress"] == {}
    assert body["errors"] == {}


def test_page_is_self_contained():
    # No CDN, no external fonts: the VM is locked down and behind a tunnel.
    text = client.get("/").text
    assert "http://" not in text.replace("http-equiv", "")
    assert "cdn" not in text.lower()


def test_page_defines_both_light_and_dark_palettes():
    text = client.get("/").text
    assert "prefers-color-scheme: dark" in text
    assert '[data-theme="dark"]' in text


def test_line_chart_handles_no_data():
    assert "no data yet" in line_chart([])


def test_line_chart_skips_nan_points():
    svg = line_chart([("s", "--c1", [(1, 1.0), (2, float("nan")), (3, 2.0)])])
    assert "polyline" in svg


def test_line_chart_draws_event_markers():
    svg = line_chart([("s", "--c1", [(1, 1.0), (20, 2.0)])], markers=(11,))
    assert "day 11" in svg


def test_bar_chart_renders_bars():
    svg = bar_chart([1, 2, 3], [4.0, 8.0, 2.0])
    assert svg.count("<rect") == 3


def test_bar_chart_handles_no_data():
    assert "no data yet" in bar_chart([], [])


def test_empty_chart_escapes_its_message():
    assert "&lt;script&gt;" in empty_chart(100, 50, "<script>")
