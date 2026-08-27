from fastapi.testclient import TestClient

from horizon.dashboard.app import app
from horizon.dashboard.decisions import (
    costly_rows,
    episode_picker,
    episode_trace_table,
    trade_rows,
)
from horizon.env.market import CASH

client = TestClient(app)


def trade(episode=1, day=16, holding=1, came_from=0, belief=42.0, m_t=42.5,
          pi_star=1, q_loss=0.0, raw="BELIEF: 42\nACTION: SWITCH B", thought=180):
    # Column order matches Repository.recent_trades.
    return (episode, day, holding, 1, 1.0662, belief, holding, m_t, 0.01,
            pi_star, q_loss, thought, "ok", raw, came_from)


def costly(episode=2, day=30, holding=0, belief=90.0, m_t=35.5, pi_star=2, q_loss=0.05):
    return (episode, day, holding, 14, 1.10, belief, holding, m_t, 0.02,
            pi_star, q_loss, 200, "ok", "BELIEF: 90\nACTION: HOLD")


def test_trade_log_names_the_move_in_plain_language():
    html = trade_rows([trade(came_from=CASH, holding=0)])
    assert "bought A" in html


def test_trade_log_calls_a_move_to_cash_a_sale():
    html = trade_rows([trade(came_from=0, holding=CASH)])
    assert "sold to cash" in html


def test_trade_log_calls_the_opening_move_an_entry():
    assert "entered" in trade_rows([trade(came_from=None, holding=0)])


def test_trade_log_marks_agreement_with_the_optimum():
    assert "matched optimum" in trade_rows([trade(holding=1, pi_star=1)])


def test_trade_log_names_what_the_optimum_would_have_held():
    html = trade_rows([trade(holding=1, pi_star=2)])
    assert "optimum held C" in html


def test_trade_log_shows_the_belief_gap_and_grades_it():
    tight = trade_rows([trade(belief=42.0, m_t=42.5)])
    wide = trade_rows([trade(belief=90.0, m_t=42.5)])
    assert 'class="gap ok"' in tight
    assert 'class="gap bad"' in wide


def test_trade_log_quotes_the_models_own_words():
    html = trade_rows([trade(raw="BELIEF: 42\nACTION: SWITCH B")])
    assert "BELIEF: 42" in html


def test_trade_log_escapes_model_output():
    html = trade_rows([trade(raw="<script>alert(1)</script>")])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_trade_log_handles_no_trades():
    assert "No position changes recorded yet" in trade_rows([])


def test_costly_rows_flag_the_lost_wealth():
    html = costly_rows([costly(q_loss=0.0512)])
    assert "0.0512" in html


def test_costly_rows_handle_no_data():
    assert "No suboptimal decisions" in costly_rows([])


def test_episode_picker_marks_the_open_episode():
    episodes = [(3, 61, 1.40, 0.06, 2), (2, 12, 1.05, 0.41, 1)]
    html = episode_picker(episodes, selected=3)
    assert html.count("chip") >= 2
    assert "chip active" in html


def test_episode_trace_marks_the_days_the_position_changed():
    rows = [
        (1, 1, 0, 1, 1.0, 50.0, 0, 50.5, 0.01, 0, 0.0, 100, "ok", "", None),
        (1, 2, 0, 2, 1.01, 49.0, 0, 49.5, 0.01, 0, 0.0, 100, "ok", "", 0),
        (1, 3, 1, 1, 1.01, 48.0, 1, 48.5, 0.01, 1, 0.0, 100, "ok", "", 0),
    ]
    html = episode_trace_table(rows, 1)
    assert html.count('class="switch"') == 1


def test_decision_endpoints_are_served_as_fragments():
    for path in ("/decisions", "/decisions?episode=1"):
        response = client.get(path)
        assert response.status_code == 200
        assert "<html" not in response.text


def test_shell_carries_both_tabs():
    html = client.get("/").text
    assert 'data-tab="overview"' in html
    assert 'data-tab="decisions"' in html
    assert html.count('class="tabpanel"') == 2
