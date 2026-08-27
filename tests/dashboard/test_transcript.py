"""The episode transcript, and the entry/exit/re-entry ledger built from it."""
from horizon.dashboard.transcript import (
    event_ledger,
    events,
    refusal_markers,
    refusal_rows,
    transcript,
    walk,
)
from horizon.env.market import CASH

A, B, C = 0, 1, 2


def day(n, holding, requested=None, error="ok", belief=40.0, m_t=40.5,
        thought_text="thinking about it", wealth=1.0, age=1):
    """One agent_days row in Repository.TRACE_COLUMNS order."""
    return (1, n, holding, age, wealth, belief,
            holding if requested is None else requested,
            m_t, 0.01, holding, 0.0, 300, error, "ACTION: HOLD", thought_text)


def test_the_first_move_out_of_cash_is_an_entry():
    walked = walk([day(1, CASH), day(2, A)])
    assert [w["event"] for w in walked] == ["held", "entered"]


def test_holding_the_same_asset_is_not_an_event():
    # Day 1 out of cash into A is itself an entry; the days after it are not.
    walked = walk([day(1, A), day(2, A), day(3, A)])
    assert [w["event"] for w in walked] == ["entered", "held", "held"]
    assert len(events(walked)) == 1


def test_an_episode_that_never_left_cash_has_no_events():
    assert events(walk([day(n, CASH) for n in range(1, 5)])) == []


def test_a_move_between_risky_assets_is_a_rotation():
    walked = walk([day(1, A), day(2, C)])
    assert walked[1]["event"] == "rotated"
    assert walked[1]["came_from"] == A


def test_a_move_back_to_cash_is_an_exit():
    walked = walk([day(1, A), day(2, CASH)])
    assert walked[1]["event"] == "exited"


def test_leaving_an_asset_marks_it_spent_from_the_next_day():
    walked = walk([day(1, A), day(2, C), day(3, C)])
    assert walked[0]["spent"] == ()
    assert walked[1]["spent"] == ()          # burned on the way out of this day
    assert walked[2]["spent"] == (A,)
    assert walked[2]["spent_on"][A] == 2
    assert A not in walked[2]["available"]


def test_cash_is_never_spent():
    walked = walk([day(1, CASH), day(2, A), day(3, CASH), day(4, B)])
    assert CASH in walked[-1]["available"]
    assert CASH not in walked[-1]["spent"]


def test_a_refused_re_entry_is_its_own_event():
    # The rules cancelled the move, so `holding` never changed -- the only trace
    # left is the requested action and the error class.
    walked = walk([
        day(1, A),
        day(2, C),
        day(3, C),
        day(4, C, requested=A, error="illegal_action"),
    ])
    assert walked[3]["event"] == "refused"
    assert walked[3]["requested"] == A
    assert refusal_markers(walked) == [(4, A)]


def test_the_ledger_names_what_it_tried_to_buy_back_and_when_it_left():
    walked = walk([day(1, A), day(2, C), day(3, C),
                   day(4, C, requested=A, error="illegal_action")])
    html = event_ledger(walked)
    assert "tried to re-enter <b>A</b>" in html
    assert "left on day 2" in html
    assert "stay in C" in html


def test_the_ledger_reports_an_episode_that_never_moved():
    walked = walk([day(1, CASH), day(2, CASH)])
    assert "never changed position" in event_ledger(walked)


def test_the_transcript_renders_every_day_including_the_quiet_ones():
    walked = walk([day(n, A) for n in range(1, 6)])
    html = transcript(walked)
    assert html.count("<article") == 5
    assert html.count('class="trace"') == 5


def test_the_transcript_shows_the_reasoning_verbatim():
    walked = walk([day(1, A, thought_text="A compounds until day 15. Holding.")])
    assert "A compounds until day 15. Holding." in transcript(walked)


def test_the_transcript_escapes_model_output():
    walked = walk([day(1, A, thought_text="<script>alert(1)</script>")])
    html = transcript(walked)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_the_transcript_says_when_a_cell_predates_trace_capture():
    walked = walk([day(1, A, thought_text="")])
    assert "before traces were being captured" in transcript(walked)


def test_each_transcript_day_is_keyed_so_expansion_survives_the_poll():
    walked = walk([day(1, A), day(2, A)])
    html = transcript(walked)
    assert 'data-key="1:1"' in html and 'data-key="1:2"' in html


def test_the_transcript_grades_the_belief_against_the_truth():
    walked = walk([day(1, A, belief=90.0, m_t=40.5)])
    assert 'class="gap bad"' in transcript(walked)
    walked = walk([day(1, A, belief=41.0, m_t=40.5)])
    assert 'class="gap ok"' in transcript(walked)


def test_the_transcript_handles_an_episode_with_no_rows():
    assert "No days recorded" in transcript([])


def test_the_refusal_table_says_so_when_nothing_was_refused():
    assert "No agent has tried to re-enter" in refusal_rows([])


def test_the_refusal_table_names_the_asset_that_was_denied():
    html = refusal_rows([day(4, C, requested=A, error="illegal_action")])
    assert "asked for <b>A</b>" in html
    assert "held C" in html
