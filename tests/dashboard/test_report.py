"""The report page and the glossary."""
from fastapi.testclient import TestClient

from horizon.dashboard.app import app
from horizon.dashboard.report import (
    GLOSSARY,
    cell_table,
    decode,
    extremes_table,
    findings,
    glossary_body,
)

client = TestClient(app)


def summary(**over):
    base = {"episodes": 40, "wealth": 1.41, "regret": 0.052, "first_trade": 17.4,
            "switches": 1.1, "exit_day": 50.2}
    base.update(over)
    return base


def run(run_id="t2-1", exit_name="uniform", info="K", done=40, total=200):
    return {"run_id": run_id, "cell": {"exit": exit_name, "info": info},
            "status": "running", "done": done, "total": total,
            "summary": summary(), "has_traces": True}


# --- decoding -------------------------------------------------------------- #


def test_a_cell_name_is_decoded_into_plain_language():
    assert decode("t2-5", {"exit": "uniform", "info": "O"}) == "uniform exit, oracle"
    assert decode("t2-2", {"exit": "bimodal", "info": "K"}) == "bimodal exit, known distribution"
    assert decode("t2-4", {"exit": "uniform", "info": "U"}) == "uniform exit, unknown distribution"


def test_an_unrecognised_cell_still_decodes_to_something_readable():
    assert "sawtooth" in decode("x", {"exit": "sawtooth", "info": "K"})


# --- glossary -------------------------------------------------------------- #


def test_the_glossary_decodes_every_run_in_the_database():
    html = glossary_body([run("t2-5", "uniform", "O"), run("t1-2", "bimodal", "K")])
    assert "t2-5" in html and "t1-2" in html
    assert "Oracle" in html and "Bimodal" in html


def test_the_glossary_explains_the_generation_prefix():
    html = glossary_body([run()])
    assert "512-token output budget" in html
    assert "Nothing was\n      deleted" in html or "Nothing was deleted" in html


def test_the_glossary_defines_the_load_bearing_terms():
    html = glossary_body([run()])
    for term in ("V*", "Regret", "Hazard", "No-re-entry rule", "Re-entry refused",
                 "Mean remaining", "Common random numbers", "Reasoning trace"):
        assert term in html


def test_every_glossary_entry_is_individually_addressable():
    html = glossary_body([run()])
    for slug, _, _ in GLOSSARY:
        assert f'id="g-{slug}"' in html


def test_the_glossary_is_honest_that_traces_are_summaries():
    html = glossary_body([run()])
    assert "no current Gemini model returns those" in html


def test_the_glossary_links_each_cell_to_itself():
    assert 'href="/?run=t2-5"' in glossary_body([run("t2-5")])


# --- findings -------------------------------------------------------------- #


def test_findings_say_how_late_the_first_trade_was():
    html = findings("t2-1", {}, summary(first_trade=17.4), 16, 0, {"ok": 100})
    assert "day 17.4" in html
    assert "1.4 days late" in html


def test_findings_call_an_on_time_first_trade_optimal():
    html = findings("t2-1", {}, summary(first_trade=16.2), 16, 0, {"ok": 100})
    assert "essentially on the optimal day" in html


def test_findings_report_an_early_trade_as_early():
    html = findings("t2-1", {}, summary(first_trade=11.0), 16, 0, {"ok": 100})
    assert "5.0 days early" in html


def test_findings_treat_no_refusals_as_a_result_not_an_absence():
    html = findings("t2-1", {}, summary(), 16, 0, {"ok": 100})
    assert "tracked correctly" in html
    assert "flagged" not in html


def test_findings_flag_refusals_when_they_happen():
    html = findings("t2-1", {}, summary(), 16, 3, {"ok": 100})
    assert "3 attempted re-entries" in html
    assert "flagged" in html


def test_findings_wait_for_the_first_episode():
    assert "No episode has finished" in findings("t2-1", {}, summary(episodes=0), 16, 0, {})


def test_every_finding_links_to_its_evidence():
    html = findings("t2-1", {}, summary(), 16, 2, {"ok": 100})
    assert html.count("data-goto=") >= 4


# --- tables ---------------------------------------------------------------- #


def test_the_cell_table_marks_the_one_being_viewed():
    html = cell_table([run("t2-1"), run("t2-2", "bimodal")], active="t2-2")
    assert html.count('class="here"') == 1
    assert 'href="/?run=t2-1"' in html


def test_the_cell_table_says_which_cells_carry_traces():
    rows = [dict(run("t1-1"), has_traces=False), run("t2-1")]
    html = cell_table(rows, active="t2-1")
    assert ">no<" in html and ">yes<" in html


def test_the_cell_table_survives_a_cell_with_no_episodes_yet():
    empty = dict(run("t2-9"), summary={"episodes": 0})
    assert "&mdash;" in cell_table([empty], active="t2-9")


def test_extremes_link_each_episode_to_its_transcript():
    rows = [("best", 12, 61, 1.52, -0.02, 1, 16), ("worst", 3, 9, 1.02, 0.44, 2, 31)]
    html = extremes_table(rows, "t2-1")
    assert 'href="/decisions?episode=12&amp;run=t2-1"' in html
    assert 'data-goto="decisions"' in html
    assert "tag-best" in html and "tag-worst" in html


def test_extremes_handle_an_episode_that_never_traded():
    html = extremes_table([("worst", 7, 4, 1.00, 0.46, 0, None)], "t2-1")
    assert "&mdash;" in html


def test_extremes_handle_no_data():
    assert "No finished episodes" in extremes_table([], "t2-1")


# --- endpoints ------------------------------------------------------------- #


def test_report_and_glossary_are_served_as_fragments():
    for path in ("/report", "/glossary", "/report?run=t2-1", "/glossary?run=t2-1"):
        response = client.get(path)
        assert response.status_code == 200
        assert "<html" not in response.text


def test_the_shell_ships_all_four_panels():
    html = client.get("/").text
    assert html.count('class="tabpanel"') == 4
    assert "Every term on this dashboard" in html
