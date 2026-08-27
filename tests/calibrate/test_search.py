import json

import pytest

from horizon.calibrate.search import DEFAULT_GRID, candidate_markets, freeze, search


def tiny_grid():
    return {
        "a_rate": [0.003],
        "b_rate": [0.006],
        "c_hi": [0.008],
        "c_k": [20],
    }


def test_candidate_markets_enumerates_the_grid():
    grid = dict(tiny_grid(), b_rate=[0.005, 0.006])
    assert len(list(candidate_markets(grid))) == 2


def test_candidate_markets_builds_the_named_assets_in_fixed_order():
    market = next(iter(candidate_markets(tiny_grid())))
    assert [a.name for a in market.assets] == ["A", "B", "C", "CASH"]


def test_every_candidate_market_is_monotone_in_holding_age():
    # The hard constraint: no asset may pay more when freshly held, or the
    # optimum farms the clock reset and the policy stops depending on t.
    for market in candidate_markets(DEFAULT_GRID):
        for asset in market.assets:
            assert asset.late_rate >= asset.early_rate


def test_search_returns_only_feasible_combinations():
    results = search(tiny_grid(), N=40)
    assert all(r["all_pass"] for r in results)


def test_search_results_are_sorted_by_margin_descending():
    results = search(DEFAULT_GRID, N=40)
    margins = [r["value_margin"] for r in results]
    assert margins == sorted(margins, reverse=True)


def test_freeze_writes_the_feasible_set_and_returns_the_winner(tmp_path):
    results = search(DEFAULT_GRID, N=40)
    if not results:
        pytest.skip("no feasible set at N=40; the real calibration runs at N=100")
    out = tmp_path / "feasible.json"
    winner = freeze(results, str(out))
    written = json.loads(out.read_text())
    assert written["feasible_set"]
    assert winner == results[0]


def test_freeze_refuses_an_empty_feasible_set(tmp_path):
    with pytest.raises(ValueError, match="no feasible"):
        freeze([], str(tmp_path / "feasible.json"))
