import json

import pytest

from horizon.calibrate.search import DEFAULT_GRID, candidate_markets, freeze, search


def tiny_grid():
    return {
        "a_hi": [0.008],
        "a_lo": [-0.002],
        "a_k": [10],
        "b_rate": [0.0025],
        "c_lo": [0.0005],
        "c_hi": [0.006],
        "c_k": [40],
    }


def test_candidate_markets_enumerates_the_grid():
    grid = dict(tiny_grid(), b_rate=[0.002, 0.0025])
    assert len(list(candidate_markets(grid))) == 2


def test_candidate_markets_builds_the_named_assets_in_fixed_order():
    market = next(iter(candidate_markets(tiny_grid())))
    assert [a.name for a in market.assets] == ["A", "B", "C", "CASH"]


def test_the_grid_includes_decaying_assets():
    # Decay is the point of the Sprinter and is safe only because re-entry is
    # forbidden. If the grid contained no decaying asset the horizon structure
    # the study measures would be gone.
    assert any(
        market.assets[0].late_rate < market.assets[0].early_rate
        for market in candidate_markets(DEFAULT_GRID)
    )


def test_search_returns_only_feasible_combinations():
    results = search(tiny_grid(), N=40)
    assert all(r["all_pass"] for r in results)


def small_grid():
    """A handful of combinations. The full DEFAULT_GRID is ~9k solves and belongs
    in `make calibrate`, not in the unit-test path."""
    return {
        "a_hi": [0.008, 0.010],
        "a_lo": [-0.002],
        "a_k": [10],
        "b_rate": [0.0025, 0.003],
        "c_lo": [0.0005],
        "c_hi": [0.006, 0.008],
        "c_k": [40],
    }


def test_search_results_are_sorted_by_margin_descending():
    results = search(small_grid(), N=60)
    margins = [r["value_margin"] for r in results]
    assert margins == sorted(margins, reverse=True)


def test_freeze_writes_the_feasible_set_and_returns_the_winner(tmp_path):
    results = search(small_grid(), N=60)
    if not results:
        pytest.skip("no feasible set in the small grid; the real calibration runs at N=100")
    out = tmp_path / "feasible.json"
    winner = freeze(results, str(out))
    written = json.loads(out.read_text())
    assert written["feasible_set"]
    assert winner == results[0]


def test_freeze_refuses_an_empty_feasible_set(tmp_path):
    with pytest.raises(ValueError, match="no feasible"):
        freeze([], str(tmp_path / "feasible.json"))
