"""Grid search for asset parameters that make the environment worth running.

Writes the whole feasible set, not just the winner, so the appendix can show
that the study's conclusions do not hinge on one lucky parameter choice.

Candidates use the README section 4 asset shapes, including the decaying
Sprinter. That is only safe because re-entry is forbidden (horizon.env.rules);
without the rule, an asset paying more when freshly held lets the optimum farm
the age-clock reset and the policy stops depending on t entirely.
"""
import itertools
import json

from horizon.calibrate.targets import evaluate_targets, fixed_asset_values
from horizon.env.distributions import bimodal_exit, uniform_exit
from horizon.env.market import AssetSpec, Market
from horizon.solver.backward import solve

DEFAULT_GRID = {
    "a_hi": [0.006, 0.008, 0.010, 0.012],
    "a_lo": [-0.004, -0.002, 0.0],
    "a_k": [8, 10, 12, 15],
    "b_rate": [0.002, 0.0025, 0.003, 0.004],
    "c_lo": [0.0, 0.0005, 0.001],
    "c_hi": [0.005, 0.006, 0.008, 0.010],
    "c_k": [25, 30, 40, 50],
}

_KEYS = ("a_hi", "a_lo", "a_k", "b_rate", "c_lo", "c_hi", "c_k")


def candidate_markets(grid: dict, switch_cost: float = 0.001):
    """Enumerate every parameter combination in the grid as a Market."""
    for combo in itertools.product(*(grid[k] for k in _KEYS)):
        p = dict(zip(_KEYS, combo))
        yield Market(
            assets=(
                AssetSpec("A", p["a_hi"], p["a_lo"], p["a_k"]),
                AssetSpec("B", p["b_rate"], p["b_rate"], 1),
                AssetSpec("C", p["c_lo"], p["c_hi"], p["c_k"]),
                AssetSpec("CASH", 0.0, 0.0, 1),
            ),
            switch_cost=switch_cost,
        )


def _params_of(market: Market) -> dict:
    a, b, c, _ = market.assets
    return {
        "a_hi": a.early_rate,
        "a_lo": a.late_rate,
        "a_k": a.breakpoint,
        "b_rate": b.early_rate,
        "c_lo": c.early_rate,
        "c_hi": c.late_rate,
        "c_k": c.breakpoint,
    }


def search(grid: dict, N: int = 100) -> list:
    """Evaluate every candidate and keep the feasible ones, best margin first."""
    dist_u, dist_b = uniform_exit(N), bimodal_exit(N)
    feasible = []
    for market in candidate_markets(grid):
        report = evaluate_targets(market, dist_u, dist_b)
        if not report.all_pass:
            continue
        sol = solve(market, dist_u)
        best_fixed = max(fixed_asset_values(market, dist_u).values())
        feasible.append(
            {
                **_params_of(market),
                "v_star": sol.v_star,
                "best_fixed": best_fixed,
                "value_margin": sol.v_star / best_fixed - 1.0,
                "all_pass": True,
            }
        )
    feasible.sort(key=lambda r: r["value_margin"], reverse=True)
    return feasible


def freeze(results: list, path: str) -> dict:
    """Persist the feasible set and return the chosen (highest-margin) parameters."""
    if not results:
        raise ValueError(
            "no feasible parameter set found; per EXPERIMENTS.md section 11, widen N "
            "to 150 or relax the value-separation target, and document the relaxation "
            "before any LLM run"
        )
    winner = results[0]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"chosen": winner, "feasible_set": results}, handle, indent=2)
    return winner
