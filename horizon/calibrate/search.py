"""Grid search for asset parameters that make the environment worth running.

Writes the whole feasible set, not just the winner, so the appendix can show
that the study's conclusions do not hinge on one lucky parameter choice.

Every candidate is monotone in holding age by construction. That is not a
stylistic choice: an asset paying more when freshly held lets the optimum farm
the age-clock reset, which collapses the policy to something identical at every
t and destroys the whole experiment (see README section 4).
"""
import itertools
import json

from horizon.calibrate.targets import evaluate_targets, fixed_asset_values
from horizon.env.distributions import bimodal_exit, uniform_exit
from horizon.env.market import AssetSpec, Market
from horizon.solver.backward import solve

DEFAULT_GRID = {
    "a_rate": [0.001, 0.002, 0.003],
    "b_rate": [0.004, 0.005, 0.006, 0.007, 0.008],
    "c_hi": [0.008, 0.010, 0.012, 0.014, 0.016, 0.020],
    "c_k": [10, 15, 20, 25, 30, 40],
}

_KEYS = ("a_rate", "b_rate", "c_hi", "c_k")


def candidate_markets(grid: dict, switch_cost: float = 0.001):
    """Enumerate every parameter combination in the grid as a Market."""
    for combo in itertools.product(*(grid[k] for k in _KEYS)):
        p = dict(zip(_KEYS, combo))
        yield Market(
            assets=(
                AssetSpec("A", p["a_rate"], p["a_rate"], 1),
                AssetSpec("B", p["b_rate"], p["b_rate"], 1),
                AssetSpec("C", 0.0, p["c_hi"], p["c_k"]),
                AssetSpec("CASH", 0.0, 0.0, 1),
            ),
            switch_cost=switch_cost,
        )


def _params_of(market: Market) -> dict:
    a, b, c, _ = market.assets
    return {
        "a_rate": a.early_rate,
        "b_rate": b.early_rate,
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
