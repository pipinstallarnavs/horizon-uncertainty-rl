"""One command that establishes the environment is sound before any API spend.

Everything here is deterministic and free. DESIGN.md T10.3 requires the whole
analysis pipeline to run end to end on scripted agents before an API key is
loaded; this is the environment half of that gate.
"""
import json

from horizon.calibrate.targets import crossover_day, expected_switches, fixed_asset_values
from horizon.env.distributions import DISTRIBUTIONS, bimodal_exit, uniform_exit
from horizon.env.market import HOLDINGS, Market, default_market
from horizon.solver.backward import optimal_holdings, solve
from horizon.solver.verify import (
    assert_non_collapse,
    brute_force_value,
    monte_carlo_value,
    time_varying_fraction,
)

MC_EPISODES = 20000
MC_SEED = 20260814
MC_TOLERANCE_SE = 4.0


def describe_path(holdings) -> str:
    """Compress a day-by-day holding list into readable runs, e.g. 'd1-10:A  d11-100:B'."""
    runs = []
    previous, start = None, 1
    for day, holding in enumerate(holdings, start=1):
        if holding != previous:
            if previous is not None:
                runs.append((start, day - 1, HOLDINGS[previous]))
            previous, start = holding, day
    runs.append((start, len(holdings), HOLDINGS[previous]))
    return "  ".join(f"d{a}-{b}:{name}" for a, b, name in runs)


def environment_report(market: Market, N: int = 100) -> dict:
    """Per-distribution summary plus the hazard and remaining-time curves (Figure 1)."""
    report = {}
    for name, factory in DISTRIBUTIONS.items():
        dist = factory(N)
        sol = solve(market, dist)
        holdings = optimal_holdings(sol, market, N)
        report[name] = {
            "v_star": sol.v_star,
            "optimal_path": describe_path(holdings),
            "fixed_asset_values": fixed_asset_values(market, dist),
            "crossover_day": crossover_day(sol, market, dist),
            "expected_switches": expected_switches(sol, market, dist),
            "time_varying_fraction": time_varying_fraction(sol, dist),
            "hazard": [dist.hazard(t) for t in range(1, N + 1)],
            "mean_remaining": [dist.mean_remaining(t) for t in range(1, N + 1)],
        }
    return report


def verify_all(market: Market) -> list:
    """Run every solver check. Raises on the first failure; returns a summary."""
    checks = []

    small = uniform_exit(8)
    exact = solve(market, small)
    best, _ = brute_force_value(market, small)
    if abs(exact.v_star - best) > 1e-12 * max(1.0, abs(best)):
        raise AssertionError(f"brute force {best!r} != solver {exact.v_star!r}")
    checks.append(f"brute force at N=8 matches V* exactly ({best:.10f})")

    dist = uniform_exit(100)
    sol = solve(market, dist)
    mean, stderr = monte_carlo_value(market, dist, sol, n=MC_EPISODES, seed=MC_SEED)
    if abs(mean - sol.v_star) > MC_TOLERANCE_SE * stderr:
        raise AssertionError(f"Monte Carlo {mean:.6f} +/- {stderr:.6f} != V* {sol.v_star:.6f}")
    checks.append(f"Monte Carlo {mean:.5f} +/- {stderr:.5f} matches V* {sol.v_star:.5f}")

    for factory in (uniform_exit, bimodal_exit):
        d = factory(100)
        assert_non_collapse(solve(market, d), d)
        checks.append(f"{d.name} passes non-collapse ({time_varying_fraction(solve(market, d), d):.1%})")

    return checks


def main() -> None:
    market = default_market()
    for line in verify_all(market):
        print(f"  OK  {line}")
    print()
    report = environment_report(market)
    for name, data in report.items():
        print(
            f"  {name:9s} V*={data['v_star']:.5f}  "
            f"switches={data['expected_switches']:.2f}  "
            f"time-varying={data['time_varying_fraction']:.1%}"
        )
        print(f"             {data['optimal_path']}")
    with open("data/environment_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print("\n  written: data/environment_report.json")


if __name__ == "__main__":
    main()
