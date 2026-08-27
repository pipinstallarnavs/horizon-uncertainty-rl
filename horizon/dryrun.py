"""The dry-run gate.

Runs the entire pipeline -- environment, solver, agents, metrics, export -- on
scripted agents with zero API calls. DESIGN.md T10.3 requires this to be green
before an API key is ever loaded, so the expensive LLM runs happen exactly once
against a debugged pipeline.
"""
import json
import sys
from pathlib import Path

from horizon.agents.scripted import FixedAgent, MyopicGreedyAgent, OptimalAgent, RandomAgent
from horizon.env.distributions import DISTRIBUTIONS
from horizon.env.market import CASH, HOLDINGS, default_market
from horizon.eval.export import export_bundle
import numpy as np

from horizon.eval.metrics import EpisodeRecord, clock_coefficient, regret, trade_alignment
from horizon.report import verify_all
from horizon.runner.worker import play_episode, sample_exit_day
from horizon.solver.backward import solve

EPISODES = 200
OUT_DIR = Path("data/dryrun")


def agents_for(market, solution):
    return [
        ("optimal", lambda: OptimalAgent(solution)),
        ("always-A", lambda: FixedAgent(0)),
        ("always-B", lambda: FixedAgent(1)),
        ("always-C", lambda: FixedAgent(2)),
        ("always-CASH", lambda: FixedAgent(CASH)),
        ("myopic-greedy", lambda: MyopicGreedyAgent(market)),
        ("random", lambda: RandomAgent(seed=20260814)),
    ]


def run_agent(factory, market, dist, solution, episodes=EPISODES):
    records = []
    for episode in range(episodes):
        agent = factory()
        exit_day = sample_exit_day(dist, episode)
        _, summary = play_episode(agent, market, dist, solution, exit_day, dist.N)
        records.append(
            EpisodeRecord(
                episode=episode,
                true_exit_day=exit_day,
                terminal_wealth=summary["terminal_wealth"],
                holdings=tuple(summary["holdings"]),
                beliefs=tuple(summary["beliefs"]),
                switch_days=tuple(summary["switch_days"]),
            )
        )
    return records


def main() -> int:
    market = default_market()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("verifying the environment")
    for line in verify_all(market):
        print(f"  OK  {line}")

    failures = []
    results = {}

    for exit_name in ("uniform", "bimodal"):
        dist = DISTRIBUTIONS[exit_name](100)
        solution = solve(market, dist)
        print(f"\n{exit_name}: V*={solution.v_star:.5f}, {EPISODES} episodes per agent")
        print(f"  {'agent':<15}{'regret':>10}{'wealth':>10}{'clock':>10}{'align':>9}{'switch':>8}")

        rows = {}
        for name, factory in agents_for(market, solution):
            records = run_agent(factory, market, dist, solution)
            value = regret(records, solution.v_star)
            wealth = sum(r.terminal_wealth for r in records) / len(records)
            clock = clock_coefficient(records)
            align = trade_alignment(records, dist)
            switches = sum(len(r.switch_days) for r in records) / len(records)
            rows[name] = {
                "regret": value,
                "mean_wealth": wealth,
                "clock_coefficient": clock,
                "trade_alignment": align,
                "mean_switches": switches,
            }
            align_text = "n/a" if align != align else f"{align:9.2f}"
            print(
                f"  {name:<15}{value:10.5f}{wealth:10.5f}{clock:10.5f}{align_text}{switches:8.2f}"
            )

            # The optimal agent's regret must be zero in EXPECTATION, but V* is an
            # expectation while these are finitely many sampled exit days, so the
            # realised mean carries sampling error. Check against that, not zero.
            if name == "optimal":
                spread = np.std([r.terminal_wealth for r in records], ddof=1)
                stderr = spread / np.sqrt(len(records))
                if abs(value) > 4 * stderr:
                    failures.append(
                        f"{exit_name}: optimal agent regret {value:.5f} exceeds "
                        f"4 SE ({4 * stderr:.5f})"
                    )

            if name == "always-CASH" and value <= 0:
                failures.append(f"{exit_name}: cash baseline should trail the optimum")

            # If a purely myopic agent matches the optimum, the environment can be
            # solved without looking past tomorrow and tests no horizon reasoning.
            if name == "myopic-greedy":
                gap = value - rows["optimal"]["regret"]
                if gap < 1e-6:
                    failures.append(
                        f"{exit_name}: myopic-greedy matches the optimum "
                        f"(gap {gap:.2e}); the condition needs no horizon reasoning"
                    )

        results[exit_name] = rows

        bundle = OUT_DIR / f"dryrun-{exit_name}.zip"
        records = run_agent(agents_for(market, solution)[0][1], market, dist, solution)
        export_bundle(
            str(bundle), f"dryrun-{exit_name}", {"exit": exit_name, "agent": "optimal"},
            records, dist, solution.v_star,
            event_day=16,
        )
        print(f"  exported {bundle}")

    (OUT_DIR / "dryrun-results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print()
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        return 1
    print("  dry run green: pipeline verified end to end with zero API calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
