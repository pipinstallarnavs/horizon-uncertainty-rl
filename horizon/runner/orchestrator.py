"""Container entrypoint: build the run, start the workers, supervise.

Everything it needs comes from the environment so the same image serves every
experiment cell without a rebuild.
"""
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from horizon.agents.gemini import (
    DEFAULT_OUTPUT_TOKENS,
    AimdLimiter,
    GeminiAgent,
    GeminiClient,
    TokenSource,
)
from horizon.env.distributions import DISTRIBUTIONS
from horizon.env.market import AssetSpec, Market, default_market
from horizon.runner.worker import Worker
from horizon.solver.backward import solve
from horizon.store.repository import Repository

CALIBRATION_PATH = Path("data/calibration.json")


def market_from_calibration(switch_cost: float = 0.001) -> Market:
    """Use frozen calibrated parameters when present; otherwise the provisional set.

    Parameters are chosen by search against the calibration targets and frozen
    before any LLM run, never hand-tuned after seeing results.
    """
    if not CALIBRATION_PATH.exists():
        return default_market(switch_cost)
    chosen = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["chosen"]
    return Market(
        assets=(
            AssetSpec("A", chosen["a_hi"], chosen["a_lo"], chosen["a_k"]),
            AssetSpec("B", chosen["b_rate"], chosen["b_rate"], 1),
            AssetSpec("C", chosen["c_lo"], chosen["c_hi"], chosen["c_k"]),
            AssetSpec("CASH", 0.0, 0.0, 1),
        ),
        switch_cost=switch_cost,
    )


def build_cell() -> dict:
    return {
        "exit": os.environ.get("HORIZON_EXIT", "uniform"),
        "info": os.environ.get("HORIZON_INFO", "K"),
        "probe": os.environ.get("HORIZON_PROBE", "1") == "1",
        "prompt": os.environ.get("HORIZON_PROMPT", "table"),
        "history": os.environ.get("HORIZON_HISTORY", "compressed"),
        "temperature": float(os.environ.get("HORIZON_TEMPERATURE", "0")),
        "thinking_budget": _optional_int("HORIZON_THINKING_BUDGET"),
        # Part of the cell, not an implementation detail: it bounds how long the
        # model is allowed to think, so two cells with different values are not
        # comparable and the stored metadata has to say which is which.
        "max_output_tokens": int(
            os.environ.get("HORIZON_MAX_OUTPUT", str(DEFAULT_OUTPUT_TOKENS))
        ),
    }


def _optional_int(name):
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else None


def main() -> int:
    run_id = os.environ.get("HORIZON_RUN_ID", "default")
    project = os.environ.get("HORIZON_PROJECT")
    key_path = os.environ.get("HORIZON_SA_KEY", "/run/secrets/sa-key.json")
    episodes = int(os.environ.get("HORIZON_EPISODES", "200"))
    n_workers = int(os.environ.get("HORIZON_WORKERS", "24"))
    horizon = int(os.environ.get("HORIZON_DAYS", "100"))

    cell = build_cell()
    market = market_from_calibration()
    dist = DISTRIBUTIONS[cell["exit"]](horizon)
    solution = solve(market, dist)

    repo = Repository()
    repo.init_schema()
    repo.create_run(run_id, cell, horizon, solution.v_star, model_snapshot="gemini-3.7-flash")
    repo.seed_work_units(run_id, episodes)
    repo.set_run_status(run_id, "running")

    print(f"run={run_id} cell={cell} V*={solution.v_star:.5f} episodes={episodes}", flush=True)

    if not project:
        print("HORIZON_PROJECT unset; nothing to run", flush=True)
        return 2

    limiter = AimdLimiter(start=int(os.environ.get("HORIZON_START_CONCURRENCY", "8")))
    tokens = TokenSource(key_path)
    client = GeminiClient(project=project, token_source=tokens, limiter=limiter)

    def agent_factory(true_exit_day=None):
        return GeminiAgent(
            client, market, dist,
            condition=cell["info"], variant=cell["prompt"], probe=cell["probe"],
            history=cell["history"], temperature=cell["temperature"],
            thinking_budget=cell["thinking_budget"],
            true_exit_day=true_exit_day,
            max_output_tokens=cell["max_output_tokens"],
        )

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    workers = [
        Worker(repo, run_id, agent_factory, market, dist, solution, horizon, stop)
        for _ in range(n_workers)
    ]
    for worker in workers:
        worker.start()

    while not stop.is_set():
        progress = repo.progress(run_id)
        if progress.get("done", 0) >= episodes:
            repo.set_run_status(run_id, "complete")
            stop.set()
            break
        print(
            f"progress={progress} concurrency={limiter.limit:.1f} "
            f"tokens={repo.token_totals(run_id)['total_tokens']:,}",
            flush=True,
        )
        time.sleep(30)

    for worker in workers:
        worker.join(timeout=30)
    print(f"final progress={repo.progress(run_id)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
