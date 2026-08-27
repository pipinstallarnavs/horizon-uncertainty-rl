"""The worker loop: claim an episode, play it, write it, repeat.

Runs on the VM under systemd. Many of these run concurrently; the leasing query
in the repository is what keeps them from colliding.
"""
import socket
import threading
import time
import traceback
import uuid

import numpy as np

from horizon.agents.base import Decision, Observation
from horizon.env.episode import wealth_path
from horizon.env.market import CASH
from horizon.env.rules import burn, is_available

# Vertex prices for gemini-3.7-flash, USD per million tokens. Fetched at
# implementation time; spend is metered for visibility, not to gate the run.
USD_PER_MILLION_PROMPT = 0.30
USD_PER_MILLION_OUTPUT = 2.50


def price(prompt_tokens: int, completion_tokens: int, thought_tokens: int) -> float:
    """Thought tokens bill as output tokens."""
    return (
        prompt_tokens * USD_PER_MILLION_PROMPT
        + (completion_tokens + thought_tokens) * USD_PER_MILLION_OUTPUT
    ) / 1_000_000


def worker_name() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


def sample_exit_day(dist, episode: int, master_seed: int = 20260814) -> int:
    """Draw episode i's exit day from the frozen seed bank.

    Derived deterministically from (master seed, episode) so every agent and
    baseline faces the identical set of worlds, making all between-agent
    comparisons paired (EXPERIMENTS.md T5.1).
    """
    rng = np.random.default_rng([master_seed, episode])
    return int(rng.choice(np.arange(1, dist.N + 1), p=dist.pmf[1:]))


def play_episode(agent, market, dist, solution, exit_day: int, horizon: int = 100):
    """Play one episode up to its realised exit day, collecting logged rows.

    The rollout stops at the exit rather than running the full horizon: each day
    past the exit would be a billed API call for a day that never happens.
    """
    rows = []
    holding, age, mask, mask_history = CASH, 0, 0, []
    holdings, beliefs, switches = [], [], []
    days = min(exit_day, horizon)

    agent.reset(0)
    for day in range(1, days + 1):
        obs = Observation(
            day=day,
            holding=holding,
            age=age,
            wealth=float(wealth_path(market, holdings)[-1]) if holdings else 1.0,
            switches=tuple(switches),
            spent=tuple(t for t in range(market.n_assets) if not is_available(mask, t)),
            horizon=horizon,
        )
        decision = agent.decide(obs)

        pi_star = int(solution.pi[day, holding, age, mask])
        action = decision.action
        if action != holding and not is_available(mask, action):
            decision = Decision(
                action=holding, belief=decision.belief, raw=decision.raw,
                thought_text=decision.thought_text,
                error_class="illegal_action", repair_used=decision.repair_used,
                prompt_tokens=decision.prompt_tokens,
                completion_tokens=decision.completion_tokens,
                thought_tokens=decision.thought_tokens, latency_ms=decision.latency_ms,
            )
            action = holding

        q_loss = _q_loss(solution, market, day, holding, age, mask, action)

        if action == holding:
            age += 1
        else:
            mask_history.append(mask)
            mask = burn(mask, holding)
            switches.append((day, action))
            age = 1
        holding = action
        holdings.append(holding)
        beliefs.append(decision.belief)

        rows.append(
            {
                "day": day,
                "holding": holding,
                "age": age,
                "wealth": float(wealth_path(market, holdings)[-1]),
                "prompt_tokens": decision.prompt_tokens,
                "completion_tokens": decision.completion_tokens,
                "thought_tokens": decision.thought_tokens,
                "cache_hit": None,
                "raw_completion": decision.raw[:4000],
                # Generous cap: the reasoning trace is the point of the log, and
                # truncating it mid-argument loses exactly the part worth reading.
                "thought_text": (decision.thought_text or "")[:16000],
                "parsed_belief": decision.belief,
                "parsed_action": decision.action,
                "repair_used": decision.repair_used,
                "error_class": decision.error_class,
                "truth_m_t": dist.mean_remaining(day),
                "truth_h_t": dist.hazard(day),
                "truth_pi_star": pi_star,
                "truth_q_loss": q_loss,
                "latency_ms": decision.latency_ms,
            }
        )

    wealth = wealth_path(market, holdings)
    errors = [r["error_class"] for r in rows]
    summary = {
        "episode": 0,
        "true_exit_day": exit_day,
        "terminal_wealth": float(wealth[-1]),
        "regret": float(solution.v_star - wealth[-1]),
        "holdings": holdings,
        "beliefs": [b for b in beliefs],
        "switch_days": [d for d, _ in switches],
        "format_error_rate": errors.count("format_error") / max(len(errors), 1),
        "empty_trunc_rate": errors.count("empty_thinking_truncation") / max(len(errors), 1),
    }
    return rows, summary


def _q_loss(solution, market, day, holding, age, mask, action) -> float:
    """Wealth multiplier given up by taking `action` instead of pi*.

    Lets errors be weighted by how costly they were rather than merely counted
    (DESIGN.md P6).
    """
    best = float(solution.V[day, holding, age, mask])
    if action == holding:
        taken = market.growth(holding, age + 1) * float(
            solution.V[day + 1, holding, min(age + 1, solution.V.shape[2] - 1), mask]
        )
    else:
        landed = burn(mask, holding)
        taken = (
            (1.0 - market.switch_cost)
            * market.growth(action, 1)
            * float(solution.V[day + 1, action, 1, landed])
        )
    return max(0.0, best - taken)


class Worker(threading.Thread):
    """One lease-play-write loop. Start as many as the rate limiter can feed."""

    def __init__(self, repo, run_id, agent_factory, market, dist, solution,
                 horizon=100, stop_event=None, empty_trunc_guard=0.005):
        super().__init__(daemon=True)
        self.repo = repo
        self.run_id = run_id
        self.agent_factory = agent_factory
        self.market = market
        self.dist = dist
        self.solution = solution
        self.horizon = horizon
        self.name_tag = worker_name()
        self.stop_event = stop_event or threading.Event()
        self.empty_trunc_guard = empty_trunc_guard
        self.completed = 0

    def run(self) -> None:
        while not self.stop_event.is_set():
            claim = self.repo.claim(self.run_id, self.name_tag)
            if claim is None:
                time.sleep(2.0)
                continue
            episode = claim["episode"]
            try:
                self.repo.clear_episode(self.run_id, episode)
                exit_day = sample_exit_day(self.dist, episode)
                agent = self.agent_factory()
                rows, summary = play_episode(
                    agent, self.market, self.dist, self.solution, exit_day, self.horizon
                )

                # An empty-reply rate above the guard means the output budget is
                # misconfigured, not that the model reasoned badly. Stop rather
                # than pollute the dataset (spec section 2.1).
                if summary["empty_trunc_rate"] > self.empty_trunc_guard:
                    self.repo.set_run_status(self.run_id, "aborted_empty_truncation")
                    self.stop_event.set()
                    self.repo.fail(self.run_id, episode, "empty_thinking_truncation above guard")
                    return

                summary["episode"] = episode
                self.repo.write_days(self.run_id, episode, rows)
                self.repo.write_episode(self.run_id, summary)
                self.repo.record_spend(
                    self.run_id,
                    sum(r["prompt_tokens"] for r in rows),
                    sum(r["completion_tokens"] for r in rows),
                    sum(r["thought_tokens"] for r in rows),
                    price(
                        sum(r["prompt_tokens"] for r in rows),
                        sum(r["completion_tokens"] for r in rows),
                        sum(r["thought_tokens"] for r in rows),
                    ),
                )
                self.repo.complete(self.run_id, episode)
                self.completed += 1
            except Exception:
                self.repo.fail(self.run_id, episode, traceback.format_exc())
                time.sleep(1.0)
