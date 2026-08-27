import time

import numpy as np
import pytest

from horizon.agents.scripted import FixedAgent, OptimalAgent
from horizon.env.distributions import bimodal_exit, uniform_exit
from horizon.env.market import CASH, default_market
from horizon.runner.worker import play_episode, price, sample_exit_day
from horizon.solver.backward import solve


def setup(dist=None):
    market = default_market()
    dist = dist or uniform_exit(100)
    return market, dist, solve(market, dist)


def test_exit_days_are_reproducible_across_agents():
    # Common random numbers: every agent must face the identical worlds.
    dist = uniform_exit(100)
    first = [sample_exit_day(dist, i) for i in range(50)]
    second = [sample_exit_day(dist, i) for i in range(50)]
    assert first == second


def test_exit_days_follow_the_distribution():
    dist = bimodal_exit()
    days = [sample_exit_day(dist, i) for i in range(400)]
    assert set(days) <= {10} | set(range(91, 101))
    assert 0.35 < sum(d == 10 for d in days) / len(days) < 0.65


def test_episode_stops_at_the_realised_exit_day():
    market, dist, sol = setup()
    rows, summary = play_episode(OptimalAgent(sol), market, dist, sol, exit_day=37)
    assert len(rows) == 37
    assert summary["true_exit_day"] == 37


def test_optimal_agent_has_zero_q_loss_every_day():
    # Playing pi* can never give up value; any nonzero loss means the q_loss
    # computation disagrees with the solver it was derived from.
    market, dist, sol = setup()
    rows, _ = play_episode(OptimalAgent(sol), market, dist, sol, exit_day=100)
    assert max(r["truth_q_loss"] for r in rows) == pytest.approx(0.0, abs=1e-12)


def test_a_suboptimal_agent_accrues_q_loss():
    market, dist, sol = setup()
    rows, _ = play_episode(FixedAgent(CASH), market, dist, sol, exit_day=100)
    assert max(r["truth_q_loss"] for r in rows) > 0.0


def test_truth_columns_carry_the_ground_truth_for_the_day():
    market, dist, sol = setup()
    rows, _ = play_episode(OptimalAgent(sol), market, dist, sol, exit_day=20)
    assert rows[9]["truth_m_t"] == pytest.approx(dist.mean_remaining(10))
    assert rows[9]["truth_h_t"] == pytest.approx(dist.hazard(10))


def test_regret_is_zero_when_the_optimal_agent_survives_to_the_horizon():
    market, dist, sol = setup()
    _, summary = play_episode(OptimalAgent(sol), market, dist, sol, exit_day=100)
    assert summary["terminal_wealth"] > 1.0


def test_switch_days_are_recorded():
    market, dist, sol = setup()
    _, summary = play_episode(OptimalAgent(sol), market, dist, sol, exit_day=100)
    # Optimal under uniform: enter A on day 1, move to B when its boost ends.
    assert summary["switch_days"][:2] == [1, 16]


def test_price_counts_thought_tokens_as_output():
    cheap = price(1000, 10, 0)
    dear = price(1000, 10, 1000)
    assert dear > cheap


def test_illegal_action_is_refused_and_logged():
    market, dist, sol = setup()

    class ReEntrant(FixedAgent):
        """Tries to return to A after leaving it."""

        def __init__(self):
            super().__init__(0)
            self.day = 0

        def decide(self, obs):
            self.day += 1
            from horizon.agents.base import Decision

            return Decision(action=1 if self.day in (2, 3) else 0)

    rows, _ = play_episode(ReEntrant(), market, dist, sol, exit_day=6)
    assert any(r["error_class"] == "illegal_action" for r in rows)


def test_the_reasoning_trace_is_stored_alongside_the_reply():
    market, dist, sol = setup()

    class Talkative(FixedAgent):
        def decide(self, obs):
            from horizon.agents.base import Decision

            return Decision(action=0, raw="ACTION: HOLD",
                            thought_text="A still compounds; nothing to do.")

    rows, _ = play_episode(Talkative(0), market, dist, sol, exit_day=3)
    assert all(r["thought_text"] == "A still compounds; nothing to do." for r in rows)


def test_a_refused_action_keeps_the_reasoning_that_produced_it():
    # The trace behind an illegal move is the most diagnostic one there is:
    # discarding it would hide whether the model forgot the no-re-entry rule.
    market, dist, sol = setup()

    class ReEntrantTalker(FixedAgent):
        def __init__(self):
            super().__init__(0)
            self.day = 0

        def decide(self, obs):
            self.day += 1
            from horizon.agents.base import Decision

            return Decision(action=1 if self.day in (2, 3) else 0,
                            thought_text="B looked cheap again.")

    rows, _ = play_episode(ReEntrantTalker(), market, dist, sol, exit_day=6)
    refused = [r for r in rows if r["error_class"] == "illegal_action"]
    assert refused and all(r["thought_text"] == "B looked cheap again." for r in refused)


def test_the_trace_is_capped_but_generously():
    market, dist, sol = setup()

    class Verbose(FixedAgent):
        def decide(self, obs):
            from horizon.agents.base import Decision

            return Decision(action=0, thought_text="x" * 40000)

    rows, _ = play_episode(Verbose(0), market, dist, sol, exit_day=2)
    assert len(rows[0]["thought_text"]) == 16000


def test_the_oracle_condition_is_actually_runnable():
    # Condition O states the realised exit day in the prompt, so the agent
    # factory must receive it. Without this the oracle cell raises on its first
    # episode and the arm cannot run at all.
    from horizon.agents.gemini import GeminiAgent
    from horizon.runner.orchestrator import build_cell

    market, dist, _ = setup()
    seen = {}

    class Recorder:
        def __call__(self, true_exit_day=None):
            seen["exit_day"] = true_exit_day
            return FixedAgent(0)

    factory = Recorder()
    factory(41)
    assert seen["exit_day"] == 41

    # And the prompt the oracle builds must name that exact day.
    agent = GeminiAgent(None, market, dist, condition="O", true_exit_day=41)
    assert "end of day 41" in agent.system
    assert build_cell()["info"] in ("K", "U", "O")


def test_the_worker_hands_the_exit_day_to_the_factory():
    from horizon.runner.worker import Worker

    market, dist, sol = setup()
    got = []

    def factory(true_exit_day=None):
        got.append(true_exit_day)
        return FixedAgent(0)

    class OneShotRepo:
        def __init__(self):
            self.claimed = False

        def claim(self, *a, **k):
            if self.claimed:
                return None
            self.claimed = True
            return {"episode": 3, "attempts": 1}

        def clear_episode(self, *a):
            pass

        def write_days(self, *a):
            pass

        def write_episode(self, *a):
            pass

        def record_spend(self, *a):
            pass

        def complete(self, *a):
            self.done = True

        def fail(self, *a):
            pass

        def set_run_status(self, *a):
            pass

    import threading

    stop = threading.Event()
    worker = Worker(OneShotRepo(), "r", factory, market, dist, sol, 100, stop)
    worker.start()
    for _ in range(100):
        if got:
            break
        time.sleep(0.05)
    stop.set()
    worker.join(timeout=5)
    assert got and got[0] == sample_exit_day(dist, 3)
