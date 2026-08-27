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
