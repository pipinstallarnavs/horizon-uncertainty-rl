import pytest

from horizon.agents.base import initial_observation
from horizon.agents.scripted import FixedAgent, MyopicGreedyAgent, OptimalAgent, RandomAgent
from horizon.env.distributions import uniform_exit
from horizon.env.episode import expected_wealth, is_legal
from horizon.env.market import CASH, default_market
from horizon.runner.rollout import run_episode
from horizon.solver.backward import solve


def test_optimal_agent_reproduces_v_star_exactly():
    # The pipeline's own sanity check: if this agent shows nonzero regret,
    # the metric code is wrong, not the agent.
    m, dist = default_market(), uniform_exit(100)
    sol = solve(m, dist)
    holdings = run_episode(OptimalAgent(sol), m, horizon=100).holdings
    assert expected_wealth(m, holdings, dist.pmf) == pytest.approx(sol.v_star)


def test_every_scripted_agent_produces_a_legal_path():
    m = default_market()
    sol = solve(m, uniform_exit(100))
    agents = [
        OptimalAgent(sol),
        FixedAgent(0),
        FixedAgent(CASH),
        MyopicGreedyAgent(m),
        RandomAgent(seed=7),
    ]
    for agent in agents:
        agent.reset(0)
        assert is_legal(run_episode(agent, m, horizon=100).holdings), agent.name


def test_fixed_agent_holds_one_thing_forever():
    m = default_market()
    result = run_episode(FixedAgent(1), m, horizon=20)
    assert set(result.holdings) == {1}


def test_myopic_agent_ignores_the_horizon():
    # Same decisions regardless of how long the episode is, because it never
    # looks past tomorrow.
    m = default_market()
    short = run_episode(MyopicGreedyAgent(m), m, horizon=20).holdings
    long = run_episode(MyopicGreedyAgent(m), m, horizon=100).holdings
    assert short == long[:20]


def test_random_agent_is_reproducible_per_episode():
    m = default_market()
    a, b = RandomAgent(seed=3), RandomAgent(seed=3)
    a.reset(11)
    b.reset(11)
    assert run_episode(a, m, horizon=50).holdings == run_episode(b, m, horizon=50).holdings
