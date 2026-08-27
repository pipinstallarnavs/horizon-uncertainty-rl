"""Scripted baselines. All free, all deterministic, none touch an API.

These exist for two reasons: they are the comparison set in EXPERIMENTS.md
section 4, and they are what the whole pipeline is debugged against before any
frontier spend (DESIGN.md T10.3).
"""
import numpy as np

from horizon.agents.base import Agent, Decision, Observation
from horizon.env.market import CASH, Market
from horizon.env.rules import burn, is_available
from horizon.solver.backward import Solution


class OptimalAgent(Agent):
    """Plays pi* exactly. Its regret is zero by construction, so it validates the
    metric pipeline: any nonzero regret reported for this agent is a bug."""

    name = "optimal"

    def __init__(self, solution: Solution):
        self.solution = solution
        self._mask = 0
        self._previous = CASH

    def reset(self, episode: int) -> None:
        self._mask = 0
        self._previous = CASH

    def decide(self, obs: Observation) -> Decision:
        action = int(self.solution.pi[obs.day, obs.holding, obs.age, self._mask])
        if action != obs.holding:
            self._mask = burn(self._mask, obs.holding)
        return Decision(action=action)


class FixedAgent(Agent):
    """Always holds one asset. Answers 'do you even need to trade?'"""

    def __init__(self, target: int):
        self.target = target
        self.name = f"always-{['A', 'B', 'C', 'CASH'][target]}"

    def decide(self, obs: Observation) -> Decision:
        return Decision(action=self.target)


class MyopicGreedyAgent(Agent):
    """Maximises tomorrow's return only, ignoring the horizon entirely.

    The ablation that isolates temporal reasoning: a myopic agent has perfect
    market knowledge and zero horizon awareness, so the gap between it and pi*
    is exactly what horizon reasoning is worth.
    """

    name = "myopic-greedy"

    def __init__(self, market: Market):
        self.market = market
        self._mask = 0

    def reset(self, episode: int) -> None:
        self._mask = 0

    def decide(self, obs: Observation) -> Decision:
        stay = self.market.growth(obs.holding, obs.age + 1)
        best_value, best_action = stay, obs.holding
        for target in range(self.market.n_assets):
            if target == obs.holding or not is_available(self._mask, target):
                continue
            value = (1.0 - self.market.switch_cost) * self.market.growth(target, 1)
            if value > best_value + 1e-15:
                best_value, best_action = value, target
        if best_action != obs.holding:
            self._mask = burn(self._mask, obs.holding)
        return Decision(action=best_action)


class RandomAgent(Agent):
    """Sanity floor. Uniform over legal actions."""

    name = "random"

    def __init__(self, n_assets: int = 4, seed: int = 0):
        self.n_assets = n_assets
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._mask = 0

    def reset(self, episode: int) -> None:
        self._rng = np.random.default_rng(self._seed + episode)
        self._mask = 0

    def decide(self, obs: Observation) -> Decision:
        legal = [obs.holding] + [
            t
            for t in range(self.n_assets)
            if t != obs.holding and is_available(self._mask, t)
        ]
        action = int(self._rng.choice(legal))
        if action != obs.holding:
            self._mask = burn(self._mask, obs.holding)
        return Decision(action=action)
