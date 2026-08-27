"""Exact optimal policy by backward induction over (day, holding, age).

Wealth factors out of the recursion because every daily return is multiplicative
and independent of wealth, so V is an expected wealth *multiplier* rather than a
wealth level. That is what keeps the state space at ~N * K * N.
"""
from dataclasses import dataclass

import numpy as np

from horizon.env.exit_dist import ExitDistribution
from horizon.env.market import CASH, Market


@dataclass(frozen=True)
class Solution:
    V: np.ndarray
    pi: np.ndarray
    v_star: float


def solve(market: Market, dist: ExitDistribution) -> Solution:
    """Compute V* and the optimal action table.

    State (t, k, age) means: it is the start of day t, the agent currently holds
    k, and has held it for `age` days before today. Holding makes today's age
    index age+1; switching makes it 1 and charges the switch cost.
    """
    N, K = dist.N, market.n_assets
    hazard = np.array([0.0] + [dist.hazard(t) for t in range(1, N + 1)])

    # Growth table G[k, age] for age 1..N, precomputed once rather than per state.
    G = np.ones((K, N + 2))
    for k in range(K):
        for age in range(1, N + 2):
            G[k, age] = market.growth(k, min(age, N + 1))

    # V[N+1] = 1.0: nothing accrues after the horizon. h(N) = 1 makes it unreachable.
    V = np.ones((N + 2, K, N + 1))
    pi = np.zeros((N + 1, K, N + 1), dtype=np.int8)

    ages = np.arange(0, N)
    held = ages + 1
    next_age = np.minimum(held, N)
    stay_cost = 1.0 - market.switch_cost

    for t in range(N, 0, -1):
        h = hazard[t]
        tail = h + (1.0 - h) * V[t + 1]

        # Switching resets the age clock to 1, so a switch's value depends only on
        # the target, never on how long the current position has been held. That
        # makes it a scalar per target instead of a per-state search.
        switch_values = stay_cost * G[:, 1] * tail[:, 1]

        for k in range(K):
            stay = G[k, held] * tail[k, next_age]

            masked = switch_values.copy()
            masked[k] = -np.inf
            best_target = int(np.argmax(masked))
            best_switch = masked[best_target]

            take_switch = best_switch > stay + 1e-15
            V[t, k, :N] = np.where(take_switch, best_switch, stay)
            pi[t, k, :N] = np.where(take_switch, best_target, k)

    return Solution(V=V, pi=pi, v_star=float(V[1, CASH, 0]))


def optimal_holdings(sol: Solution, market: Market, days: int) -> list:
    """Unroll pi into the day-by-day holding sequence from the day-1 CASH start.

    With deterministic returns the only randomness is the exit time, which ends
    the episode rather than revealing anything, so the optimal policy is an
    open-loop sequence. verify.brute_force_value exploits exactly this. Adding
    return noise breaks the assumption and needs a closed-loop treatment.
    """
    holdings = []
    current, age = CASH, 0
    for t in range(1, days + 1):
        action = int(sol.pi[t, current, age])
        age = age + 1 if action == current else 1
        current = action
        holdings.append(current)
    return holdings
