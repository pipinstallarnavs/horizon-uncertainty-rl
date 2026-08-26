"""Exact optimal policy by backward induction over (day, holding, age, spent).

Wealth factors out of the recursion because every daily return is multiplicative
and independent of wealth, so V is an expected wealth *multiplier* rather than a
wealth level.

The fourth state component is the no-re-entry mask (horizon.env.rules): a 3-bit
record of which risky assets have already been left. It multiplies the state
space by 8 and is what makes decaying assets safe to use at all.
"""
from dataclasses import dataclass

import numpy as np

from horizon.env.exit_dist import ExitDistribution
from horizon.env.market import CASH, Market
from horizon.env.rules import burn, legal_targets

N_MASKS = 8


@dataclass(frozen=True)
class Solution:
    V: np.ndarray
    pi: np.ndarray
    v_star: float


def solve(market: Market, dist: ExitDistribution) -> Solution:
    """Compute V* and the optimal action table.

    State (t, k, age, mask): it is the start of day t, the agent holds k, has
    held it for `age` days before today, and `mask` records which risky assets
    have already been left. Holding makes today's age index age+1; switching
    makes it 1, charges the switch cost, and spends the asset being left.
    """
    N, K = dist.N, market.n_assets
    hazard = np.array([0.0] + [dist.hazard(t) for t in range(1, N + 1)])

    growth = np.ones((K, N + 2))
    for k in range(K):
        for age in range(1, N + 2):
            growth[k, age] = market.growth(k, min(age, N + 1))

    V = np.ones((N + 2, K, N + 1, N_MASKS))
    pi = np.zeros((N + 1, K, N + 1, N_MASKS), dtype=np.int8)

    ages = np.arange(0, N)
    held = ages + 1
    next_age = np.minimum(held, N)
    keep = 1.0 - market.switch_cost

    for t in range(N, 0, -1):
        h = hazard[t]
        tail = h + (1.0 - h) * V[t + 1]
        for k in range(K):
            for mask in range(N_MASKS):
                # Leaving k spends it, so the mask a switch lands in is fixed by
                # (k, mask) and every legal target shares it.
                landed = burn(mask, k)
                stay = growth[k, held] * tail[k, next_age, mask]

                best_switch, best_target = -np.inf, k
                for target in legal_targets(mask, k, K):
                    value = keep * growth[target, 1] * tail[target, 1, landed]
                    if value > best_switch:
                        best_switch, best_target = value, target

                take = best_switch > stay + 1e-15
                V[t, k, :N, mask] = np.where(take, best_switch, stay)
                pi[t, k, :N, mask] = np.where(take, best_target, k)

    return Solution(V=V, pi=pi, v_star=float(V[1, CASH, 0, 0]))


def optimal_holdings(sol: Solution, market: Market, days: int) -> list:
    """Unroll pi into the day-by-day holding sequence from the day-1 CASH start.

    With deterministic returns the only randomness is the exit time, which ends
    the episode rather than revealing anything, so the optimal policy is an
    open-loop sequence. verify.brute_force_value exploits exactly this. Adding
    return noise breaks the assumption and needs a closed-loop treatment.
    """
    holdings = []
    current, age, mask = CASH, 0, 0
    for t in range(1, days + 1):
        action = int(sol.pi[t, current, age, mask])
        if action == current:
            age += 1
        else:
            mask = burn(mask, current)
            age = 1
        current = action
        holdings.append(current)
    return holdings
