"""Three independent checks on the solver, plus the degeneracy guard.

The solver is the denominator of every reported regret, so it is verified
against an exhaustive search (exact, small N), a Monte-Carlo rollout
(statistical, full N), and a structural assertion that the environment is not
degenerate.
"""
import itertools

import numpy as np

from horizon.env.episode import expected_wealth, wealth_path
from horizon.env.exit_dist import ExitDistribution
from horizon.env.market import Market
from horizon.solver.backward import Solution, optimal_holdings

MAX_BRUTE_FORCE_DAYS = 8


def brute_force_value(market: Market, dist: ExitDistribution):
    """Exhaustively search every holding sequence. Exact, and exponential.

    Legitimate because deterministic returns make the optimal policy open-loop:
    the exit time is the only randomness and it ends the episode rather than
    revealing anything the agent could condition on.
    """
    N, K = dist.N, market.n_assets
    if N > MAX_BRUTE_FORCE_DAYS:
        raise ValueError(f"brute force is intractable for N={N} (max {MAX_BRUTE_FORCE_DAYS})")
    best_value, best_sequence = -np.inf, None
    for sequence in itertools.product(range(K), repeat=N):
        value = expected_wealth(market, list(sequence), dist.pmf)
        if value > best_value:
            best_value, best_sequence = value, list(sequence)
    return best_value, best_sequence


def monte_carlo_value(market: Market, dist: ExitDistribution, sol: Solution, n: int, seed: int):
    """Roll out pi* against sampled exit times; returns (mean, standard error)."""
    rng = np.random.default_rng(seed)
    days = np.arange(1, dist.N + 1)
    draws = rng.choice(days, size=n, p=dist.pmf[1:])
    path = wealth_path(market, optimal_holdings(sol, market, dist.N))
    samples = path[draws]
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n))


def time_varying_fraction(sol: Solution, dist: ExitDistribution) -> float:
    """Fraction of reachable (holding, age) pairs whose optimal action varies with t.

    A pair is reachable at day t only if age < t, since the position cannot have
    been held longer than the episode has run.
    """
    N = dist.N
    K = sol.pi.shape[1]
    varying = total = 0
    for k in range(K):
        for age in range(0, N):
            days = [t for t in range(1, N + 1) if age < t]
            if len(days) < 2:
                continue
            total += 1
            if len({int(sol.pi[t, k, age]) for t in days}) > 1:
                varying += 1
    return varying / total if total else 0.0


def assert_non_collapse(sol: Solution, dist: ExitDistribution, threshold: float = 0.20) -> None:
    """Guard against spending API budget on an environment that tests nothing."""
    fraction = time_varying_fraction(sol, dist)
    if fraction < threshold:
        raise AssertionError(
            f"policy collapse: only {fraction:.1%} of reachable states are "
            f"time-varying (need {threshold:.0%}); the environment is degenerate"
        )
