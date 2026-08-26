"""Three independent checks on the solver, plus the degeneracy guard.

The solver is the denominator of every reported regret, so it is verified
against an exhaustive search (exact, small N), a Monte-Carlo rollout
(statistical, full N), and a structural assertion that the environment is not
degenerate.
"""
import itertools

import numpy as np

from horizon.env.episode import expected_wealth, is_legal, wealth_path
from horizon.env.exit_dist import ExitDistribution
from horizon.env.market import CASH, Market
from horizon.env.rules import burn, legal_targets
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
        candidate = list(sequence)
        if not is_legal(candidate):
            continue
        value = expected_wealth(market, candidate, dist.pmf)
        if value > best_value:
            best_value, best_sequence = value, candidate
    return best_value, best_sequence


def monte_carlo_value(market: Market, dist: ExitDistribution, sol: Solution, n: int, seed: int):
    """Roll out pi* against sampled exit times; returns (mean, standard error)."""
    rng = np.random.default_rng(seed)
    days = np.arange(1, dist.N + 1)
    draws = rng.choice(days, size=n, p=dist.pmf[1:])
    path = wealth_path(market, optimal_holdings(sol, market, dist.N))
    samples = path[draws]
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n))


def reachable_days(N: int, n_assets: int = 4) -> dict:
    """Map each (holding, age, mask) state to the days on which it can occur.

    Enumerated forward from the day-1 CASH start under the no-re-entry rule.
    Using true reachability matters: scoring over the full index space counts
    states no legal play can ever produce, and the optimal action in those states
    is arbitrary, which adds noise that swamps the signal the check is after.
    """
    current = {(CASH, 0, 0)}
    days_of = {}
    for t in range(1, N + 1):
        for state in current:
            days_of.setdefault(state, []).append(t)
        following = set()
        for holding, age, mask in current:
            following.add((holding, age + 1, mask))
            for target in legal_targets(mask, holding, n_assets):
                following.add((target, 1, burn(mask, holding)))
        current = following
    return days_of


def time_varying_fraction(sol: Solution, dist: ExitDistribution) -> float:
    """Fraction of reachable states whose optimal action depends on the day."""
    N = dist.N
    varying = total = 0
    for (holding, age, mask), days in reachable_days(N, sol.pi.shape[1]).items():
        if len(days) < 2 or age > N - 1:
            continue
        total += 1
        if len({int(sol.pi[t, holding, age, mask]) for t in days}) > 1:
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
