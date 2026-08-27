"""The metrics from README section 8 and DESIGN.md Part IV.

Every metric here is computed from logged episode records alone, so the whole
analysis is reproducible from artifacts without touching the database.
"""
from dataclasses import dataclass

import numpy as np

from horizon.env.exit_dist import ExitDistribution
from horizon.env.market import CASH


@dataclass(frozen=True)
class EpisodeRecord:
    """One completed episode, as logged."""

    episode: int
    true_exit_day: int
    terminal_wealth: float
    holdings: tuple
    beliefs: tuple = ()
    switch_days: tuple = ()


def regret(records, v_star: float) -> float:
    """V* minus mean realised terminal wealth. The headline metric.

    Reported instead of raw PnL because "lost money" is uninterpretable, whereas
    regret against an exactly-solved optimum is not (README section 3.2).
    """
    if not records:
        return float("nan")
    return v_star - float(np.mean([r.terminal_wealth for r in records]))


def paired_bootstrap_ci(a, b, n_resamples: int = 10000, seed: int = 20260814, alpha: float = 0.05):
    """Bootstrap CI for the paired mean difference a - b.

    Paired on episode index because every agent faces the identical set of exit
    realisations (common random numbers, EXPERIMENTS.md T5.1), which removes the
    world-to-world variance that would otherwise dominate.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must align: {a.shape} vs {b.shape}")
    differences = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    means = differences[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(differences.mean()), float(lo), float(hi)


def paired_permutation_p(a, b, n_permutations: int = 10000, seed: int = 20260814) -> float:
    """Two-sided paired permutation test on the mean difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    differences = a - b
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n_permutations + 1))


def clock_coefficient(records, n_assets: int = 4) -> float:
    """Slope of holding-riskiness on elapsed day, pooled across episodes.

    Answers README section 8.2: does the agent's portfolio depend on the clock at
    all? A temporally rigid agent scores about zero. Holdings are scored on an
    ordinal risk scale so the regression has a meaningful direction.
    """
    risk = {0: 3.0, 1: 2.0, 2: 2.0, CASH: 0.0}
    days, scores = [], []
    for record in records:
        for day, holding in enumerate(record.holdings, start=1):
            days.append(day)
            scores.append(risk.get(holding, 1.0))
    if len(set(days)) < 2:
        return 0.0
    days_arr = np.asarray(days, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    centred = days_arr - days_arr.mean()
    denominator = float((centred**2).sum())
    if denominator == 0.0:
        return 0.0
    return float((centred * (scores_arr - scores_arr.mean())).sum() / denominator)


def switch_day_histogram(records, horizon: int = 100):
    """Counts of switches by day, for the Figure 2 histograms."""
    counts = np.zeros(horizon + 1, dtype=int)
    for record in records:
        for day in record.switch_days:
            if 1 <= day <= horizon:
                counts[day] += 1
    return counts


def event_timing_score(records, event_day: int, window: int = 5) -> dict:
    """Does the agent trade at the information event more than its own base rate?

    Compares P(switch within [event_day, event_day+window) | survived) against the
    agent's own per-day switch rate elsewhere, so a hyperactive trader cannot
    score well simply by switching constantly (README section 8.4).
    """
    survivors = [r for r in records if r.true_exit_day >= event_day]
    if not survivors:
        return {"in_window": 0.0, "base_rate": 0.0, "lift": float("nan"), "n": 0}

    window_days = set(range(event_day, event_day + window))
    in_window = sum(
        1 for r in survivors if any(d in window_days for d in r.switch_days)
    ) / len(survivors)

    outside_switches = outside_days = 0
    for record in survivors:
        alive = min(record.true_exit_day, len(record.holdings))
        outside_days += max(0, alive - len(window_days))
        outside_switches += sum(1 for d in record.switch_days if d not in window_days)
    # Laplace-smoothed so an agent that switches ONLY at the event still yields a
    # finite lift rather than dividing by a zero base rate.
    base_rate = (outside_switches + 1.0) / (outside_days + 1.0 / len(window_days))
    base_rate = min(base_rate, 1.0)
    expected = 1.0 - (1.0 - base_rate) ** len(window_days)

    return {
        "in_window": in_window,
        "base_rate": expected,
        "lift": (in_window / expected) if expected > 0 else float("nan"),
        "n": len(survivors),
    }


def belief_error_curve(records, dist: ExitDistribution):
    """Mean |stated belief - true E[T-t | T>t]| by day (Figure 4).

    Separates belief-formation failures from belief-to-action failures: an agent
    whose beliefs track m(t) but whose actions do not is a different finding from
    one whose beliefs are wrong in the first place (DESIGN.md P7).
    """
    truth = np.array([dist.mean_remaining(t) for t in range(dist.N + 1)])
    totals = np.zeros(dist.N + 1)
    counts = np.zeros(dist.N + 1)
    for record in records:
        for day, belief in enumerate(record.beliefs, start=1):
            if belief is None or day > dist.N:
                continue
            totals[day] += abs(belief - truth[day])
            counts[day] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return curve[1:], counts[1:]


def information_intensity(dist: ExitDistribution):
    """I(t) = |m(t) - m(t-1)|, the belief-change measure from DESIGN.md P4.

    Only meaningful when the early mode is concentrated; with mass spread over a
    window it does not peak at the event at all (see the P4 amendment).
    """
    m = np.array([dist.mean_remaining(t) for t in range(dist.N + 1)])
    return np.abs(np.diff(m))


def trade_alignment(records, dist: ExitDistribution) -> float:
    """Normalised alignment of trades with information intensity.

    About 1.0 for information-blind trading; much greater than 1 for event-timed
    trading.
    """
    intensity = information_intensity(dist)
    mean_intensity = float(intensity.mean())
    if mean_intensity <= 0:
        return float("nan")
    weighted = trades = 0.0
    for record in records:
        for day in record.switch_days:
            if 1 <= day <= len(intensity):
                weighted += intensity[day - 1]
                trades += 1
    if trades == 0:
        return float("nan")
    return float(weighted / (trades * mean_intensity))


def switch_counts(records):
    """Switches per episode, for the churn guard in DESIGN.md P4."""
    return np.array([len(r.switch_days) for r in records], dtype=float)
