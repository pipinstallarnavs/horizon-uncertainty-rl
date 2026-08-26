"""The no-re-entry rule.

Once the agent leaves a risky asset it may never hold that asset again. Cash is
exempt: it may be entered and left freely.

This rule exists to make the age-clock exploit impossible rather than merely
expensive. Because return clocks reset on switching, any asset that pays more
when freshly held can otherwise be farmed by bouncing out and back -- the exact
optimum for the original parameters was A(10 days) -> B(1 day) -> A(10 days),
forever, which made the optimal policy identical at every t and destroyed the
temporal structure the study measures. Penalising the cycle with a higher switch
cost does not work: the cost needed to deter it also deters the single genuine
adaptation the experiment is about.

The rule is disclosed to the agent verbatim in the prompt.
"""
from horizon.env.market import CASH


def burn(mask: int, holding: int) -> int:
    """Mark `holding` as spent when the agent leaves it. Leaving cash spends nothing."""
    return mask if holding == CASH else mask | (1 << holding)


def is_available(mask: int, target: int) -> bool:
    """Cash is always available; a risky asset is available until it has been left."""
    return target == CASH or not (mask >> target) & 1


def legal_targets(mask: int, holding: int, n_assets: int) -> list:
    """Every holding the agent may switch into from `holding` under `mask`."""
    return [t for t in range(n_assets) if t != holding and is_available(mask, t)]
