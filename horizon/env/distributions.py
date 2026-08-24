"""The exit distributions the study runs on.

Only non-memoryless distributions are used for headline conditions: a constant
hazard would collapse the objective into ordinary discounting and test nothing
temporal (README section 3.1). Geometric is retained purely as a control that
should produce stationary behaviour.
"""
import numpy as np

from horizon.env.exit_dist import ExitDistribution


def uniform_exit(N: int = 100) -> ExitDistribution:
    """T ~ Uniform{1..N}. Hazard rises smoothly from 1/N to 1: gradual updating."""
    pmf = np.zeros(N + 1)
    pmf[1:] = 1.0 / N
    return ExitDistribution("uniform", pmf)


def bimodal_exit(
    N: int = 100,
    early: tuple = (1, 10),
    late: tuple = (91, 100),
    w: float = 0.5,
) -> ExitDistribution:
    """Mass split between an early and a late window, with a dead zone between.

    Surviving the early window fully resolves the mode, so expected remaining
    time jumps discontinuously. That bright line is the single most interpretable
    behavioural checkpoint in the study.
    """
    pmf = np.zeros(N + 1)
    e0, e1 = early
    l0, l1 = late
    pmf[e0 : e1 + 1] = w / (e1 - e0 + 1)
    pmf[l0 : l1 + 1] = (1.0 - w) / (l1 - l0 + 1)
    return ExitDistribution("bimodal", pmf)


def geometric_exit(N: int = 100, p: float = 0.02) -> ExitDistribution:
    """Constant hazard p, truncated at N with the residual mass piled on day N."""
    pmf = np.zeros(N + 1)
    days = np.arange(1, N)
    pmf[1:N] = p * (1.0 - p) ** (days - 1)
    pmf[N] = (1.0 - p) ** (N - 1)
    return ExitDistribution("geometric", pmf)


def sawtooth_exit(
    N: int = 100, period: int = 20, lo: float = 0.005, hi: float = 0.08
) -> ExitDistribution:
    """Hazard ramps up within each period then resets, forced to 1 on day N.

    Built hazard-first and converted to a pmf, because the shape is defined by
    its hazard. Structurally unlike any textbook lifecycle case, which is the
    point: it separates reciting a known heuristic from reasoning (T9).
    """
    hazard = np.zeros(N + 1)
    days = np.arange(1, N)
    hazard[1:N] = lo + (hi - lo) * (((days - 1) % period) / period)
    hazard[N] = 1.0

    pmf = np.zeros(N + 1)
    surviving = 1.0
    for t in range(1, N + 1):
        pmf[t] = surviving * hazard[t]
        surviving -= pmf[t]
    return ExitDistribution("sawtooth", pmf)


DISTRIBUTIONS = {
    "uniform": uniform_exit,
    "bimodal": bimodal_exit,
    "geometric": geometric_exit,
    "sawtooth": sawtooth_exit,
}
