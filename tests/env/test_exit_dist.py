import numpy as np
import pytest

from horizon.env.exit_dist import ExitDistribution


def make_uniform(n=100):
    pmf = np.zeros(n + 1)
    pmf[1:] = 1.0 / n
    return ExitDistribution("uniform", pmf)


def test_survival_is_probability_of_outlasting_t():
    d = make_uniform(100)
    assert d.survival(0) == pytest.approx(1.0)
    assert d.survival(50) == pytest.approx(0.5)
    assert d.survival(100) == pytest.approx(0.0)


def test_uniform_hazard_matches_closed_form():
    # Closed form from README section 3.1: h(t) = 1 / (N - t + 1)
    d = make_uniform(100)
    for t in (1, 37, 99, 100):
        assert d.hazard(t) == pytest.approx(1.0 / (100 - t + 1))
    assert d.hazard(100) == pytest.approx(1.0)


def test_uniform_mean_remaining_matches_closed_form():
    # Closed form: remaining time is Uniform{1..N-t}, mean (N - t + 1) / 2
    d = make_uniform(100)
    for t in (0, 1, 50, 99):
        assert d.mean_remaining(t) == pytest.approx((100 - t + 1) / 2)
    assert d.mean_remaining(100) == pytest.approx(0.0)


def test_pmf_must_sum_to_one():
    bad = np.zeros(11)
    bad[1:] = 0.05
    with pytest.raises(ValueError, match="sum to 1"):
        ExitDistribution("bad", bad)


def test_pmf_index_zero_must_be_unused():
    bad = np.zeros(11)
    bad[0] = 0.5
    bad[1:] = 0.05
    with pytest.raises(ValueError, match="index 0"):
        ExitDistribution("bad", bad)


def test_pmf_must_be_non_negative():
    bad = np.zeros(11)
    bad[1:] = 0.1
    bad[3] = -0.1
    bad[4] = 0.3
    with pytest.raises(ValueError, match="non-negative"):
        ExitDistribution("bad", bad)
