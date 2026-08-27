import numpy as np
import pytest

from horizon.env.distributions import bimodal_exit, uniform_exit
from horizon.env.market import CASH
from horizon.eval.metrics import (
    EpisodeRecord,
    belief_error_curve,
    clock_coefficient,
    event_timing_score,
    information_intensity,
    paired_bootstrap_ci,
    paired_permutation_p,
    regret,
    switch_day_histogram,
    trade_alignment,
)


def record(episode=0, exit_day=50, wealth=1.0, holdings=None, beliefs=(), switches=()):
    return EpisodeRecord(
        episode=episode,
        true_exit_day=exit_day,
        terminal_wealth=wealth,
        holdings=tuple(holdings or [0] * 100),
        beliefs=tuple(beliefs),
        switch_days=tuple(switches),
    )


def test_regret_is_zero_for_an_agent_that_achieves_v_star():
    assert regret([record(wealth=1.2), record(wealth=1.2)], v_star=1.2) == pytest.approx(0.0)


def test_regret_is_positive_for_a_worse_agent():
    assert regret([record(wealth=1.0)], v_star=1.2) == pytest.approx(0.2)


def test_paired_bootstrap_brackets_a_known_difference():
    # The improvement must carry its own noise, or the paired differences are a
    # constant and the bootstrap CI collapses to zero width.
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.05, 400)
    better = base + rng.normal(0.10, 0.02, 400)
    mean, lo, hi = paired_bootstrap_ci(better, base)
    assert lo <= mean <= hi
    assert lo <= 0.10 <= hi
    assert lo > 0.0  # the improvement is detected as real


def test_paired_bootstrap_requires_aligned_samples():
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_permutation_test_detects_a_real_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(1.0, 0.02, 200)
    assert paired_permutation_p(base + 0.05, base) < 0.01


def test_permutation_test_does_not_cry_wolf_on_identical_samples():
    rng = np.random.default_rng(2)
    base = rng.normal(1.0, 0.02, 200)
    assert paired_permutation_p(base, base.copy()) > 0.5


def test_clock_coefficient_is_zero_for_a_temporally_rigid_agent():
    # Holds one asset forever: portfolio composition does not track the clock.
    assert clock_coefficient([record(holdings=[1] * 100)]) == pytest.approx(0.0)


def test_clock_coefficient_is_negative_when_the_agent_de_risks_over_time():
    holdings = [0] * 50 + [CASH] * 50
    assert clock_coefficient([record(holdings=holdings)]) < 0


def test_switch_day_histogram_counts_by_day():
    counts = switch_day_histogram([record(switches=(11,)), record(switches=(11, 40))])
    assert counts[11] == 2
    assert counts[40] == 1


def test_event_timing_lift_exceeds_one_for_an_event_timed_agent():
    # Every survivor switches exactly at the event and nowhere else.
    records = [record(episode=i, exit_day=95, switches=(11,)) for i in range(50)]
    score = event_timing_score(records, event_day=11)
    assert score["in_window"] == pytest.approx(1.0)
    assert score["lift"] > 1.0


def test_event_timing_ignores_episodes_that_ended_before_the_event():
    records = [record(episode=i, exit_day=5, switches=()) for i in range(10)]
    assert event_timing_score(records, event_day=11)["n"] == 0


def test_belief_error_is_zero_for_a_perfectly_calibrated_agent():
    dist = uniform_exit(100)
    truth = [dist.mean_remaining(t) for t in range(1, 101)]
    curve, counts = belief_error_curve([record(beliefs=truth)], dist)
    assert np.nanmax(curve) == pytest.approx(0.0)
    assert counts[0] == 1


def test_belief_error_grows_with_a_biased_agent():
    dist = uniform_exit(100)
    biased = [dist.mean_remaining(t) + 10.0 for t in range(1, 101)]
    curve, _ = belief_error_curve([record(beliefs=biased)], dist)
    assert np.nanmean(curve) == pytest.approx(10.0)


def test_information_intensity_spikes_at_the_concentrated_event():
    intensity = information_intensity(bimodal_exit())
    assert int(np.argmax(intensity)) + 1 == 10
    assert intensity.max() > 20 * intensity.mean()


def test_trade_alignment_is_about_one_for_information_blind_trading():
    # Switching on an arbitrary quiet day should score near the blind baseline.
    records = [record(episode=i, exit_day=95, switches=(50,)) for i in range(20)]
    assert trade_alignment(records, bimodal_exit()) < 1.0


def test_trade_alignment_is_large_for_event_timed_trading():
    records = [record(episode=i, exit_day=95, switches=(10,)) for i in range(20)]
    assert trade_alignment(records, bimodal_exit()) > 5.0
