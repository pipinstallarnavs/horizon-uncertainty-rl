import pytest

from horizon.env.distributions import (
    DISTRIBUTIONS,
    bimodal_exit,
    geometric_exit,
    sawtooth_exit,
    uniform_exit,
)


@pytest.mark.parametrize("factory", [uniform_exit, bimodal_exit, geometric_exit, sawtooth_exit])
def test_all_distributions_are_exhausted_by_day_N(factory):
    # README section 2.1: the distributions are constructed so P(T <= N) = 1.
    d = factory()
    assert d.survival(d.N) == pytest.approx(0.0)
    assert d.hazard(d.N) == pytest.approx(1.0)


def test_bimodal_hazard_is_zero_in_the_dead_zone():
    # DESIGN.md P1: hazard spikes on 1..10, is exactly zero on 11..90, spikes on 91..100.
    d = bimodal_exit()
    for t in range(11, 91):
        assert d.hazard(t) == pytest.approx(0.0)
    assert d.hazard(5) > 0.0
    assert d.hazard(95) > 0.0


def test_bimodal_resolves_the_mode_by_day_ten():
    # Mode resolution is complete at t=10, but it arrives GRADUALLY: each survived
    # day shifts posterior weight toward the late mode, so mean remaining time
    # climbs smoothly from 50.5 to 85.5 rather than jumping.
    d = bimodal_exit()
    assert d.mean_remaining(0) == pytest.approx(50.5)
    assert d.mean_remaining(10) == pytest.approx(85.5)
    curve = [d.mean_remaining(t) for t in range(0, 11)]
    assert all(later > earlier for earlier, later in zip(curve, curve[1:]))


def test_bimodal_bright_line_lives_in_the_hazard_not_the_mean():
    # Corrects README section 3.1 and DESIGN.md P4: what changes discontinuously
    # at day 11 is the hazard, not expected remaining time. Nothing is LEARNED at
    # day 11 (the mode resolved at day 10); the decision problem changes instead.
    # This matters because P4's I(t) = |m(t) - m(t-1)| does not peak at day 11,
    # so it cannot be used to score event-timed trading. See docs amendment.
    d = bimodal_exit()
    assert d.hazard(10) > 0.09
    assert d.hazard(11) == pytest.approx(0.0)

    m = [d.mean_remaining(t) for t in range(d.N + 1)]
    intensity = [abs(m[t] - m[t - 1]) for t in range(1, d.N + 1)]
    assert intensity[10] < max(intensity)
    assert intensity[9] == pytest.approx(max(intensity))


def test_bimodal_weight_splits_mass_between_modes():
    d = bimodal_exit(w=0.5)
    assert d.pmf[1:11].sum() == pytest.approx(0.5)
    assert d.pmf[91:101].sum() == pytest.approx(0.5)


def test_geometric_hazard_is_constant_before_truncation():
    # The control condition: constant hazard is what makes the policy stationary.
    d = geometric_exit(p=0.02)
    for t in (1, 25, 60, 99):
        assert d.hazard(t) == pytest.approx(0.02)


def test_sawtooth_hazard_resets_each_period():
    # Hazard climbs within a period then drops, so it matches no textbook case (T9).
    d = sawtooth_exit(period=20, lo=0.005, hi=0.08)
    assert d.hazard(20) > d.hazard(21)
    assert d.hazard(1) == pytest.approx(0.005)


def test_registry_exposes_every_distribution_by_name():
    assert set(DISTRIBUTIONS) == {"uniform", "bimodal", "geometric", "sawtooth"}
