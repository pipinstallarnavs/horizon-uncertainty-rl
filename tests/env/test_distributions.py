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


def test_bimodal_hazard_is_zero_outside_the_two_modes():
    # Hazard is zero everywhere except the concentrated early day and the late window.
    d = bimodal_exit()
    for t in list(range(1, 10)) + list(range(11, 91)):
        assert d.hazard(t) == pytest.approx(0.0)
    assert d.hazard(10) == pytest.approx(0.5)
    assert d.hazard(95) > 0.0


def test_bimodal_resolves_the_mode_sharply_at_day_ten():
    # With the early mass concentrated on day 10, surviving it is a coin flip that
    # resolves the mode in one step: expected remaining time jumps 43.75 -> 85.5.
    d = bimodal_exit()
    assert d.mean_remaining(9) == pytest.approx(43.75)
    assert d.mean_remaining(10) == pytest.approx(85.5)
    assert d.mean_remaining(10) - d.mean_remaining(9) > 40.0


def test_spread_early_mode_resolves_gradually_instead():
    # The rejected variant, kept as a regression guard: spreading the early mass
    # makes mean remaining time climb smoothly, which is why it has no bright line.
    d = bimodal_exit(early=(1, 10))
    curve = [d.mean_remaining(t) for t in range(0, 11)]
    assert all(later > earlier for earlier, later in zip(curve, curve[1:]))
    assert curve[10] - curve[9] < 10.0


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
    assert d.pmf[10] == pytest.approx(0.5)
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
