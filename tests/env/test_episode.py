import numpy as np
import pytest

from horizon.env.episode import ages_for, expected_wealth, is_legal, wealth_path
from horizon.env.market import CASH, default_market


def test_ages_reset_on_switch_and_increment_on_hold():
    a = 0
    # Start in CASH (age 0 before day 1). Holding CASH on day 1 makes its age 1.
    assert ages_for([CASH, CASH, a, a, a]) == [1, 2, 1, 2, 3]


def test_holding_cash_forever_preserves_wealth():
    m = default_market()
    w = wealth_path(m, [CASH] * 10)
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(1.0)


def test_sprinter_compounds_then_decays():
    m = default_market()
    a = m.index("A")
    w = wealth_path(m, [a] * 12)
    # Day 1 pays the boost rate but also the switch cost out of CASH.
    assert w[1] == pytest.approx(0.999 * 1.008)
    # Days 11 and 12 are past the breakpoint and shrink wealth.
    assert w[12] < w[10]


def test_re_entry_is_rejected_rather_than_priced():
    m = default_market()
    a, b = m.index("A"), m.index("B")
    with pytest.raises(ValueError, match="illegal re-entry"):
        wealth_path(m, [a, b, a])


def test_cash_may_be_re_entered_freely():
    m = default_market()
    a = m.index("A")
    assert is_legal([CASH, a, CASH])
    wealth_path(m, [CASH, a, CASH])


def test_is_legal_matches_what_wealth_path_accepts():
    m = default_market()
    a, b = m.index("A"), m.index("B")
    assert is_legal([a, a, b, b])
    assert not is_legal([a, b, a])


def test_switching_charges_the_cost_once_per_switch():
    m = default_market()
    a, b = m.index("A"), m.index("B")
    w = wealth_path(m, [a, a, b])
    # Switch into A (cost), hold A (free), switch to B (cost).
    expected = 0.999 * 1.008 * 1.008 * 0.999 * 1.0025
    assert w[3] == pytest.approx(expected)


def test_staying_in_cash_from_the_start_is_free():
    m = default_market()
    # The agent is auto-entered holding CASH, so day 1 in CASH is not a switch.
    assert wealth_path(m, [CASH])[1] == pytest.approx(1.0)


def test_expected_wealth_weights_the_path_by_the_exit_pmf():
    m = default_market()
    pmf = np.zeros(4)
    pmf[2] = 0.5
    pmf[3] = 0.5
    w = wealth_path(m, [CASH, CASH, CASH])
    assert expected_wealth(m, [CASH, CASH, CASH], pmf) == pytest.approx(0.5 * w[2] + 0.5 * w[3])
