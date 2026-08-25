import numpy as np
import pytest

from horizon.env.episode import ages_for, expected_wealth, wealth_path
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


def test_late_bloomer_pays_nothing_until_its_breakpoint():
    m = default_market()
    c = m.index("C")
    w = wealth_path(m, [c] * 25)
    # Day 1 pays C's zero early rate but still charges the switch cost out of CASH.
    assert w[1] == pytest.approx(0.999)
    # Nothing accrues through the 20-day dead period.
    assert w[20] == pytest.approx(0.999)
    # Past the breakpoint it compounds at the late rate.
    assert w[25] == pytest.approx(0.999 * 1.008 ** 5)


def test_switching_charges_the_cost_once_per_switch():
    m = default_market()
    a, b = m.index("A"), m.index("B")
    w = wealth_path(m, [a, a, b])
    # Switch into A (cost), hold A (free), switch to B (cost).
    expected = 0.999 * 1.003 * 1.003 * 0.999 * 1.006
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
