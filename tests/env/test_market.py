import pytest

from horizon.env.market import CASH, HOLDINGS, AssetSpec, Market, default_market


def test_holding_order_is_fixed():
    # Index order is relied on by the solver and every logged row.
    assert HOLDINGS == ("A", "B", "C", "CASH")
    assert CASH == 3


def test_sprinter_decays_after_its_boost_window():
    # README section 4: A is +0.8%/day for the first 10 days held, then -0.2%/day.
    a = AssetSpec("A", early_rate=0.008, late_rate=-0.002, breakpoint=10)
    assert a.daily_return(1) == pytest.approx(0.008)
    assert a.daily_return(10) == pytest.approx(0.008)
    assert a.daily_return(11) == pytest.approx(-0.002)


def test_compounder_is_flat_at_every_age():
    b = AssetSpec("B", early_rate=0.0025, late_rate=0.0025, breakpoint=1)
    for age in (1, 40, 99):
        assert b.daily_return(age) == pytest.approx(0.0025)


def test_late_bloomer_improves_after_its_breakpoint():
    c = AssetSpec("C", early_rate=0.0005, late_rate=0.006, breakpoint=40)
    assert c.daily_return(40) == pytest.approx(0.0005)
    assert c.daily_return(41) == pytest.approx(0.006)


def test_cash_never_moves():
    m = default_market()
    for age in (1, 50, 100):
        assert m.growth(CASH, age) == pytest.approx(1.0)


def test_growth_is_one_plus_daily_return():
    m = default_market()
    assert m.growth(m.index("A"), 1) == pytest.approx(1.008)
    assert m.growth(m.index("A"), 11) == pytest.approx(0.998)


def test_switch_cost_defaults_to_ten_basis_points():
    assert default_market().switch_cost == pytest.approx(0.001)


def test_age_must_be_positive():
    m = default_market()
    with pytest.raises(ValueError, match="age"):
        m.growth(0, 0)
