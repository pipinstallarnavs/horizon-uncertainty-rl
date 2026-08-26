import pytest

from horizon.env.market import CASH
from horizon.env.rules import burn, is_available, legal_targets


def test_leaving_a_risky_asset_spends_it():
    assert is_available(0, 0)
    spent = burn(0, 0)
    assert not is_available(spent, 0)


def test_leaving_cash_spends_nothing():
    assert burn(0, CASH) == 0
    assert is_available(burn(0, CASH), CASH)


def test_cash_is_always_available_even_after_being_left():
    mask = burn(burn(0, 0), CASH)
    assert is_available(mask, CASH)


def test_legal_targets_exclude_the_current_holding_and_spent_assets():
    mask = burn(0, 0)  # A has been left
    assert legal_targets(mask, 1, 4) == [2, CASH]


def test_every_risky_asset_can_be_spent_independently():
    mask = burn(burn(0, 0), 2)
    assert not is_available(mask, 0)
    assert is_available(mask, 1)
    assert not is_available(mask, 2)
