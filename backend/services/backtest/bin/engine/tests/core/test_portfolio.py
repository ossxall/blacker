from src.core.portfolio import Portfolio, Side
from src.orders import OrderRole, OrderManager, RiskManager, Signal
import json
import pytest


BRACKET = {
    "stop": {"type": "percent", "value": 0.01},
    "targets": [{"type": "percent", "value": 0.02}],
}


class FakeTick:
    def __init__(self, price, index=0, time=0):
        self.price = price
        self.tick_index = index
        self.time = time


def account(initial_cash=100_000.0):
    return OrderManager(
        portfolio=Portfolio(initial_cash),
        risk_manager=RiskManager(BRACKET),
    )


def fill(manager, order, price, tick):
    manager.on_fill(manager.make_fill(order, price, order.quantity, tick))


def open_position(manager, side, price, quantity=1):
    entry = manager.handle(Signal(action=side, quantity=quantity))[0]
    fill(manager, entry, price, FakeTick(price))
    return manager.portfolio


def target_of(manager):
    # ``==`` and not ``is``: conftest makes ``src.orders`` and ``orders``
    # two module objects, so the enum members are equal but not identical.
    return [o for o in manager.working_orders() if o.role == OrderRole.TARGET][0]


def test_cash_is_the_available_balance_not_the_account_value():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)

    assert portfolio.cash == 10_000.0
    assert portfolio.equity() == 100_000.0


def test_equity_follows_the_mark_of_an_open_long():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)

    portfolio.mark(95_000.0)

    assert portfolio.cash == 10_000.0
    assert portfolio.unrealized_pnl() == 5_000.0
    assert portfolio.equity() == 105_000.0


def test_equity_follows_the_mark_of_an_open_short():
    manager = account()

    portfolio = open_position(manager, "SELL", 90_000.0)

    assert portfolio.cash == 190_000.0

    portfolio.mark(85_000.0)

    assert portfolio.unrealized_pnl() == 5_000.0
    assert portfolio.equity() == 105_000.0

    portfolio.mark(95_000.0)

    assert portfolio.unrealized_pnl() == -5_000.0
    assert portfolio.equity() == 95_000.0


def test_equity_never_moves_with_the_tick_while_flat():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)

    fill(manager, target_of(manager), 91_800.0, FakeTick(91_800.0))

    assert portfolio.position is None
    assert portfolio.realized_pnl == 1_800.0

    portfolio.mark(1.0)
    portfolio.mark(500_000.0)

    assert portfolio.equity() == 101_800.0
    assert portfolio.unrealized_pnl() == 0.0


def test_equity_accounts_for_realized_plus_open_pnl():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)

    fill(manager, target_of(manager), 91_800.0, FakeTick(91_800.0))

    portfolio = open_position(manager, "BUY", 92_000.0)
    portfolio.mark(94_000.0)

    assert portfolio.cash == 9_800.0
    assert portfolio.realized_pnl == 1_800.0
    assert portfolio.unrealized_pnl() == 2_000.0
    assert portfolio.equity() == 103_800.0
    assert portfolio.equity() == (
        portfolio.initial_cash + portfolio.realized_pnl + portfolio.unrealized_pnl()
    )


def test_unrealized_pnl_is_zero_until_the_book_is_marked():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)

    assert portfolio.mark_price is None
    assert portfolio.unrealized_pnl() == 0.0
    assert portfolio.equity() == 100_000.0


@pytest.mark.parametrize(
    "price", [0.0, -1.0, float("nan"), float("inf"), None, "n/a"]
)
def test_mark_ignores_prices_that_are_not_tradable(price):
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)

    portfolio.mark(95_000.0)
    portfolio.mark(price)

    assert portfolio.mark_price == 95_000.0
    assert portfolio.equity() == 105_000.0


def test_payload_publishes_equity_and_open_pnl():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)
    portfolio.mark(95_000.0)

    data = portfolio.to_dict()

    assert data["cash"] == 10_000.0
    assert data["mark_price"] == 95_000.0
    assert data["unrealized_pnl"] == 5_000.0
    assert data["equity"] == 105_000.0
    assert data["position"]["side"] == Side.BUY.value


def test_flat_payload_publishes_equity_equal_to_cash():
    portfolio = Portfolio(100_000.0)

    portfolio.mark(95_000.0)

    data = portfolio.to_dict()

    assert data["position"] is None
    assert data["unrealized_pnl"] == 0.0
    assert data["equity"] == 100_000.0


def test_derived_values_are_not_restored_from_the_payload():
    portfolio = Portfolio(100_000.0)
    portfolio.mark(95_000.0)

    stale = portfolio.to_dict()
    stale["equity"] = 1.0
    stale["unrealized_pnl"] = 1.0
    stale["mark_price"] = 1.0

    restored = Portfolio.from_dict(stale)

    # The mark is restored, but equity is recomputed from the position
    # and that mark: a snapshot may not overwrite the live valuation.
    assert restored.mark_price == 1.0
    assert restored.equity() == 100_000.0
    assert restored.unrealized_pnl() == 0.0


def test_round_trip_keeps_equity_and_open_pnl():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)
    portfolio.mark(95_000.0)

    data = json.loads(json.dumps(portfolio.to_dict()))
    restored = Portfolio.from_dict(data)

    # The mark travels inside the payload, so a resumed backtest can
    # value its position before its first tick.
    assert restored.to_dict() == portfolio.to_dict()
    assert restored.cash == 10_000.0
    assert restored.equity() == 105_000.0


def test_legacy_payload_without_a_mark_still_loads():
    manager = account()

    portfolio = open_position(manager, "BUY", 90_000.0)
    portfolio.mark(95_000.0)

    legacy = portfolio.to_dict()
    legacy.pop("mark_price")
    legacy.pop("equity")
    legacy.pop("unrealized_pnl")

    restored = Portfolio.from_dict(legacy)

    assert restored.mark_price is None
    assert restored.cash == 10_000.0
    assert restored.unrealized_pnl() == 0.0
    assert restored.equity() == 100_000.0
