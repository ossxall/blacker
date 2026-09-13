from src.ingestion.tick import Tick
from src.core.portfolio import Portfolio
from src.orders import (
    OrderManager,
    OrderRole,
    OrderStatus,
    OrderType,
    RiskManager,
    Side,
    Signal,
)


def make_tick(index, price, time=0):
    return Tick(
        boot_id="boot",
        config_id="config",
        tick_index=index,
        trade_id=index,
        time=time,
        price=price,
        qty=1.0,
        is_buyer_maker=0,
    )


def fill_entry(manager, order, price, tick):
    fill = manager.make_fill(order, price, order.quantity, tick)
    return manager.on_fill(fill)


def fill_order(manager, order, price, tick):
    fill = manager.make_fill(order, price, order.quantity, tick)
    manager.on_fill(fill)
    return manager


def find_role(orders, role):
    return [o for o in orders if o.role == role]


def test_buy_signal_creates_market_entry_order():
    manager = OrderManager(portfolio=Portfolio())

    orders = manager.handle(Signal(action="BUY", quantity=1))

    assert len(orders) == 1

    entry = orders[0]
    assert entry.side == Side.BUY
    assert entry.type == OrderType.MARKET
    assert entry.role == OrderRole.ENTRY
    assert entry.quantity == 1.0


def test_entry_signal_ignored_while_position_is_open():
    manager = OrderManager(portfolio=Portfolio())

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    assert manager.portfolio.position is not None

    assert manager.handle(Signal(action="BUY", quantity=1)) == []
    assert manager.handle(Signal(action="SELL", quantity=1)) == []


def test_entry_fill_creates_stop_and_target_bracket():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [
            {"type": "percent", "value": 0.02},
            {"type": "percent", "value": 0.03},
        ],
    }
    manager = OrderManager(
        portfolio=Portfolio(),
        risk_manager=RiskManager(config),
    )

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    working = manager.working_orders()

    stops = find_role(working, OrderRole.STOP)
    targets = find_role(working, OrderRole.TARGET)

    assert len(stops) == 1
    assert stops[0].price == 99.0
    assert stops[0].quantity == 1.0

    assert sorted(t.price for t in targets) == [102.0, 103.0]
    assert all(t.quantity == 0.5 for t in targets)

    position = manager.portfolio.position
    assert position is not None
    assert position.side == Side.BUY
    assert position.avg_price == 100.0


def test_target_fill_closes_position_and_cancels_siblings():
    config = {
        "targets": [{"type": "percent", "value": 0.03}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    target = find_role(manager.working_orders(), OrderRole.TARGET)[0]
    assert target.price == 103.0

    fill_order(manager, target, 103.0, make_tick(1, 103.0))

    assert manager.portfolio.position is None
    assert manager.working_orders() == []
    assert manager.portfolio.realized_pnl == 3.0


def test_stop_fill_closes_position_as_loss():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [{"type": "percent", "value": 0.03}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    stop = find_role(manager.working_orders(), OrderRole.STOP)[0]
    assert stop.price == 99.0

    fill_order(manager, stop, 99.0, make_tick(1, 99.0))

    assert manager.portfolio.position is None
    assert manager.working_orders() == []
    assert manager.portfolio.realized_pnl == -1.0


def test_partial_target_reduces_stop_quantity_then_stop_closes():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [
            {"type": "percent", "value": 0.02},
            {"type": "percent", "value": 0.04},
        ],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    first_target = find_role(manager.working_orders(), OrderRole.TARGET)[0]
    fill_order(manager, first_target, 102.0, make_tick(1, 102.0))

    assert manager.portfolio.position.quantity == 0.5
    assert manager.portfolio.realized_pnl == 1.0

    stop = find_role(manager.working_orders(), OrderRole.STOP)[0]
    assert stop.quantity == 0.5

    fill_order(manager, stop, 99.0, make_tick(2, 99.0))

    assert manager.portfolio.position is None
    assert manager.working_orders() == []


def test_exit_signal_cancels_bracket_and_submits_exit_order():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [{"type": "percent", "value": 0.03}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    assert len(manager.working_orders()) == 2

    orders = manager.handle(Signal(action="EXIT"))

    assert len(orders) == 1
    assert orders[0].role == OrderRole.EXIT
    assert orders[0].side == Side.SELL
    assert orders[0].type == OrderType.MARKET
    assert orders[0].status == OrderStatus.WORKING

    assert manager.working_orders() == orders


def test_trailing_stop_ratchets_up_for_long():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "trailing": {
            "enabled": True,
            "distance": {"type": "percent", "value": 0.005},
        },
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    stop = find_role(manager.working_orders(), OrderRole.STOP)[0]
    assert stop.price == 99.0
    assert stop.trailing is True
    assert stop.trailing_distance == 0.5

    manager.update_trailing_stops(101.0)
    assert stop.price == 100.5

    manager.update_trailing_stops(100.0)
    assert stop.price == 100.5


def test_serialization_roundtrip():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [{"type": "percent", "value": 0.03}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    restored = OrderManager.from_dict(
        manager.to_dict(),
        portfolio=manager.portfolio,
        risk_manager=manager.risk_manager,
    )

    assert len(restored.working_orders()) == len(manager.working_orders())
    assert restored.to_dict() == manager.to_dict()