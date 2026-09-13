from src.ingestion.tick import Tick
from src.core.portfolio import Portfolio
from src.execution.execution import Execution
from src.orders import (
    OrderManager,
    OrderRole,
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


def run_ticks(execution, ticks):
    for tick in ticks:
        execution.update(state=None, tick=tick)


def test_market_entry_fills_next_tick():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager())
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    order = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([order])

    fills = execution.update(state=None, tick=make_tick(1, 101.0))

    assert len(fills) == 1
    assert fills[0].price == 101.0

    position = manager.portfolio.position
    assert position is not None
    assert position.quantity == 1.0
    assert position.avg_price == 101.0


def test_limit_target_triggers_when_price_crosses():
    config = {"targets": [{"type": "percent", "value": 0.03}]}
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])

    execution.update(state=None, tick=make_tick(1, 100.0))

    assert manager.portfolio.position is not None

    # Target is a SELL limit at 103. Below 103 it must not fill.
    run_ticks(execution, [make_tick(2, 102.0)])

    assert manager.portfolio.position is not None

    # A tick at/above the limit triggers the target.
    fills = execution.update(state=None, tick=make_tick(3, 103.0))

    assert len(fills) == 1
    assert fills[0].role == OrderRole.TARGET
    assert manager.portfolio.position is None


def test_stop_triggers_when_price_drops_below_stop():
    config = {"stop": {"type": "percent", "value": 0.01}}
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])

    execution.update(state=None, tick=make_tick(1, 100.0))

    assert manager.portfolio.position is not None

    # Above the stop: no trigger.
    run_ticks(execution, [make_tick(2, 99.5)])

    assert manager.portfolio.position is not None

    # At/below the stop: the position must close.
    fills = execution.update(state=None, tick=make_tick(3, 99.0))

    assert len(fills) == 1
    assert fills[0].role == OrderRole.STOP
    assert manager.portfolio.position is None
    assert manager.portfolio.realized_pnl == -1.0


def test_trailing_stop_ratchets_then_triggers():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "trailing": {
            "enabled": True,
            "distance": {"type": "percent", "value": 0.005},
        },
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])

    execution.update(state=None, tick=make_tick(1, 100.0))

    stop = [o for o in manager.working_orders() if o.role == OrderRole.STOP][0]
    assert stop.price == 99.0

    # Price rises: stop ratchets up to 100.5.
    execution.update(state=None, tick=make_tick(2, 101.0))
    assert stop.price == 100.5

    # A pullback below the trailing level closes the position.
    fills = execution.update(state=None, tick=make_tick(3, 100.0))

    assert len(fills) == 1
    assert fills[0].role == OrderRole.STOP
    assert manager.portfolio.position is None


def test_position_is_single_source_of_truth_after_stop():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [{"type": "percent", "value": 0.03}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])

    execution.update(state=None, tick=make_tick(1, 100.0))

    # A fresh BUY signal after the position exists must be ignored.
    assert manager.handle(Signal(action="BUY", quantity=1)) == []

    # Stop closes the position; a subsequent entry is allowed again.
    fills = execution.update(state=None, tick=make_tick(2, 99.0))
    assert len(fills) == 1
    assert manager.portfolio.position is None

    orders = manager.handle(Signal(action="BUY", quantity=1))
    assert len(orders) == 1