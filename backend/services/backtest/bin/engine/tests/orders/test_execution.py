from src.ingestion.tick import Tick
from src.core.portfolio import Portfolio
from src.execution.execution import Execution
from src.orders import (
    OrderManager,
    OrderRole,
    OrderStatus,
    RiskManager,
    Side,
    Signal,
)
import pytest


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


def test_a_static_stop_holds_its_level_and_triggers_there():
    config = {"stop": {"type": "percent", "value": 0.01}}
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])

    execution.update(state=None, tick=make_tick(1, 100.0))

    stop = [o for o in manager.working_orders() if o.role == OrderRole.STOP][0]
    assert stop.price == 99.0

    # Price rises. The stop is static, so it does not follow: a trade that
    # runs in favour is still protected at the level the entry priced.
    execution.update(state=None, tick=make_tick(2, 101.0))
    assert stop.price == 99.0

    # The pullback to the entry level is enough to take it out.
    fills = execution.update(state=None, tick=make_tick(3, 99.0))

    assert len(fills) == 1
    assert fills[0].role == OrderRole.STOP
    assert manager.portfolio.position is None


def test_limit_target_fills_at_its_limit_not_at_the_print():
    # Regression: a crossed take-profit used to book the whole gap as profit.
    # A sell limit at 103 filled at whatever the tick printed, so a tick at
    # 150 realized +50 instead of +3 and every backtest looked better than it
    # was.
    config = {"targets": [{"type": "percent", "value": 0.03}]}
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])
    execution.update(state=None, tick=make_tick(1, 100.0))

    target = [o for o in manager.working_orders() if o.role == OrderRole.TARGET][0]
    assert target.price == 103.0

    fills = execution.update(state=None, tick=make_tick(2, 150.0))

    assert len(fills) == 1
    assert fills[0].price == 103.0
    assert manager.portfolio.realized_pnl == 3.0


def test_buy_limit_fills_at_the_price_when_it_improves():
    config = {"targets": [{"type": "absolute", "value": 3.0}]}
    manager = OrderManager(
        portfolio=Portfolio(),
        risk_manager=RiskManager(config),
    )
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    # A short entry gets a buy limit target at 97.
    entry = manager.handle(Signal(action="SELL", quantity=1))[0]
    execution.submit([entry])
    execution.update(state=None, tick=make_tick(1, 100.0))

    target = [o for o in manager.working_orders() if o.role == OrderRole.TARGET][0]
    assert target.price == 97.0

    # The market gaps through the limit: the fill happens at the print.
    fills = execution.update(state=None, tick=make_tick(2, 90.0))

    assert len(fills) == 1
    assert fills[0].price == 90.0
    assert manager.portfolio.realized_pnl == 10.0


def test_stop_fills_at_the_print_even_when_it_gaps():
    config = {"stop": {"type": "percent", "value": 0.01}}
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])
    execution.update(state=None, tick=make_tick(1, 100.0))

    # A stop is not a limit: once triggered it is a market order, so the
    # downside of a gap is not clipped at the stop price.
    fills = execution.update(state=None, tick=make_tick(2, 80.0))

    assert len(fills) == 1
    assert fills[0].price == 80.0
    assert manager.portfolio.realized_pnl == -20.0


def test_order_with_nothing_left_to_fill_is_retired():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [{"type": "percent", "value": 0.02}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])
    execution.update(state=None, tick=make_tick(1, 100.0))

    target = [o for o in manager.working_orders() if o.role == OrderRole.TARGET][0]

    # The group is flat while the order is still working. It must be retired,
    # not left in the published book with a fill quantity of zero.
    manager.groups()[0].remaining_quantity = 0.0

    fills = execution.update(state=None, tick=make_tick(2, 103.0))

    assert fills == []
    assert target.status == OrderStatus.CANCELLED
    assert target not in manager.working_orders()
    assert manager.portfolio.position is not None


def test_no_fill_is_reported_for_an_order_retired_mid_pass():
    # Regression: working_orders() is a snapshot, so an order cancelled by an
    # earlier fill of the same pass was still triggered. It produced a fill
    # that never touched the portfolio, and the fill id counter skipped.
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [
            {"type": "percent", "value": 0.02},
            {"type": "percent", "value": 0.04},
        ],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))
    execution = Execution(manager)

    run_ticks(execution, [make_tick(0, 100.0)])

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    execution.submit([entry])
    execution.update(state=None, tick=make_tick(1, 100.0))

    fills = execution.update(state=None, tick=make_tick(2, 150.0))

    # Both targets are genuinely triggerable, so both fill, each at its own
    # limit, and the stop is cancelled without being triggered.
    assert [f.role for f in fills] == [OrderRole.TARGET, OrderRole.TARGET]
    assert [f.price for f in fills] == [102.0, 104.0]
    assert [f.id for f in fills] == [2, 3]
    assert manager.portfolio.position is None
    assert manager.portfolio.realized_pnl == 3.0
    assert manager.working_orders() == []


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