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


def fill_entry(manager, order, price, tick):
    fill = manager.make_fill(order, price, order.quantity, tick)
    return manager.on_fill(fill)


def fill_order(manager, order, price, tick):
    fill = manager.make_fill(order, price, order.quantity, tick)
    manager.on_fill(fill)
    return manager


def find_role(orders, role):
    return [o for o in orders if o.role == role]


# An explicit stop + single target, so a filled entry always produces exactly
# two working orders. Relying on the default RiskManager would make these
# tests depend on whether it invents a stop for an empty config.
BRACKET = {
    "stop": {"type": "percent", "value": 0.01},
    "targets": [{"type": "percent", "value": 0.02}],
}


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


def test_default_risk_places_default_stop_when_none_configured():
    manager = OrderManager(portfolio=Portfolio())

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    working = manager.working_orders()

    stops = find_role(working, OrderRole.STOP)

    assert len(stops) == 1
    assert stops[0].price == 99.0
    assert stops[0].quantity == 1.0


def test_explicit_stop_overrides_default():
    config = {
        "stop": {"type": "percent", "value": 0.02},
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    stops = find_role(manager.working_orders(), OrderRole.STOP)

    assert len(stops) == 1
    assert stops[0].price == 98.0


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


def test_partial_exit_closes_only_the_requested_quantity():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    orders = manager.handle(Signal(action="EXIT", quantity=0.5))

    assert len(orders) == 1
    assert orders[0].role == OrderRole.EXIT
    assert orders[0].quantity == 0.5

    # The bracket is NOT cancelled: 0.5 of the position is still open and
    # has to stay protected. A full exit is the only thing that invalidates
    # it.
    assert manager.portfolio.position.quantity == 1.0
    assert {o.role for o in manager.working_orders()} == {
        OrderRole.STOP,
        OrderRole.TARGET,
        OrderRole.EXIT,
    }

    fill_order(manager, orders[0], 110.0, make_tick(1, 110.0))

    assert manager.portfolio.position.quantity == 0.5
    assert manager.portfolio.realized_pnl == 5.0

    # The residual keeps its protection: the stop shrank to the quantity
    # that is left and the target is still there, capped by the group when
    # it fills.
    working = manager.working_orders()
    assert {o.role for o in working} == {OrderRole.STOP, OrderRole.TARGET}

    stop = find_role(working, OrderRole.STOP)[0]
    assert stop.quantity == 0.5
    assert stop.status == OrderStatus.WORKING


def test_partial_exit_leaves_the_target_ladder_intact():
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

    exit_order = manager.handle(Signal(action="EXIT", quantity=0.5))[0]
    fill_order(manager, exit_order, 110.0, make_tick(1, 110.0))

    # Both rungs survive the partial exit. They keep their own size and are
    # capped by what is left of the group, so the first to trigger closes
    # the residual and the second is retired with it.
    targets = find_role(manager.working_orders(), OrderRole.TARGET)
    assert len(targets) == 2
    assert all(t.status == OrderStatus.WORKING for t in targets)

    fill_order(manager, targets[0], 102.0, make_tick(2, 102.0))

    assert manager.portfolio.position is None
    assert manager.working_orders() == []


def test_exit_quantity_larger_than_the_position_is_capped():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    # The venue would fill what exists. Asking for more than the position
    # closes it instead of killing the run.
    orders = manager.handle(Signal(action="EXIT", quantity=5))

    assert orders[0].quantity == 1.0
    assert manager.working_orders() == orders


def test_exit_signal_requires_a_positive_quantity():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    before = manager.to_dict()

    for quantity in (0, -1, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            manager.handle(Signal(action="EXIT", quantity=quantity))

    # Nothing reached the book and the bracket is untouched: a rejected
    # exit is not an exit.
    assert manager.to_dict() == before


def test_fractional_slices_close_the_position_instead_of_leaving_dust():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    for index, quantity in enumerate((0.1, 0.1, 0.1, 0.7), start=1):
        exit_order = manager.handle(Signal(action="EXIT", quantity=quantity))[0]
        fill_order(manager, exit_order, 110.0, make_tick(index, 110.0))

    # Binary floating point leaves 1.1e-16 after those four slices. A
    # position that survives that sliver would keep the strategy reading a
    # live position and block every future entry.
    assert manager.portfolio.position is None
    assert manager.working_orders() == []
    assert manager.portfolio.realized_pnl == pytest.approx(10.0)


def test_the_single_position_gate_reopens_after_a_partial_exit_closes_out():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    exit_order = manager.handle(Signal(action="EXIT", quantity=0.5))[0]
    fill_order(manager, exit_order, 110.0, make_tick(1, 110.0))

    # Still open, so still gated.
    assert manager.handle(Signal(action="BUY", quantity=1)) == []

    for index, quantity in enumerate((0.25, 0.25), start=2):
        exit_order = manager.handle(Signal(action="EXIT", quantity=quantity))[0]
        fill_order(manager, exit_order, 110.0, make_tick(index, 110.0))

    assert manager.portfolio.position is None

    # Flat, so the gate lets the next trade through.
    assert len(manager.handle(Signal(action="BUY", quantity=1))) == 1


def test_a_partial_exit_that_empties_the_group_cancels_the_bracket():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    # A quantity within the tolerance of the whole position is a full
    # close: the bracket goes with it rather than lingering on dust.
    exit_order = manager.handle(Signal(action="EXIT", quantity=0.99999999999))[0]

    assert manager.working_orders() == [exit_order]

    fill_order(manager, exit_order, 110.0, make_tick(1, 110.0))

    assert manager.portfolio.position is None
    assert manager.working_orders() == []


def test_the_bracket_stop_never_moves_after_the_entry_fill():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    stop = find_role(manager.working_orders(), OrderRole.STOP)[0]

    # Priced once, against the entry fill.
    assert stop.price == 99.0

    # Nothing in the book carries a ratchet and there is no per-tick hook
    # left that could move the stop, so it stays at its entry level.
    assert not hasattr(stop, "trailing")
    assert not hasattr(manager, "update_trailing_stops")
    assert not hasattr(RiskManager(BRACKET), "trailing_distance")
    assert stop.price == 99.0

    # It still triggers on that level.
    fill_order(manager, stop, 99.0, make_tick(1, 99.0))

    assert stop.status == OrderStatus.FILLED
    assert manager.portfolio.position is None


def test_stop_takes_matching_priority_over_the_targets():
    manager = OrderManager(
        portfolio=Portfolio(),
        risk_manager=RiskManager(
            {
                "stop": {"type": "percent", "value": 0.01},
                "targets": [{"type": "percent", "value": 0.02}],
            }
        ),
    )

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    working = manager.working_orders()

    assert [o.role for o in working] == [OrderRole.STOP, OrderRole.TARGET]


def test_entry_signal_requires_a_positive_quantity():
    manager = OrderManager(portfolio=Portfolio())

    for quantity in (0, -1, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            manager.handle(Signal(action="BUY", quantity=quantity))

    # Nothing reached the book: a zero or negative entry can never fill and
    # would sit working forever while the strategy still reads a flat account.
    assert manager.working_orders() == []
    assert manager.groups() == []


def test_second_entry_rejected_while_the_first_is_still_pending():
    manager = OrderManager(portfolio=Portfolio())

    manager.handle(Signal(action="BUY", quantity=1))

    # The entry fills on the next tick, so until then there is no position to
    # block a second one.
    assert manager.handle(Signal(action="BUY", quantity=1)) == []
    assert manager.handle(Signal(action="SELL", quantity=1)) == []

    fill_entry(manager, manager.working_orders()[0], 100.0, make_tick(0, 100.0))

    assert manager.handle(Signal(action="BUY", quantity=1)) == []


def test_exit_order_attaches_to_the_group_that_backs_the_position():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [{"type": "percent", "value": 0.02}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    # First trade, closed by its own bracket.
    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))
    target = find_role(manager.working_orders(), OrderRole.TARGET)[0]
    fill_order(manager, target, 102.0, make_tick(1, 102.0))

    # Second trade: the exit must land on group 2, not on the closed group 1.
    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 200.0, make_tick(2, 200.0))

    exits = [o for o in manager.handle(Signal(action="EXIT")) if o.role == OrderRole.EXIT]

    assert len(exits) == 1
    assert exits[0].group_id == 2


def test_fill_quantity_reports_nothing_left_for_a_flat_group():
    config = {
        "stop": {"type": "percent", "value": 0.01},
        "targets": [{"type": "percent", "value": 0.02}],
    }
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(config))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    target = find_role(manager.working_orders(), OrderRole.TARGET)[0]

    assert manager.fill_quantity(target) == 1.0

    manager.groups()[0].remaining_quantity = 0.0

    assert manager.fill_quantity(target) == 0.0


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


def churn(manager, cycles):
    """Runs ``cycles`` complete round trips: entry fills, stop takes it out."""
    for index in range(cycles):
        entry = manager.handle(Signal(action="BUY", quantity=1))[0]
        fill_entry(manager, entry, 100.0, make_tick(index * 2, 100.0))

        stop = find_role(manager.working_orders(), OrderRole.STOP)[0]
        fill_order(manager, stop, stop.price, make_tick(index * 2 + 1, stop.price))


def test_prune_never_drops_a_working_order():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    churn(manager, 5)

    # A live entry with no bracket yet: it is the engine's current truth.
    manager.handle(Signal(action="BUY", quantity=1))

    manager.prune(keep=0)

    working = manager.working_orders()

    assert len(working) == 1
    assert working[0].role == OrderRole.ENTRY
    assert all(o.status == OrderStatus.WORKING for o in working)


def test_prune_keeps_the_most_recent_settled_orders():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    churn(manager, 12)

    before = len(manager.to_dict()["orders"])

    removed, _ = manager.prune(keep=5)

    after = manager.to_dict()["orders"]

    assert removed == before - len(after)
    assert len(after) == 5

    # What survives is the tail, in creation order, so the client can still
    # render the most recent activity.
    assert [o["id"] for o in after] == sorted(o["id"] for o in after)
    assert all(o["status"] != OrderStatus.WORKING for o in after)


def test_prune_bounds_the_book_over_a_long_run():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    churn(manager, 200)

    manager.prune(keep=10)

    assert len(manager.to_dict()["orders"]) == 10
    assert len(manager.to_dict()["groups"]) == 10


def test_prune_keeps_a_group_that_still_backs_quantity():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager({"stop": None}))

    churn(manager, 8)

    # A group whose orders are all settled but that still carries quantity is
    # an exposed position with no protection. Its record is the only evidence
    # of that, so pruning must not hide it.
    live_group = manager.groups()[0]
    live_group.remaining_quantity = 5.0

    manager.prune(keep=0)

    assert live_group.id in [group.id for group in manager.groups()]


def test_prune_keeps_working_orders_of_a_flat_group():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    entry = manager.handle(Signal(action="BUY", quantity=1))[0]
    fill_entry(manager, entry, 100.0, make_tick(0, 100.0))

    group = manager.groups()[0]

    # Force the group flat while its bracket is still working: the orders
    # are the thing that decides, not the group's bookkeeping.
    group.remaining_quantity = 0.0

    manager.prune(keep=0)

    assert group.id in [candidate.id for candidate in manager.groups()]
    assert len(manager.working_orders()) == 2


def test_prune_with_a_negative_window_is_a_noop():
    manager = OrderManager(portfolio=Portfolio(), risk_manager=RiskManager(BRACKET))

    churn(manager, 3)

    before = manager.to_dict()

    assert manager.prune(keep=-1) == (0, 0)
    assert manager.to_dict() == before
