from src.core.engine import TradingEngine
from src.ingestion.tick import Tick
from src.orders import OrderManager, OrderRole, Side, Signal
from src.strategy.base import Strategy
from src.strategy.my_strategy import Strategy1
import pytest

BASE_TS = 1700 * 60_000


def make_tick(index, time, price):
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


def ema_spec(hist, label, period):
    return {
        "id": "ema_" + label.replace(" ", ""),
        "kind": "EMA",
        "level": 1,
        "primary": False,
        "overlay": True,
        "params": {"label": label, "period": period},
        "live": None,
        "history": hist,
    }


def adx_state(t, start_ts, adx, plus, minus):
    return {
        "time": t,
        "start_ts": start_ts,
        "end_ts": start_ts + 300_000,
        "adx": adx,
        "plus_di": plus,
        "minus_di": minus,
        "adx_color": "lime",
        "is_reversal": False,
        "reversal_level": None,
        "high": 100.0,
        "low": 99.0,
        "close": 100.0,
        "tr_rma": 2.0,
        "plus_dm_rma": 0.6,
        "minus_dm_rma": 0.2,
    }


def tf_state(tf_id, ms, e20, e50, adx_hist, adx_live):
    return {
        "id": tf_id,
        "timeframe_ms": ms,
        "live": None,
        "closed": None,
        "is_new": False,
        "is_closed": False,
        "series": {
            "candle": {
                "id": "candle",
                "kind": "Candlestick",
                "level": 0,
                "primary": True,
                "overlay": False,
                "params": {},
                "live": None,
                "history": [],
            },
            "ema20": ema_spec(e20, "EMA 20", 20),
            "ema50": ema_spec(e50, "EMA 50", 50),
            "adx": {
                "id": "adx",
                "kind": "ADX",
                "level": 1,
                "primary": False,
                "overlay": True,
                "params": {"label": "ADX 14", "dilen": 14, "adxlen": 14, "key_level": 23},
                "live": adx_live,
                "history": adx_hist,
            },
        },
    }


def build_engine_state(risk=None, direction="long", params=None):
    """
    Synthetic engine_state for set_state():

        - the five timeframes Strategy1 declares (1h, 30m, 15m, 5m, 1m)
          with EMA 20, EMA 50 and ADX 14
        - the EMAs are seeded already aligned with `direction`, which is
          what a config looks like at the moment a strategy starts: the
          strategy sees a full agreement and acts straight away
        - the ADX of every timeframe is aligned too, so 1h counts as
          trending and the strategy gets past its first gate

    All five are needed: `evaluate()` returns None straight away if 1h or
    30m is missing, which used to make every test here fail with "strategy
    never emitted a BUY signal".

    `direction` also decides whether the steady move that follows is up
    or down, so the position opened by the strategy stays in its favour
    unless a test asks for a reversal.
    """
    if direction == "long":
        above, below = 110.0, 90.0
    elif direction == "short":
        above, below = 90.0, 110.0
    else:
        raise ValueError(direction)

    e50 = 100.0

    adx5_hist = [adx_state(1 - i, BASE_TS - i * 300_000, 25.0 + i * 0.1, above, below) for i in range(28)]
    adx5_live = dict(adx5_hist[-1])
    adx5_live["adx"] = 28.0

    adx15_hist = [adx_state(1 - i, BASE_TS - i * 900_000, 26.0, above, below) for i in range(28)]
    adx15_live = dict(adx15_hist[-1])

    adx30_hist = [adx_state(1 - i, BASE_TS - i * 1_800_000, 25.0, above, below) for i in range(28)]
    adx30_live = dict(adx30_hist[-1])

    adx1h_hist = [adx_state(1 - i, BASE_TS - i * 3_600_000, 27.0, above, below) for i in range(28)]
    adx1h_live = dict(adx1h_hist[-1])

    return {
        "tick_index": 0,
        "time": 0,
        "timeframes": {
            "1m": tf_state("1m", 60_000, [{"time": 1, "value": above}], [{"time": 1, "value": e50}], [], None),
            "5m": tf_state("5m", 300_000, [{"time": 1, "value": above}], [{"time": 1, "value": e50}], adx5_hist, adx5_live),
            "15m": tf_state("15m", 900_000, [{"time": 1, "value": above}], [{"time": 1, "value": e50}], adx15_hist, adx15_live),
            "30m": tf_state("30m", 1_800_000, [{"time": 1, "value": above}], [{"time": 1, "value": e50}], adx30_hist, adx30_live),
            "1h": tf_state("1h", 3_600_000, [{"time": 1, "value": above}], [{"time": 1, "value": e50}], adx1h_hist, adx1h_live),
        },
        "strategy": {"kind": "Strategy1", "params": params or {}},
        "risk": risk or None,
        "portfolio": None,
        "orders": None,
    }


RISK = {
    "stop": {"type": "percent", "value": 0.01},
    "targets": [{"type": "percent", "value": 0.02}],
}


def price_at(minute: int, direction: str = "long") -> float:
    if direction == "short":
        return 100.0 - 0.4 * minute
    return 100.0 + 0.4 * minute


def run_until_signal(
    engine: TradingEngine,
    direction: str,
    expected: str,
    ticks: int = 120,
) -> int:
    """
    Feeds one tick per minute until the strategy emits `expected`.

    Returns the minute the signal appeared on.
    """
    for i in range(0, ticks):
        _, sig = engine.on_tick(make_tick(i, BASE_TS + i * 60_000, price_at(i, direction)))

        if sig is not None and sig.action == expected:
            return i

    raise AssertionError(f"strategy never emitted a {expected} signal")


def test_engine_full_order_cycle():
    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=RISK))

    assert engine.risk == RISK

    buy_minute = run_until_signal(engine, "long", "BUY")
    assert buy_minute is not None

    # --- entry order fills on the next tick -> bracket placed ---
    state, sig = engine.on_tick(make_tick(buy_minute + 1, BASE_TS + (buy_minute + 1) * 60_000, price_at(buy_minute + 1)))

    position = engine.portfolio.position
    assert position is not None
    assert position.side == Side.BUY
    assert position.quantity == 1.0
    assert position.avg_price == price_at(buy_minute + 1)

    working = engine.order_manager.working_orders()
    assert len(working) == 2

    stop = [o for o in working if o.role == OrderRole.STOP][0]
    target = [o for o in working if o.role == OrderRole.TARGET][0]
    assert stop.price == pytest.approx(position.avg_price * 0.99, rel=1e-9)
    assert target.price == pytest.approx(position.avg_price * 1.02, rel=1e-9)

    # --- the published state reflects this tick's work ---
    d = state.to_dict()
    assert d["risk"] == RISK
    assert d["portfolio"]["position"]["side"] == "BUY"
    assert d["portfolio"]["position"]["quantity"] == 1.0
    assert d["orders"] == engine.order_manager.to_dict()


def test_engine_state_restore_preserves_open_position():
    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=RISK))

    buy_minute = run_until_signal(engine, "long", "BUY")
    engine.on_tick(make_tick(buy_minute + 1, BASE_TS + (buy_minute + 1) * 60_000, price_at(buy_minute + 1)))

    assert engine.portfolio.position is not None

    # Serialize the engine state exactly like the publisher would.
    published = engine.state.to_dict()

    # Restore into a fresh engine (simulating get-state after restart).
    restored = TradingEngine()
    restored.set_state(
        "boot",
        "config",
        {
            "tick_index": buy_minute + 1,
            "time": BASE_TS + (buy_minute + 1) * 60_000,
            "timeframes": published["timeframes"],
            "strategy": {"kind": "Strategy1", "params": {}},
            "risk": published["risk"],
            "portfolio": published["portfolio"],
            "orders": published["orders"],
        },
    )

    position = restored.portfolio.position
    assert position is not None
    assert position.quantity == 1.0
    assert position.avg_price == price_at(buy_minute + 1)

    working = restored.order_manager.working_orders()
    assert len(working) == 2
    assert {o.role for o in working} == {OrderRole.STOP, OrderRole.TARGET}
    assert restored.risk == RISK

    # The restored bracket must still be triggerable.
    stop = [o for o in working if o.role == OrderRole.STOP][0]
    fills = restored.execution.update(None, make_tick(9000, BASE_TS + 9000 * 60_000, stop.price))
    assert len(fills) == 1
    assert fills[0].role == OrderRole.STOP
    assert restored.portfolio.position is None


def test_engine_short_entry_bracket_geometry():
    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=RISK, direction="short"))

    sell_minute = run_until_signal(engine, "short", "SELL")
    assert sell_minute is not None

    engine.on_tick(make_tick(sell_minute + 1, BASE_TS + (sell_minute + 1) * 60_000, price_at(sell_minute + 1, "short")))

    position = engine.portfolio.position
    assert position is not None
    assert position.side == Side.SELL
    assert position.quantity == 1.0

    working = engine.order_manager.working_orders()
    assert len(working) == 2
    assert {o.role for o in working} == {OrderRole.STOP, OrderRole.TARGET}

    # Short bracket geometry: stop above, target below the entry.
    stop = [o for o in working if o.role == OrderRole.STOP][0]
    target = [o for o in working if o.role == OrderRole.TARGET][0]
    assert stop.price == pytest.approx(position.avg_price * 1.01, rel=1e-9)
    assert target.price == pytest.approx(position.avg_price * 0.98, rel=1e-9)


def flip_structure(engine, tf_id, fast, slow):
    """
    Turns the EMA structure of `tf_id` to (fast, slow).

    The strategy reads ``series._closed``, and the fixture seeds those
    once and never recomputes them, so no amount of feeding ticks will
    make a structure turn on its own. Driving the turn here is what a
    new closed candle would do in a real run.
    """
    timeframe = engine.state.timeframes.get(tf_id)

    for label, value in (("EMA 20", fast), ("EMA 50", slow)):
        series = timeframe.get_series("EMA", label)
        series._closed = type("Closed", (), {"value": value})()


def test_engine_short_internal_exit_path():
    # The short is opened while the 1h structure is bearish. It only exits
    # once that structure turns bullish again, so the test flips the 1h
    # EMAs rather than trying to move the price: the fixture's series are
    # static, and the strategy only ever reads their confirmed values.
    #
    # The stop is deliberately wide: this test is about the exit the
    # *strategy* asks for, and with the default 1% stop the reversal would
    # take the position out through the bracket first.
    risk = {
        "stop": {"type": "percent", "value": 0.25},
        "targets": [{"type": "percent", "value": 0.05}],
    }

    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=risk, direction="short"))

    sell_minute = run_until_signal(engine, "short", "SELL")

    # --- the entry fills on the next tick ---
    _, sig = engine.on_tick(make_tick(sell_minute + 1, BASE_TS + (sell_minute + 1) * 60_000, price_at(sell_minute + 1, "short")))

    position = engine.portfolio.position
    assert position is not None
    assert position.side == Side.SELL
    assert position.quantity == 1.0

    # Still bearish, so the strategy holds and the bracket is untouched.
    working = engine.order_manager.working_orders()
    assert len(working) == 2
    assert {o.role for o in working} == {OrderRole.STOP, OrderRole.TARGET}
    assert sig is None

    # --- the structure turns: the strategy exits ---
    # 1h goes bullish. This is a 1h reversal, so the close is a full one:
    # leaving 30m untouched is deliberate, otherwise the 30m cross would
    # ask for a scale-out and the position would never close.
    flip_structure(engine, "1h", 110.0, 100.0)

    price = price_at(sell_minute + 1, "short")
    exit_minute = None

    for step in range(1, 120):
        minute = sell_minute + 1 + step
        price += 0.5
        _, sig = engine.on_tick(make_tick(minute, BASE_TS + minute * 60_000, price))

        if sig is not None and sig.action == "EXIT":
            exit_minute = minute
            break

    assert exit_minute is not None, "strategy never exited the short"
    assert sig.quantity is None, "a 1h reversal must close in full"

    # The bracket is cancelled and replaced by a single market exit.
    working = engine.order_manager.working_orders()
    assert len(working) == 1
    assert working[0].role == OrderRole.EXIT

    # --- the market exit order fills on the next tick ---
    price += 0.5
    engine.on_tick(make_tick(exit_minute + 1, BASE_TS + (exit_minute + 1) * 60_000, price))

    assert engine.portfolio.position is None
    assert len(engine.portfolio.trades) == 1
    assert engine.portfolio.trades[0].side == Side.SELL

    # The exit order and the whole bracket are gone. The only thing that may
    # be left working is a brand new entry: the structure is bullish again,
    # the strategy carries no cooldown, and it is free to turn straight
    # around on the very tick the short closed.
    assert all(
        o.role == OrderRole.ENTRY
        for o in engine.order_manager.working_orders()
    )


def test_restored_order_manager_matches_serialized_book():
    manager = OrderManager()
    assert manager.to_dict() == OrderManager.from_dict(manager.to_dict()).to_dict()


def test_strategy_sees_the_book_as_it_is_after_this_tick():
    # Regression: the state handed to evaluate() used to be built before the
    # trigger pass, so on the tick an entry filled the strategy still saw the
    # entry as WORKING and the freshly created bracket as missing. A
    # strategy guarding on "is my order still pending?" would have read it
    # backwards for one tick.
    observed = []

    engine = TradingEngine()
    state = build_engine_state(risk=RISK, direction="short")
    engine.set_state("boot", "config", state)

    class Spy(Strategy1):
        def evaluate(self, state):
            observed.append(
                {
                    order["id"]: order["status"]
                    for order in state.orders["orders"]
                    if order["status"] == "WORKING"
                }
            )
            return super().evaluate(state)

    engine.strategy = Spy("Strategy1", state["strategy"]["params"])

    sell_minute = run_until_signal(engine, "short", "SELL")

    for minute in range(sell_minute + 1, sell_minute + 6):
        published, _ = engine.on_tick(
            make_tick(minute, BASE_TS + minute * 60_000, price_at(minute, "short"))
        )

        assert observed[-1] == {
            order["id"]: order["status"]
            for order in published.orders["orders"]
            if order["status"] == "WORKING"
        }


def test_engine_book_stays_bounded_over_a_long_run():
    # Regression: settled orders were never dropped, and the whole book was
    # reserialized twice per tick, so both the payload and the per-tick cost
    # grew with the length of the replay instead of with live exposure.
    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=RISK))

    settled_cap = OrderManager.prune.__defaults__[0]
    samples = []
    plateau = None

    # Run until the book stops growing instead of guessing a tick count:
    # prune() retains the last 200 settled orders, so the ceiling is only
    # reached once enough trades have settled to fill it. How many ticks
    # that takes depends on how often the strategy trades.
    for index in range(20_000):
        # Zigzag: enough crossings to trade repeatedly in both directions.
        price = 100.0 + 0.4 * index if (index // 25) % 2 == 0 else 100.0 - 0.4 * (index % 50)
        state, _ = engine.on_tick(
            make_tick(index, BASE_TS + index * 60_000, price)
        )

        if (index + 1) % 400 != 0:
            continue

        size = len(state.orders["orders"])
        samples.append((index + 1, size))

        if len(samples) > 2 and size == samples[-3][1] == samples[-2][1]:
            plateau = index + 1
            break

    assert plateau is not None, f"the book never settled, sizes: {samples}"

    # Once capped, the book stops tracking the length of the replay.
    assert len(engine.order_manager.working_orders()) <= 3
    assert len(engine.order_manager._orders) <= settled_cap + 2

    # The trade history itself is the backtest output and is untouched.
    assert len(engine.portfolio.trades) > 10


def test_engine_partial_exit_keeps_the_residual_open_and_protected():
    # A strategy that scales out instead of closing: it opens one position,
    # exits half of it, and lets the rest run under its original bracket.
    #
    # It drives itself off the portfolio rather than off the series, so the
    # test is about the order lifecycle and not about which timeframes a
    # particular strategy happens to read.
    engine = TradingEngine()
    state = build_engine_state(risk=RISK)
    engine.set_state("boot", "config", state)

    class ScaleOut(Strategy):
        def __init__(self):
            super().__init__("ScaleOut", {})

        def evaluate(self, state):
            position = state.portfolio.position

            if position is None:
                return Signal(action="BUY", quantity=1)

            # Scale out the first time only, then hold the residual.
            if position.quantity == 1.0:
                return Signal(action="EXIT", quantity=position.quantity * 0.5)

            return None

    engine.strategy = ScaleOut()

    # --- the entry is submitted, then fills on the next tick ---
    _, signal = engine.on_tick(make_tick(0, BASE_TS, 100.0))

    assert signal is not None
    assert signal.action == "BUY"

    published, signal = engine.on_tick(make_tick(1, BASE_TS + 60_000, 100.0))

    position = engine.portfolio.position
    assert position is not None
    assert position.quantity == 1.0
    assert position.avg_price == 100.0

    # The entry filled, its bracket went in, and the strategy already asked
    # to scale out on the very tick it filled: the exit is working
    # alongside the stop and the target.
    assert signal is not None
    assert signal.action == "EXIT"
    assert signal.quantity == 0.5

    exit_orders = [o for o in engine.order_manager.working_orders() if o.role == OrderRole.EXIT]
    assert len(exit_orders) == 1
    assert exit_orders[0].quantity == 0.5

    # The bracket survived the partial exit: the residual is still covered.
    assert {o.role for o in engine.order_manager.working_orders()} == {
        OrderRole.STOP,
        OrderRole.TARGET,
        OrderRole.EXIT,
    }

    # --- the market exit fills on the next tick ---
    #
    # Inside the bracket on purpose: a tick past the target would let the
    # protective order win the race, which is the documented priority and
    # not what this test is about.
    exit_price = 101.0
    published, _ = engine.on_tick(make_tick(2, BASE_TS + 120_000, exit_price))

    position = engine.portfolio.position
    assert position is not None
    assert position.quantity == 0.5
    assert engine.portfolio.realized_pnl == pytest.approx(0.5, rel=1e-9)

    # Exactly one trade was booked, for the slice that actually closed.
    assert len(engine.portfolio.trades) == 1
    assert engine.portfolio.trades[0].quantity == 0.5

    # The published state reports the residual, not the original position.
    assert published.to_dict()["portfolio"]["position"]["quantity"] == 0.5

    # The stop shrank to the quantity that is left.
    stop = [o for o in engine.order_manager.working_orders() if o.role == OrderRole.STOP]
    assert len(stop) == 1
    assert stop[0].quantity == 0.5

    # The target is still there to close the residual...
    target = [o for o in engine.order_manager.working_orders() if o.role == OrderRole.TARGET]
    assert len(target) == 1

    # ...and when it does, the bracket retires with it and the position is
    # gone. The only thing left working is the new entry the strategy
    # opened on that same flat tick.
    target_price = target[0].price
    engine.on_tick(make_tick(3, BASE_TS + 180_000, target_price))

    assert engine.portfolio.position is None
    assert engine.portfolio.realized_pnl == pytest.approx(0.5 + 1.0, rel=1e-9)

    working = engine.order_manager.working_orders()
    assert all(o.role == OrderRole.ENTRY for o in working)
    assert all(o.group_id == 2 for o in working)
