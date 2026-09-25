from src.core.engine import TradingEngine
from src.ingestion.tick import Tick
from src.orders import OrderManager, OrderRole, Side
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


def build_engine_state(risk=None):
    """
    Synthetic engine_state for set_state():

        - 15m / 5m / 1m timeframes with EMA 20, EMA 50 and ADX 14
        - EMA/ADX values are seeded so that:
            15m: EMA20 > EMA50, ADX 26 (macro uptrend, already valid)
            5m : EMA20 < EMA50 but rising (a fresh bull cross appears as
                 the price climbs), ADX 28 with a wide +DI / -DI gap
            1m : EMA20 > EMA50 (momentum already aligned)

    A steady rise (100 + 0.4/min) then triggers the fresh 5m EMA cross,
    which makes the strategy emit a BUY once 15m, ADX and DI filters align.
    """
    adx5_hist = [adx_state(1 - i, BASE_TS - i * 300_000, 25.0 + i * 0.1, 30.0, 6.0) for i in range(28)]
    adx5_live = dict(adx5_hist[-1])
    adx5_live["adx"] = 28.0

    adx15_hist = [adx_state(1 - i, BASE_TS - i * 900_000, 26.0, 32.0, 6.0) for i in range(28)]
    adx15_live = dict(adx15_hist[-1])

    return {
        "tick_index": 0,
        "time": 0,
        "timeframes": {
            "1m": tf_state("1m", 60_000, [{"time": 1, "value": 102.0}], [{"time": 1, "value": 100.0}], [], None),
            "5m": tf_state("5m", 300_000, [{"time": 1, "value": 90.0}], [{"time": 1, "value": 100.0}], adx5_hist, adx5_live),
            "15m": tf_state("15m", 900_000, [{"time": 1, "value": 110.0}], [{"time": 1, "value": 100.0}], adx15_hist, adx15_live),
        },
        "strategy": {"kind": "Strategy1", "params": {}},
        "risk": risk or None,
        "portfolio": None,
        "orders": None,
    }


RISK = {
    "stop": {"type": "percent", "value": 0.01},
    "targets": [{"type": "percent", "value": 0.02}],
}


def price_at(minute: int) -> float:
    return 100.0 + 0.4 * minute


def run_until_buy(engine: TradingEngine):
    """
    Feeds one tick per minute. No signal must appear until the fresh
    5m cross; when it lands it must be a BUY.
    """
    signal = None

    for i in range(0, 49):
        _, sig = engine.on_tick(make_tick(i, BASE_TS + i * 60_000, price_at(i)))
        assert sig is None, f"unexpected early signal at minute {i}: {sig}"

    buy_minute = None

    for i in range(49, 80):
        _, sig = engine.on_tick(make_tick(i, BASE_TS + i * 60_000, price_at(i)))
        if sig is not None:
            signal = sig
            buy_minute = i
            break

    assert signal is not None, "strategy never emitted a BUY"
    assert signal.action == "BUY"
    return buy_minute


def test_engine_full_order_cycle():
    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=RISK))

    assert engine.risk == RISK

    buy_minute = run_until_buy(engine)
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

    buy_minute = run_until_buy(engine)
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


def test_restored_order_manager_matches_serialized_book():
    manager = OrderManager()
    assert manager.to_dict() == OrderManager.from_dict(manager.to_dict()).to_dict()