from src.core.engine import TradingEngine
from src.ingestion.tick import Tick
from src.orders import OrderManager, OrderRole, Side


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


def build_engine_state(risk=None):
    """
    Synthetic engine_state for set_state():

        - 1m timeframe with a Candlestick series
        - EMA labeled "EMA 55" (period 2), seeded at 10.0
        - EMA labeled "EMA 200" (period 4), seeded at 100.0

    The seeding produces previous_55 <= previous_200, so a sharp
    price jump makes the strategy emit a BUY signal.
    """
    return {
        "tick_index": 0,
        "time": 0,
        "timeframes": {
            "1m": {
                "id": "1m",
                "timeframe_ms": 60_000,
                "live": None,
                "closed": None,
                "is_new": False,
                "is_closed": False,
                "series": {
                    "candle-series": {
                        "id": "candle-series",
                        "kind": "Candlestick",
                        "level": 0,
                        "primary": True,
                        "overlay": False,
                        "params": {},
                    },
                    "ema55": {
                        "id": "ema55",
                        "kind": "EMA",
                        "level": 1,
                        "primary": False,
                        "overlay": True,
                        "params": {"label": "EMA 55", "period": 2},
                        "live": None,
                        "history": [{"time": 1, "value": 10.0}],
                    },
                    "ema200": {
                        "id": "ema200",
                        "kind": "EMA",
                        "level": 1,
                        "primary": False,
                        "overlay": True,
                        "params": {"label": "EMA 200", "period": 4},
                        "live": None,
                        "history": [{"time": 1, "value": 100.0}],
                    },
                },
            }
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


def test_engine_full_order_cycle():
    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=RISK))

    assert engine.risk == RISK

    # --- warm up: no cross yet ---
    _, signal = engine.on_tick(make_tick(0, 0, 50.0))
    _, signal = engine.on_tick(make_tick(1, 1, 50.0))
    assert signal is None

    # --- price jump triggers an EMA cross up -> BUY signal ---
    _, signal = engine.on_tick(make_tick(2, 2, 300.0))
    assert signal is not None
    assert signal.action == "BUY"

    # --- entry order fills on the next tick -> bracket placed ---
    state, signal = engine.on_tick(make_tick(3, 3, 300.0))

    position = engine.portfolio.position
    assert position is not None
    assert position.side == Side.BUY
    assert position.avg_price == 300.0
    assert position.quantity == 1.0

    working = engine.order_manager.working_orders()
    assert len(working) == 2

    stop = [o for o in working if o.role == OrderRole.STOP][0]
    target = [o for o in working if o.role == OrderRole.TARGET][0]
    assert stop.price == 297.0
    assert target.price == 306.0

    # --- the published state reflects this tick's work ---
    d = state.to_dict()
    assert d["risk"] == RISK
    assert d["portfolio"]["position"]["side"] == "BUY"
    assert d["portfolio"]["position"]["quantity"] == 1.0
    assert d["orders"] == engine.order_manager.to_dict()


def test_engine_state_restore_preserves_open_position():
    engine = TradingEngine()
    engine.set_state("boot", "config", build_engine_state(risk=RISK))

    engine.on_tick(make_tick(0, 0, 50.0))
    engine.on_tick(make_tick(1, 1, 50.0))
    _, signal = engine.on_tick(make_tick(2, 2, 300.0))
    assert signal.action == "BUY"

    engine.on_tick(make_tick(3, 3, 300.0))
    assert engine.portfolio.position is not None

    # Serialize the engine state exactly like the publisher would.
    published = engine.state.to_dict()

    # Restore into a fresh engine (simulating get-state after restart).
    restored = TradingEngine()
    restored.set_state(
        "boot",
        "config",
        {
            "tick_index": 3,
            "time": 3,
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
    assert position.avg_price == 300.0

    working = restored.order_manager.working_orders()
    assert len(working) == 2
    assert {o.role for o in working} == {OrderRole.STOP, OrderRole.TARGET}
    assert restored.risk == RISK

    # The restored bracket must still be triggerable.
    fills = restored.execution.update(None, make_tick(4, 4, 297.0))
    assert len(fills) == 1
    assert fills[0].role == OrderRole.STOP
    assert restored.portfolio.position is None


def test_restored_order_manager_matches_serialized_book():
    manager = OrderManager()
    assert manager.to_dict() == OrderManager.from_dict(manager.to_dict()).to_dict()