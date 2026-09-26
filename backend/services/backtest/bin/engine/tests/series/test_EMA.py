from src.series import EMA
from src.series.registry import SERIES_REGISTRY
import json
import pytest


class Candle:
    def __init__(self, time, close):
        self.time = time
        self.start_ts = time
        self.end_ts = time + 1
        self.high = close
        self.low = close
        self.close = close


class FakeTimeframe:
    def __init__(self):
        self.timeframe_ms = 900_000
        self.closed = None
        self.live = None
        self.is_closed = False

    def close_bar(self, ema, closed, live_price):
        ema._timeframe = self

        self.closed = closed
        self.live = Candle(closed.time + 1, live_price)
        self.is_closed = True

        ema.update()

        self.is_closed = False


def make(period=3):
    ema = EMA(
        id="ema",
        kind="EMA",
        level=1,
        primary=False,
        overlay=True,
        params={"label": "EMA 3", "period": period},
    )
    return ema, FakeTimeframe()


def test_first_close_seeds_with_the_raw_close():
    ema, tf = make(period=3)

    tf.close_bar(ema, Candle(1, 100.0), 100.0)

    assert ema._closed.value == pytest.approx(100.0)
    assert list(ema.history) == []


def test_recursion_uses_the_previous_closed_value():
    ema, tf = make(period=3)
    # alpha = 2 / (3 + 1) = 0.5

    tf.close_bar(ema, Candle(1, 100.0), 100.0)
    # 0.5 * 110 + 0.5 * 100 = 105
    tf.close_bar(ema, Candle(2, 110.0), 110.0)
    assert ema._closed.value == pytest.approx(105.0)
    # 0.5 * 120 + 0.5 * 105 = 112.5
    tf.close_bar(ema, Candle(3, 120.0), 120.0)
    assert ema._closed.value == pytest.approx(112.5)


def test_restored_series_keeps_smoothing_from_the_same_state():
    # `history` lags `_closed` by one bar, so a restored EMA that rebuilt
    # itself from `history[-1]` would resume one bar stale and silently
    # change every downstream signal.
    ema, tf = make(period=3)

    for i, close in enumerate([100.0, 110.0, 120.0, 115.0], start=1):
        tf.close_bar(ema, Candle(i, close), close)

    assert len(ema.history) == 3

    restored = EMA(
        id="ema",
        kind="EMA",
        level=1,
        primary=False,
        overlay=True,
        params={"label": "EMA 3", "period": 3},
    )
    restored_tf = FakeTimeframe()
    restored._timeframe = restored_tf
    restored.set_state(ema.to_dict())

    assert restored._closed == ema._closed

    for i, close in enumerate([130.0, 125.0, 140.0], start=5):
        tf.close_bar(ema, Candle(i, close), close)
        restored_tf.close_bar(restored, Candle(i, close), close)

        assert restored._closed == ema._closed
        assert restored.live == ema.live


def test_legacy_state_without_closed_field_still_restores():
    ema, tf = make(period=3)

    for i, close in enumerate([100.0, 110.0, 120.0], start=1):
        tf.close_bar(ema, Candle(i, close), close)

    state = ema.to_dict()
    del state["closed"]

    restored = EMA(
        id="ema",
        kind="EMA",
        level=1,
        primary=False,
        overlay=True,
        params={"label": "EMA 3", "period": 3},
    )
    restored.set_state(state)

    # falls back to the old behaviour: the last value in history
    assert restored._closed == ema.history[-1]


def test_state_round_trip_through_json():
    ema, tf = make(period=3)

    for i, close in enumerate([100.0, 110.0, 120.0, 115.0], start=1):
        tf.close_bar(ema, Candle(i, close), close)

    restored = EMA(
        id="ema",
        kind="EMA",
        level=1,
        primary=False,
        overlay=True,
        params={"label": "EMA 3", "period": 3},
    )
    restored.set_state(json.loads(json.dumps(ema.to_dict())))

    assert restored.live == ema.live
    assert list(restored.history) == list(ema.history)
    assert restored._closed == ema._closed


def test_registry_instantiates_and_restores():
    assert "EMA" in SERIES_REGISTRY

    ema = SERIES_REGISTRY["EMA"](
        id="ema",
        kind="EMA",
        level=1,
        primary=False,
        overlay=True,
        params={"label": "EMA 3", "period": 3},
    )

    restored = SERIES_REGISTRY["EMA"](
        id="ema",
        kind="EMA",
        level=1,
        primary=False,
        overlay=True,
        params={"label": "EMA 3", "period": 3},
    )
    restored.set_state(ema.to_dict())

    assert restored._closed == ema._closed
