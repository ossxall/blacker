from src.series import ATR
from src.series.registry import SERIES_REGISTRY
import pytest


class Candle:
    def __init__(self, time, high, low, close):
        self.time = time
        self.start_ts = time
        self.end_ts = time + 1
        self.high = high
        self.low = low
        self.close = close


class FakeTimeframe:
    """
    Minimal stand-in for Timeframe. `close_bar` mirrors what a real tick
    does: optionally close a bar, then always expose a brand new live bar
    that starts at the next price.
    """

    def __init__(self, timeframe_ms=900_000):
        self.timeframe_ms = timeframe_ms
        self.closed = None
        self.live = None
        self.is_closed = False

    def close_bar(self, atr, closed, live_price):
        atr._timeframe = self

        self.closed = closed
        # the new bar exists, but has traded only one tick so far
        self.live = Candle(closed.time + 1, live_price, live_price, live_price)
        self.is_closed = True

        atr.update()

        self.is_closed = False

    def provisional(self, atr, live):
        atr._timeframe = self

        self.closed = None
        self.live = live
        self.is_closed = False

        atr.update()


def make(period=14):
    atr = ATR(
        id="atr",
        kind="ATR",
        level=1,
        primary=False,
        overlay=False,
        params={"label": "ATR 14", "period": period},
    )
    return atr, FakeTimeframe()


def test_period_must_be_positive():
    with pytest.raises(ValueError):
        ATR(
            id="atr",
            kind="ATR",
            level=1,
            primary=False,
            overlay=False,
            params={"period": 0},
        )


def test_period_accepts_parameter_descriptor():
    # the dashboard sends {"value": n, ...} rather than a bare number
    atr = ATR(
        id="atr",
        kind="ATR",
        level=1,
        primary=False,
        overlay=False,
        params={"period": {"value": 21}},
    )

    assert atr.period == 21


def test_first_true_range_is_the_bar_range():
    atr, tf = make()

    tf.close_bar(atr, Candle(1, high=110.0, low=100.0, close=105.0), 105.0)

    # no previous close yet, so tr falls back to high - low
    assert atr._closed.tr == pytest.approx(10.0)
    # still warming up: fewer than `period` samples
    assert atr._closed.atr is None
    assert atr._closed.atr_pct is None


def test_true_range_accounts_for_gaps():
    atr, tf = make()

    tf.close_bar(atr, Candle(1, high=110.0, low=100.0, close=105.0), 105.0)
    # gaps up: |high - previous_close| = 115 - 105 = 10 beats high - low = 5
    tf.close_bar(atr, Candle(2, high=115.0, low=110.0, close=114.0), 114.0)

    assert atr._closed.tr == pytest.approx(10.0)


def test_true_range_accounts_for_down_gaps():
    atr, tf = make()

    tf.close_bar(atr, Candle(1, high=110.0, low=100.0, close=105.0), 105.0)
    # gaps down: |low - previous_close| = 100 - 105 = 5 beats high - low = 3
    tf.close_bar(atr, Candle(2, high=103.0, low=100.0, close=101.0), 101.0)

    assert atr._closed.tr == pytest.approx(5.0)


def test_atr_seeds_with_mean_of_first_period_true_ranges():
    atr, tf = make(period=3)

    # constant 10 point ranges -> seeded ATR must be exactly 10
    for i in range(1, 4):
        tf.close_bar(atr, Candle(i, high=110.0, low=100.0, close=105.0), 105.0)

    assert atr._closed.atr == pytest.approx(10.0)
    assert atr._closed.atr_pct == pytest.approx(10.0 / 105.0)


def test_atr_uses_wilder_recursion_after_seeding():
    atr, tf = make(period=2)

    # tr = 10 (no previous close yet, falls back to the bar range)
    tf.close_bar(atr, Candle(1, high=105.0, low=95.0, close=100.0), 100.0)
    # tr = 10 -> seeds the average at 10
    tf.close_bar(atr, Candle(2, high=110.0, low=100.0, close=105.0), 105.0)
    assert atr._closed.atr == pytest.approx(10.0)

    # next tr = 20 -> (10 * 1 + 20) / 2 = 15
    tf.close_bar(atr, Candle(3, high=120.0, low=100.0, close=110.0), 110.0)
    assert atr._closed.atr == pytest.approx(15.0)

    # next tr = 10 -> (15 * 1 + 10) / 2 = 12.5
    tf.close_bar(atr, Candle(4, high=120.0, low=110.0, close=115.0), 115.0)
    assert atr._closed.atr == pytest.approx(12.5)


def test_live_tracks_the_in_progress_bar_without_polluting_state():
    atr, tf = make(period=2)

    tf.close_bar(atr, Candle(1, high=105.0, low=95.0, close=100.0), 100.0)
    tf.close_bar(atr, Candle(2, high=110.0, low=100.0, close=105.0), 105.0)
    assert atr._closed.atr == pytest.approx(10.0)

    # a wide provisional bar widens the live ATR... (tr = high - low = 100)
    tf.provisional(atr, Candle(3, high=200.0, low=100.0, close=190.0))
    assert atr.live.atr == pytest.approx((10.0 + 100.0) / 2.0)

    # ...but the closed value feeding the next bar is untouched
    assert atr._closed.atr == pytest.approx(10.0)
    assert atr._prev_close == pytest.approx(105.0)


def test_history_only_receives_closed_bars():
    atr, tf = make(period=2)

    for i in range(1, 6):
        tf.close_bar(atr, Candle(i, high=110.0, low=100.0, close=105.0), 105.0)

    # 5 closed bars -> the 4 previous values are in history
    assert len(atr.history) == 4
    assert all(v.time < atr._closed.time for v in atr.history)


def test_registry_instantiates_and_restores():
    assert "ATR" in SERIES_REGISTRY

    atr = SERIES_REGISTRY["ATR"](
        id="atr",
        kind="ATR",
        level=1,
        primary=False,
        overlay=False,
        params={"label": "ATR 14", "period": 14},
    )
    tf = FakeTimeframe()
    atr._timeframe = tf

    for i in range(1, 20):
        tf.close_bar(
            atr,
            Candle(i, high=110.0 + i, low=100.0, close=105.0 + i),
            105.0 + i,
        )

    state = atr.to_dict()

    restored = SERIES_REGISTRY["ATR"](
        id="atr",
        kind="ATR",
        level=1,
        primary=False,
        overlay=False,
        params={"label": "ATR 14", "period": 14},
    )
    restored.set_state(state)

    assert restored.live == atr.live
    assert list(restored.history) == list(atr.history)
    assert restored._closed == atr._closed
    assert restored._prev_close == atr._prev_close


def test_restored_series_keeps_warming_from_the_same_state():
    # a resumed engine must produce the same ATR as one that never
    # restarted, otherwise state restore silently changes the backtest
    atr, tf = make(period=3)

    for i in range(1, 6):
        tf.close_bar(atr, Candle(i, high=110.0, low=100.0, close=105.0), 105.0)

    restored = ATR(
        id="atr",
        kind="ATR",
        level=1,
        primary=False,
        overlay=False,
        params={"label": "ATR 14", "period": 3},
    )
    restored._timeframe = FakeTimeframe()
    restored.set_state(atr.to_dict())

    for i in range(6, 12):
        # a big range that must be smoothed by the restored Wilder state
        tf.close_bar(atr, Candle(i, high=140.0, low=100.0, close=140.0), 140.0)
        restored._timeframe.close_bar(
            restored, Candle(i, high=140.0, low=100.0, close=140.0), 140.0
        )

        assert restored.live == atr.live
        assert restored._closed == atr._closed


def test_state_round_trip_through_json():
    import json

    atr, tf = make(period=3)

    for i in range(1, 8):
        tf.close_bar(atr, Candle(i, high=110.0, low=100.0, close=105.0 + i), 105.0)

    state = json.loads(json.dumps(atr.to_dict()))

    restored = ATR(
        id="atr",
        kind="ATR",
        level=1,
        primary=False,
        overlay=False,
        params={"label": "ATR 14", "period": 3},
    )
    restored.set_state(state)

    assert restored.live == atr.live
    assert list(restored.history) == list(atr.history)
    assert restored._closed == atr._closed


def test_legacy_state_without_closed_field_still_restores():
    # states serialized before `closed` existed must keep working
    atr, tf = make(period=3)

    for i in range(1, 6):
        tf.close_bar(atr, Candle(i, high=110.0, low=100.0, close=105.0), 105.0)

    state = atr.to_dict()
    del state["closed"]

    restored = ATR(
        id="atr",
        kind="ATR",
        level=1,
        primary=False,
        overlay=False,
        params={"label": "ATR 14", "period": 3},
    )
    restored.set_state(state)

    assert restored._closed == atr.history[-1]
