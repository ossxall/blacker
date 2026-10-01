from src.series import ADX
from src.series.registry import SERIES_REGISTRY
import json
import pytest


class Candle:
    def __init__(self, high, low, close):
        self.time = 1
        self.start_ts = 1
        self.end_ts = 2
        self.high = high
        self.low = low
        self.close = close


class BarCandle:
    """A candle that carries its own timestamps, so bars are told apart."""

    def __init__(self, index, high, low, close):
        self.time = index
        self.start_ts = index * 1000
        self.end_ts = (index + 1) * 1000
        self.high = high
        self.low = low
        self.close = close


class FakeTimeframe:
    def __init__(self):
        self.timeframe_ms = 60_000
        self.closed = None
        self.live = None
        self.is_closed = False

    def rollover(self, bar):
        """The bar that was live is closed and a new one opens."""

        self.closed = self.live
        self.live = bar
        self.is_closed = True

    def flush(self):
        """The bar that was live is finalized and no new bar replaces it."""

        self.closed = self.live
        self.is_closed = True


def make(dilen=2, adxlen=2):
    adx = ADX(
        id="adx",
        kind="ADX",
        level=1,
        primary=False,
        overlay=True,
        params={"dilen": dilen, "adxlen": adxlen, "key_level": 23},
    )

    return adx, FakeTimeframe()


def run_bars(adx, tf, bars):
    """
    Feed one tick per bar, which is the shortest path a bar takes: each new
    bar closes the previous one, so every bar but the last is confirmed.
    """

    adx._timeframe = tf

    tf.rollover(bars[0])
    adx.update()
    tf.is_closed = False

    for bar in bars[1:]:
        tf.rollover(bar)
        adx.update()
        tf.is_closed = False


RISING = [
    BarCandle(i, 100.0 + i, 90.0 + i, 100.0 + i)
    for i in range(1, 9)
]


def test_closed_confirms_the_bar_that_just_ended():
    adx, tf = make()

    run_bars(adx, tf, RISING)

    # The last bar of the run is still open, so the confirmed value is the
    # one before it -- and it is exactly what `history` received.
    assert adx._closed == adx.history[-1]
    assert adx._closed.start_ts == RISING[-2].start_ts


def tick(adx, tf, bar):
    """Another tick on the bar that is already open."""

    tf.live = bar
    adx.update()


def test_closed_never_follows_the_open_bar():
    """
    The regression this contract exists for: `live` is the bar that is still
    open and moves on every tick, so a reader that used it would repaint.
    `_closed` has to stay on the confirmed bar while `live` advances.
    """

    adx, tf = make()

    run_bars(adx, tf, RISING)

    confirmed = adx._closed

    # The open bar keeps taking ticks and `live` moves with it.
    tick(adx, tf, BarCandle(8, 130.0, 70.0, 120.0))

    assert adx.live != confirmed
    assert adx._closed == confirmed


def test_closed_is_suppressed_during_warmup():
    """
    adx / +DI / -DI are not meaningful until the chain is warm, which is why
    `live` is suppressed. Publishing them in `_closed` before that would
    hand a reader the value that `live` refuses to give.
    """

    adx, tf = make(dilen=14, adxlen=14)

    run_bars(adx, tf, RISING)

    assert adx.history, "the chain advanced"
    assert adx.live is None
    assert adx._closed is None

    run_bars(adx, tf, [
        BarCandle(i, 100.0 + i, 90.0 + i, 100.0 + i)
        for i in range(9, 40)
    ])

    assert adx.live is not None
    assert adx._closed is not None
    assert adx._closed == adx.history[-1]


def test_flush_confirms_the_last_bar():
    """
    `Timeframe.flush()` finalizes the live bar without opening a new one, so
    the confirmation happens after the step is computed rather than on the
    next rollover. That is the one path where `_closed` is not `history[-1]`,
    and it is why `_closed` is published instead of read off `history`.
    """

    adx, tf = make()

    run_bars(adx, tf, RISING)

    history_before = len(adx.history)

    # Before the flush the open bar is unconfirmed.
    assert adx._closed.start_ts == RISING[-2].start_ts
    assert adx.live.start_ts == RISING[-1].start_ts

    tf.flush()
    adx.update()

    # The finalized bar is now the confirmed one.
    assert adx._closed.start_ts == RISING[-1].start_ts
    assert adx._closed == adx.live

    # It was confirmed once, and `history` still lags it by one bar.
    assert len(adx.history) == history_before
    assert adx._closed != adx.history[-1]

    # The next rollover confirms the bar after it and they line up again.
    tf.rollover(BarCandle(9, 115.0, 85.0, 110.0))
    adx.update()

    assert adx._closed == adx.history[-1]


def test_restored_series_resumes_the_same_confirmation():
    adx, tf = make()

    run_bars(adx, tf, RISING)

    restored = ADX(
        id="adx",
        kind="ADX",
        level=1,
        primary=False,
        overlay=True,
        params={"dilen": 2, "adxlen": 2, "key_level": 23},
    )
    restored_tf = FakeTimeframe()
    restored._timeframe = restored_tf
    restored.set_state(json.loads(json.dumps(adx.to_dict())))

    assert restored._closed == adx._closed

    more = [
        BarCandle(i, 100.0 + i, 90.0 + i, 100.0 + i)
        for i in range(9, 14)
    ]

    for bar in more:
        tf.rollover(bar)
        adx.update()
        tf.is_closed = False

        restored_tf.rollover(bar)
        restored.update()
        restored_tf.is_closed = False

        assert restored._closed == adx._closed
        assert restored.live == adx.live
        assert list(restored.history) == list(adx.history)


def test_legacy_state_without_closed_field_still_restores():
    adx, tf = make()

    run_bars(adx, tf, RISING)

    state = adx.to_dict()
    del state["closed"]

    restored = ADX(
        id="adx",
        kind="ADX",
        level=1,
        primary=False,
        overlay=True,
        params={"dilen": 2, "adxlen": 2, "key_level": 23},
    )
    restored.set_state(state)

    # `history[-1]` is the exact alias of `_closed` on a checkpoint taken
    # with a bar open, which is every checkpoint the engine publishes.
    assert restored._closed == adx._closed
    assert restored._closed == adx.history[-1]


def test_compute_step_first_candle_initializes_state():
    adx = ADX(
        id="adx",
        kind="ADX",
        level=0,
        primary=False,
        overlay=True,
        params={},
    )

    candle = Candle(
        high=110.0,
        low=100.0,
        close=105.0,
    )

    result = adx._compute_step(
        candle=candle,
        prev_chain=None,
        prev1=None,
        prev2=None,
    )

    # True Range initializes from the candle range.
    assert result.tr_rma == pytest.approx(10.0)

    # Directional movement starts at zero.
    assert result.plus_dm_rma == pytest.approx(0.0)
    assert result.minus_dm_rma == pytest.approx(0.0)

    # Directional indicators are zero.
    assert result.plus_di == pytest.approx(0.0)
    assert result.minus_di == pytest.approx(0.0)

    # DX = 0, therefore the first ADX is also zero.
    assert result.adx == pytest.approx(0.0)

    # No trend information exists yet.
    assert result.adx_color == "red"
    assert result.is_reversal is False
    assert result.reversal_level is None

    # Internal state must preserve the candle values.
    assert result.high == 110.0
    assert result.low == 100.0
    assert result.close == 105.0


def test_registry_instantiates_and_restores_like_candlestick():
    # ADX must be registered the same way Candlestick is, so the
    # engine can build and restore an ADX from serialized state.
    assert "ADX" in SERIES_REGISTRY

    adx = SERIES_REGISTRY["ADX"](
        id="adx",
        kind="ADX",
        level=1,
        primary=False,
        overlay=True,
        params={"dilen": 14, "adxlen": 14, "key_level": 23},
    )

    state = {
        "id": "adx",
        "kind": "ADX",
        "level": 1,
        "primary": False,
        "overlay": True,
        "params": {"dilen": 14, "adxlen": 14, "key_level": 23},
        "live": None,
        "history": [],
    }

    adx.set_state(state)

    assert adx.id == "adx"
    assert adx.kind == "ADX"
    assert adx.dilen == 14
    assert adx.adxlen == 14
    assert adx.key_level == pytest.approx(23.0)
    assert adx.live is None
    assert adx._internal is None


def test_params_accept_ui_descriptor_objects():
    # The frontend sends parameter descriptor objects carrying the value
    # (as it does for EMA), so the engine must unwrap them on instantiation.
    adx = ADX(
        id="adx",
        kind="ADX",
        level=1,
        primary=False,
        overlay=True,
        params={
            "dilen": {"value": 21, "affectsCompute": True, "min": 1},
            "adxlen": {"value": 7, "affectsCompute": True, "min": 1},
            "key_level": {"value": 30, "affectsCompute": True, "min": 1},
            "label": "ADX",
        },
    )

    assert adx.dilen == 21
    assert adx.adxlen == 7
    assert adx.key_level == pytest.approx(30.0)