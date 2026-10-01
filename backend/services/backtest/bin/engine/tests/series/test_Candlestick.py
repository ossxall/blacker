"""
Candlestick y su confirmación.

La serie expone `_closed` como las demás, pero con una diferencia
deliberada: en EMA/ATR `history` va una vela por detrás de `_closed`, y
aquí no. `history` es el registro que dibuja el frontend, así que
adelantarlo una vela para que las dos SERIES_quadrEN igual costaría la vela
cerrada más reciente en todos los estados publicados.

Lo que hay que garantizar es que la serie nunca quede una vela por detrás de
`timeframe.closed`, y en particular tras un `flush()`, que es donde se
colaba.
"""

from src.series import Candlestick
from src.series.registry import SERIES_REGISTRY
import json
import pytest


class Bar:
    def __init__(self, index, close):
        self.time = index
        self.start_ts = index * 1000
        self.end_ts = (index + 1) * 1000
        self.open = close
        self.high = close
        self.low = close
        self.close = close
        self.total_volume = 1.0


class FakeTimeframe:
    def __init__(self):
        self.timeframe_ms = 60_000
        self.closed = None
        self.live = None
        self.is_new = False
        self.is_closed = False

    def tick(self, bar):
        """Another tick on the bar that is already open."""

        self.live = bar
        self.update()

    def rollover(self, bar):
        """The bar that was live is closed and a new one opens."""

        self.closed = self.live
        self.live = bar
        self.is_new = True
        self.is_closed = True
        self.update()

    def flush(self):
        """The bar that was live is finalized and no new bar replaces it."""

        self.closed = self.live
        self.is_new = False
        self.is_closed = True
        self.update()

    def update(self):
        for series in self._series.values():
            series.update()

        self.is_new = False
        self.is_closed = False


def make():
    candles = Candlestick(
        id="candles",
        kind="Candlestick",
        level=0,
        primary=True,
        overlay=False,
        params={"label": "Candles"},
    )

    tf = FakeTimeframe()
    tf._series = {"candles": candles}
    candles._timeframe = tf

    return candles, tf


def run(candles, tf, count):
    tf.rollover(Bar(1, 100.0))

    for index in range(2, count + 1):
        tf.rollover(Bar(index, 100.0 + index))


def test_first_bar_is_not_confirmed():
    candles, tf = make()

    tf.rollover(Bar(1, 100.0))

    assert candles._closed is None
    assert list(candles.history) == []
    assert candles.live.start_ts == 1000


def test_rollover_confirms_the_bar_that_just_ended():
    candles, tf = make()

    run(candles, tf, 5)

    # `timeframe.closed` is the bar before the open one.
    assert candles._closed.start_ts == tf.closed.start_ts
    assert candles._closed is candles.history[-1]


def test_history_keeps_meaning_the_last_closed_bar():
    """
    The arrangement that makes Candlestick differ from EMA/ATR: `_closed`
    and `history[-1]` are the same bar, so `history[-2]` is the one before.
    This is the contract a reader builds on, and it must not shift.
    """

    candles, tf = make()

    run(candles, tf, 5)

    assert [c.start_ts for c in candles.history] == [1000, 2000, 3000, 4000]
    assert candles.history[-1].start_ts == 4000
    assert candles.history[-2].start_ts == 3000


def test_flush_confirms_the_finalized_bar():
    """
    The regression: `flush()` sets `is_new = False` on purpose, so a series
    that only confirmed on `is_new` stayed one bar behind its Timeframe.
    """

    candles, tf = make()

    run(candles, tf, 5)

    history_before = len(candles.history)

    tf.flush()

    assert candles._closed.start_ts == tf.closed.start_ts
    assert len(candles.history) == history_before + 1
    assert candles.history[-1] is candles._closed


def test_a_flush_does_not_confirm_the_same_bar_twice():
    candles, tf = make()

    run(candles, tf, 5)

    tf.flush()

    confirmed = candles._closed
    history_before = len(candles.history)

    # `Timeframe.flush()` returns early when there is no live bar, so a
    # second one is a no-op rather than a duplicate.
    tf.flush()

    assert candles._closed is confirmed
    assert len(candles.history) == history_before


def test_a_bar_starting_after_a_flush_is_not_confirmed_twice():
    """
    The tricky one: `flush()` leaves `live` pointing at the finalized bar
    while `timeframe.live` is cleared. The bar born after it used to push
    that same candle into `history` a second time.
    """

    candles, tf = make()

    run(candles, tf, 5)

    tf.flush()

    history_before = len(candles.history)
    confirmed_before = candles._closed

    tf.rollover(Bar(6, 106.0))

    assert len(candles.history) == history_before
    assert candles._closed is confirmed_before, "a bar was confirmed twice"
    assert candles.live.start_ts == 6000


def test_restored_series_resumes_the_same_confirmation():
    candles, tf = make()

    run(candles, tf, 6)

    restored = Candlestick(
        id="candles",
        kind="Candlestick",
        level=0,
        primary=True,
        overlay=False,
        params={"label": "Candles"},
    )
    restored_tf = FakeTimeframe()
    restored_tf._series = {"candles": restored}
    restored._timeframe = restored_tf
    restored.set_state(json.loads(json.dumps(candles.to_dict())))
    # Neither the Timeframe's open bar nor its flags are part of the series
    # state, so the restored one has to be handed the same bar back. It is a
    # `Bar`, not a `Candle`: the Timeframe always speaks in bars.
    assert isinstance(tf.live, Bar)
    restored_tf.live = Bar(6, 106.0)

    assert restored._closed == candles._closed

    tf.rollover(Bar(7, 107.0))
    restored_tf.rollover(Bar(7, 107.0))

    assert restored._closed == candles._closed
    assert list(restored.history) == list(candles.history)
    assert restored.live == candles.live


def test_legacy_state_without_closed_field_still_restores():
    candles, tf = make()

    run(candles, tf, 6)

    state = candles.to_dict()
    del state["closed"]

    restored = Candlestick(
        id="candles",
        kind="Candlestick",
        level=0,
        primary=True,
        overlay=False,
        params={"label": "Candles"},
    )
    restored.set_state(state)

    # `history[-1]` is the exact alias of `_closed` here, so an old
    # checkpoint restores to the very same state.
    assert restored._closed == candles._closed
    assert restored._closed == candles.history[-1]


def test_a_fresh_series_restores_to_nothing():
    restored = Candlestick(
        id="candles",
        kind="Candlestick",
        level=0,
        primary=True,
        overlay=False,
        params={"label": "Candles"},
    )
    restored.set_state(
        {"id": "candles", "kind": "Candlestick", "params": {}, "live": None, "history": []}
    )

    assert restored._closed is None
    assert restored.live is None


def test_registry_instantiates_and_restores():
    assert "Candlestick" in SERIES_REGISTRY

    candles = SERIES_REGISTRY["Candlestick"](
        "candles",
        "Candlestick",
        0,
        True,
        False,
        {"label": "Candles"},
    )

    candles.set_state({"live": None, "history": []})

    assert candles.kind == "Candlestick"
    assert candles._closed is None