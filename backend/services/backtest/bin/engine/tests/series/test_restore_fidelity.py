"""
Restart fidelity: a series restored with ``set_state`` must continue
exactly like one that was never interrupted.

Every case runs the same price stream twice. The first run is the
reference. The second run is cut in half: the engine publishes its state
where the master would checkpoint it, a fresh engine is rebuilt through
``TradingEngine.set_state`` -- the same call ``main.py`` makes when
``/engine`` comes back -- and both runs are then fed the remaining ticks.
Anything the restore dropped or approximated shows up as a difference in
the published state.

The cut points are chosen on purpose: one inside the warm-up, where a
series suppresses its visible value, and one long after it, where the
visible value is the only alias of the internal chain.
"""

import copy
import json

import msgpack
import pytest

from src.core.engine import TradingEngine
from src.ingestion.tick import Tick


STEP = 10_000          # six ticks per minute: a 1m bar every six ticks
BASE_TS = 1_700_000_000_000


def make_tick(index, price):
    return Tick(
        boot_id="boot",
        config_id="config",
        tick_index=index,
        trade_id=index,
        time=BASE_TS + index * STEP,
        price=price,
        qty=1.0,
        is_buyer_maker=0,
    )


def prices(seed=3, count=1_200):
    """
    A deterministic random walk, so a case that passes does so by
    resuming the chain and not by accident on flat bars.
    """
    import random

    random.seed(seed)

    out = [100.0]

    for _ in range(count):
        out.append(max(1.0, out[-1] + random.uniform(-0.6, 0.6)))

    return out


def series_spec(kind, label, params, level=1):
    return {
        "id": f"{kind}{label}".replace(" ", ""),
        "kind": kind,
        "level": level,
        "primary": kind == "Candlestick",
        "overlay": kind not in ("ATR", "ADX"),
        "params": {"label": label, **params},
        "live": None,
        "history": [],
    }


EMA = series_spec("EMA", "EMA 20", {"period": 20})
ATR = series_spec("ATR", "ATR 14", {"period": 14})
RSI = series_spec("RSI", "RSI 14", {"period": 14})
BB = series_spec("BollingerBands", "BB 20", {"period": 20, "mult": 2.0})
ADX = series_spec(
    "ADX", "ADX 14", {"dilen": 14, "adxlen": 14, "key_level": 23}
)
STOCH = series_spec(
    "StochRSI",
    "StochRSI",
    {"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3},
)
TRAP = series_spec(
    "ReversalTrap",
    "ReversalTrap",
    {"envelope_len": 55, "multiplier": 4.0, "trap_window": 10, "signal_gap": 10},
)
CANDLE = series_spec("Candlestick", "Candles", {}, level=0)


def engine_state(series):
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
                    spec["id"]: spec
                    for spec in series
                },
            }
        },
        "strategy": {"kind": "Strategy1", "params": {}},
        "risk": {"stop": {"type": "percent", "value": 0.01}},
        "portfolio": None,
        "orders": None,
    }


def wire(state, transport):
    """
    The checkpoint crosses a JSON endpoint on the way back in
    (``get-state``) and MessagePack on the way out (Pulsar), so a
    restore is only proven when it survives both.
    """
    if transport == "json":
        return json.loads(json.dumps(state))

    if transport == "msgpack":
        return msgpack.unpackb(
            msgpack.packb(state, use_bin_type=True), raw=False
        )

    return state


def run(engine, stream, lo, hi):
    for index in range(lo, hi):
        engine.on_tick(make_tick(index, stream[index]))


def resume_equals_reference(series, cut, transport="json", seed=3):
    stream = prices(seed)

    reference = TradingEngine()
    reference.set_state("boot", "config", engine_state(series))
    run(reference, stream, 0, cut)

    checkpoint = wire(reference.state.to_dict(), transport)

    restored = TradingEngine()
    restored.set_state("boot", "config", checkpoint)

    run(reference, stream, cut, len(stream))
    run(restored, stream, cut, len(stream))

    return reference.state.to_dict() == restored.state.to_dict()


# The warm-up cuts land before a series exposes ``live``: that is exactly
# when an internal chain used to be rebuilt from ``history`` instead.
WARMUP_CUT = 37
WARM_CUT = 600


@pytest.mark.parametrize("transport", ["in-memory", "json", "msgpack"])
@pytest.mark.parametrize(
    "series",
    [
        pytest.param([EMA], id="ema"),
        pytest.param([ATR], id="atr"),
        pytest.param([RSI], id="rsi"),
        pytest.param([BB], id="bollinger"),
        pytest.param([ADX], id="adx"),
        pytest.param([RSI, STOCH], id="stochrsi"),
        pytest.param([TRAP], id="reversal-trap"),
        pytest.param([CANDLE], id="candlestick"),
        pytest.param(
            [EMA, ATR, RSI, STOCH, BB, ADX, TRAP, CANDLE], id="every-series"
        ),
    ],
)
@pytest.mark.parametrize("cut", [WARMUP_CUT, WARM_CUT])
def test_restart_resumes_the_same_state(series, cut, transport):
    assert resume_equals_reference(series, cut, transport), (
        f"a {transport} checkpoint restored at tick {cut} does not resume "
        f"the same state"
    )


def test_restart_preserves_every_confirmed_bar():
    """
    A restore that quietly drops the newest confirmed bar looks correct
    one tick later, so the bar count is asserted directly: the chain must
    not confirm the previous bar a second time.
    """
    stream = prices()

    reference = TradingEngine()
    reference.set_state("boot", "config", engine_state([ADX]))
    run(reference, stream, 0, 40)

    restored = TradingEngine()
    restored.set_state("boot", "config", copy.deepcopy(reference.state.to_dict()))
    run(reference, stream, 40, 200)
    run(restored, stream, 40, 200)

    reference_history = (
        reference.state.to_dict()["timeframes"]["1m"]["series"]["ADXADX14"][
            "history"
        ]
    )
    restored_history = (
        restored.state.to_dict()["timeframes"]["1m"]["series"]["ADXADX14"][
            "history"
        ]
    )

    assert len(restored_history) == len(reference_history)
    assert restored_history == reference_history


def test_stochrsi_windows_are_restored_not_approximated():
    """
    The %K / %D windows advance once per update, not once per bar, so they
    cannot be rebuilt from the bar history. Rebuilding them shifts the
    oscillator, which is what a strategy reading %K would trade on.
    """
    stream = prices()

    reference = TradingEngine()
    reference.set_state("boot", "config", engine_state([RSI, STOCH]))
    run(reference, stream, 0, 600)

    restored = TradingEngine()
    restored.set_state("boot", "config", copy.deepcopy(reference.state.to_dict()))

    published = restored.state.to_dict()["timeframes"]["1m"]["series"][
        STOCH["id"]
    ]

    assert published["buffers"]["ks"], "the k window was not published"
    assert published["buffers"]["ds"], "the d window was not published"
    assert published["buffers"]["rsis"], "the rsi window was not published"

    def windows(engine):
        series = engine.timeframes["1m"]._series[STOCH["id"]]

        return (
            list(series._rsis),
            list(series._ks),
            list(series._ds),
            series._prev_k,
            series._prev_d,
        )

    assert windows(restored) == windows(reference)

    run(reference, stream, 600, 1_000)
    run(restored, stream, 600, 1_000)

    assert reference.state.to_dict() == restored.state.to_dict()


@pytest.mark.parametrize(
    "series, extra_keys",
    [
        pytest.param([ADX], ["internal", "closed"], id="adx"),
        pytest.param([CANDLE], ["closed"], id="candlestick"),
        pytest.param([RSI], ["internal"], id="rsi"),
        pytest.param([BB], ["internal"], id="bollinger"),
    ],
)
def test_checkpoints_written_by_an_older_engine_still_restore(series, extra_keys):
    """
    Snapshots already on disk (``data/replay.bin``) and checkpoints held
    in the master were written without the fields this fix added. Where
    the dropped state is genuinely recoverable -- `live` is an exact alias
    of the internal chain once the series is warm -- the old checkpoint
    has to keep restoring to the very same thing.
    """
    stream = prices()

    engine = TradingEngine()
    engine.set_state("boot", "config", engine_state(series))
    run(engine, stream, 0, 300)

    legacy = copy.deepcopy(engine.state.to_dict())

    for timeframe in legacy["timeframes"].values():
        for spec in timeframe["series"].values():
            for key in extra_keys:
                spec.pop(key, None)

    # The regression this guards: a restore that raises, or that leaves the
    # series without the state it needs to produce a value at all.
    restored = TradingEngine()
    restored.set_state("boot", "config", legacy)

    assert restored.state.to_dict() == engine.state.to_dict()


@pytest.mark.parametrize(
    "series, extra_keys, label",
    [
        pytest.param(
            [STOCH, RSI], ["internal", "buffers"], STOCH["id"], id="stochrsi"
        ),
        pytest.param([TRAP], ["closed"], TRAP["id"], id="reversal-trap"),
    ],
)
def test_older_checkpoint_falls_back_without_raising(
    series, extra_keys, label
):
    """
    Two series cannot be recovered from a checkpoint written before this
    fix, because what the restore needs was never published: the %K / %D
    windows advance once per update rather than once per bar, and the
    newest confirmed ``ReversalTrap`` bar only reaches ``history`` on the
    bar after it. The old checkpoints are still lossy by nature, so the
    only thing to guarantee is that they keep restoring the way they did
    before the fix instead of failing.
    """
    stream = prices()

    engine = TradingEngine()
    engine.set_state("boot", "config", engine_state(series))
    run(engine, stream, 0, 300)

    legacy = copy.deepcopy(engine.state.to_dict())

    for timeframe in legacy["timeframes"].values():
        for spec in timeframe["series"].values():
            for key in extra_keys:
                spec.pop(key, None)

    # The regression this guards: a restore that raises, or that leaves the
    # series unable to produce a value at all.
    restored = TradingEngine()
    restored.set_state("boot", "config", legacy)

    run(restored, stream, 300, 600)

    published = restored.state.to_dict()["timeframes"]["1m"]["series"][label]

    assert published["history"], f"{label} produced no bars from an old checkpoint"

    # A current checkpoint gets the exact state; compare against a run that
    # never stopped to show the legacy path is the only thing still lossy and
    # not the new code.
    current = TradingEngine()
    current.set_state("boot", "config", engine_state(series))
    run(current, stream, 0, 300)
    run(current, stream, 300, 600)

    current_series = current.state.to_dict()["timeframes"]["1m"]["series"][label]

    assert current_series["history"] != [], f"{label} produced no bars at all"
    assert len(published["history"]) == len(current_series["history"])
