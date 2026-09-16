"""Tests for the engine-compatible ReversalTrap series.

The math is validated against a clean-room, windowed reference
(`_reference`) that does not reuse the streaming implementation, so
any shared bug in the port does not silently reproduce in the oracle.

Extra hand-verified micro datasets cover the signal-gap cooldown, the
shared `last_signal_bar`, the target-before-stop priority, the stop-loss
branch, the RSI edge cases (100 / 0) and the warm-up period.
"""

import json
import random
import pytest

from src.series import ReversalTrap
from src.series.registry import SERIES_REGISTRY
from aggregator.bar_aggregator import Bar
from timeframes.timeframe import Timeframe


# ----------------------------------------------------------------------
# windowed reference implementation (clean room).
# ----------------------------------------------------------------------

def ema_ref(values, length):
    alpha = 2.0 / (length + 1.0)
    out = [None] * len(values)
    acc = 0.0
    cnt = 0
    prev = None
    for i, v in enumerate(values):
        acc += v
        cnt += 1
        if cnt == length:
            prev = acc / length
            out[i] = prev
        elif cnt > length:
            prev = alpha * v + (1.0 - alpha) * prev
            out[i] = prev
    return out


def rma_ref(values, length):
    alpha = 1.0 / length
    out = [None] * len(values)
    acc = 0.0
    cnt = 0
    prev = None
    for i, v in enumerate(values):
        if v is None:
            continue
        acc += v
        cnt += 1
        if cnt == length:
            prev = acc / length
            out[i] = prev
        elif cnt > length:
            prev = alpha * v + (1.0 - alpha) * prev
            out[i] = prev
    return out


def _round(x):
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def _reference(bars, envelope_len, multiplier, trap_window,
                   signal_gap, rsi_len, stop_mult, atr_len,
                   target_source="Basis Line", max_bars=5000,
                   max_trades=500):
    n = len(bars)
    highs = [b[0] for b in bars]
    lows = [b[1] for b in bars]
    closes = [b[2] for b in bars]

    ema = ema_ref(closes, envelope_len)
    tr = []
    for i in range(n):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
    atr_env = rma_ref(tr, envelope_len)
    atr100 = rma_ref(tr, atr_len)

    rsi = [None] * n
    if n >= 2:
        ups = [None] + [max(closes[i] - closes[i - 1], 0.0)
                        for i in range(1, n)]
        dns = [None] + [max(closes[i - 1] - closes[i], 0.0)
                        for i in range(1, n)]
        avg_up = rma_ref(ups, rsi_len)
        avg_down = rma_ref(dns, rsi_len)
        for i in range(n):
            u = avg_up[i]
            d = avg_down[i]
            if u is not None and d is not None:
                if d == 0.0:
                    rsi[i] = 100.0
                elif u == 0.0:
                    rsi[i] = 0.0
                else:
                    rsi[i] = 100.0 - 100.0 / (1.0 + u / d)
    rsi_bucket = [
        (max(0, min(10, _round(rsi[i] / 10.0))) if rsi[i] is not None
         else None)
        for i in range(n)
    ]

    close_above_count = 0
    close_below_count = 0
    last_signal_bar = -1000
    active_bull = False
    active_bear = False
    bull_bucket = None
    bear_bucket = None
    bull_target_price = None
    bull_stop_price = None
    bear_target_price = None
    bear_stop_price = None
    bull_entry_bar = None
    bear_entry_bar = None
    bull_total = [0] * 11
    bull_wins = [0] * 11
    bear_total = [0] * 11
    bear_wins = [0] * 11
    total_trades_count = 0

    refs = []
    for i in range(n):
        high = highs[i]
        low = lows[i]
        close = closes[i]
        prev_close = closes[i - 1] if i > 0 else None
        prev_low = lows[i - 1] if i > 0 else None
        prev_high = highs[i - 1] if i > 0 else None

        basis = ema[i]
        vola = atr_env[i]
        upper = (basis + multiplier * vola
                 if basis is not None and vola is not None else None)
        lower = (basis - multiplier * vola
                 if basis is not None and vola is not None else None)
        atr_stop = (atr100[i] * stop_mult if atr100[i] is not None else None)

        lowest2 = (min(low, prev_low) if prev_low is not None else None)
        highest2 = (max(high, prev_high) if prev_high is not None else None)
        bull_stop_level = (lowest2 - atr_stop if atr_stop is not None else None)
        bear_stop_level = (highest2 + atr_stop if atr_stop is not None else None)

        raw_bear = False
        if upper is not None and close < upper and i > 0:
            before_window = close_above_count <= trap_window
            if high > upper and prev_close < upper and before_window:
                raw_bear = True
            elif prev_close > upper and before_window:
                raw_bear = True
        close_above_count = (
            (close_above_count + 1) if upper is not None and high > upper
            else 0
        )

        raw_bull = False
        if lower is not None and close > lower and i > 0:
            before_window = close_below_count <= trap_window
            if low < lower and prev_close > lower and before_window:
                raw_bull = True
            elif prev_close < lower and before_window:
                raw_bull = True
        close_below_count = (
            (close_below_count + 1) if lower is not None and low < lower
            else 0
        )

        can_fire = (i - last_signal_bar) >= signal_gap
        bull_trap = raw_bull and can_fire
        bear_trap = raw_bear and can_fire
        if bull_trap or bear_trap:
            last_signal_bar = i

        bars_from_live = 0
        allowed_by_limits = (
            bars_from_live <= max_bars and total_trades_count < max_trades
        )

        if bull_trap and not active_bull and allowed_by_limits:
            active_bull = True
            bull_bucket = rsi_bucket[i]
            total_trades_count += 1
            bull_entry_bar = i
            bull_target_price = (basis if target_source == "Basis Line"
                                 else upper)
            bull_stop_price = bull_stop_level

        if active_bull:
            bucket = bull_bucket if bull_bucket is not None else 0
            if (bull_target_price is not None
                    and high >= bull_target_price
                    and (i - last_signal_bar) > 1):
                bull_total[bucket] += 1
                bull_wins[bucket] += 1
                active_bull = False
                bull_entry_bar = None
            elif (bull_stop_price is not None
                    and low < bull_stop_price
                    and (i - last_signal_bar) > 1):
                bull_total[bucket] += 1
                active_bull = False
                bull_entry_bar = None

        if bear_trap and not active_bear and allowed_by_limits:
            active_bear = True
            bear_bucket = rsi_bucket[i]
            total_trades_count += 1
            bear_entry_bar = i
            bear_target_price = (basis if target_source == "Basis Line"
                                 else lower)
            bear_stop_price = bear_stop_level

        if active_bear:
            bucket = bear_bucket if bear_bucket is not None else 0
            if (bear_target_price is not None
                    and low <= bear_target_price
                    and (i - last_signal_bar) > 1):
                bear_total[bucket] += 1
                bear_wins[bucket] += 1
                active_bear = False
                bear_entry_bar = None
            elif (bear_stop_price is not None
                    and high > bear_stop_price
                    and (i - last_signal_bar) > 1):
                bear_total[bucket] += 1
                active_bear = False
                bear_entry_bar = None

        refs.append({
            "time": i * 60,
            "start_ts": i * 60000,
            "end_ts": i * 60000 + 60000,
            "bar_index": i,
            "basis": basis,
            "upper_band": upper,
            "lower_band": lower,
            "atr_envelope": vola,
            "rsi": rsi[i],
            "rsi_bucket": rsi_bucket[i],
            "atr_stop": atr_stop,
            "bull_stop_level": bull_stop_level,
            "bear_stop_level": bear_stop_level,
            "raw_bull_trap": raw_bull,
            "raw_bear_trap": raw_bear,
            "bull_trap": bull_trap,
            "bear_trap": bear_trap,
            "close_above_envelope_count": close_above_count,
            "close_below_envelope_count": close_below_count,
            "active_bull": active_bull,
            "active_bear": active_bear,
            "bull_bucket": bull_bucket,
            "bear_bucket": bear_bucket,
            "bull_target_price": bull_target_price,
            "bull_stop_price": bull_stop_price,
            "bear_target_price": bear_target_price,
            "bear_stop_price": bear_stop_price,
            "bull_entry_bar": bull_entry_bar,
            "bear_entry_bar": bear_entry_bar,
            "allowed_by_limits": allowed_by_limits,
            "bull_total": tuple(bull_total),
            "bull_wins": tuple(bull_wins),
            "bear_total": tuple(bear_total),
            "bear_wins": tuple(bear_wins),
            "total_trades_count": total_trades_count,
            "last_signal_bar": last_signal_bar,
        })

    return refs


COMPARE_FIELDS = [
    "time", "start_ts", "end_ts", "bar_index", "basis", "upper_band",
    "lower_band", "atr_envelope", "rsi", "rsi_bucket", "atr_stop",
    "bull_stop_level", "bear_stop_level", "raw_bull_trap", "raw_bear_trap",
    "bull_trap", "bear_trap", "close_above_envelope_count",
    "close_below_envelope_count", "active_bull", "active_bear",
    "bull_bucket", "bear_bucket", "bull_target_price", "bull_stop_price",
    "bear_target_price", "bear_stop_price", "bull_entry_bar",
    "bear_entry_bar", "allowed_by_limits", "bull_total", "bull_wins",
    "bear_total", "bear_wins", "total_trades_count", "last_signal_bar",
]


def assert_value_matches(value, ref):
    assert value is not None, "expected a value, got None"
    for key in COMPARE_FIELDS:
        ev = getattr(value, key)
        rv = ref[key]
        if isinstance(rv, tuple):
            assert ev == rv, (key, ev, rv)
        elif rv is None:
            assert ev is None, (key, ev)
        elif isinstance(rv, float):
            assert ev == pytest.approx(rv), (key, ev, rv)
        else:
            assert ev == rv, (key, ev, rv)


# ----------------------------------------------------------------------
# Harness: drive the series through a real Timeframe, one bar per step.
# ----------------------------------------------------------------------

class Harness:
    def __init__(self, **params):
        self.series = ReversalTrap(
            id="rt",
            kind="ReversalTrap",
            level=0,
            primary=False,
            overlay=True,
            params=params,
        )
        tf = Timeframe()
        tf.add_series(self.series)
        tf.build_levels()
        self.tf = tf

    @staticmethod
    def bar(high, low, close, idx):
        ts = idx * 60000
        return Bar(
            time=ts // 1000,
            open=close,
            high=high,
            low=low,
            close=close,
            start_ts=ts,
            end_ts=ts + 60000,
        )

    def feed(self, bars):
        """Feed each bar as a newly opened (closing the previous one)."""
        lives = []
        prev = None
        for bar in bars:
            self.tf.live = bar
            self.tf.closed = prev
            self.tf.is_new = True
            self.tf.is_closed = prev is not None
            self.tf.update()
            lives.append(self.series.live)
            prev = bar
        return lives

    def flush(self):
        self.tf.flush()
        return self.series.live


# ----------------------------------------------------------------------
# Utility: deterministic noisy bars with wick spikes (trap generators).
# ----------------------------------------------------------------------

def make_bars(n, seed):
    rnd = random.Random(seed)
    closes = [100.0]
    for _ in range(1, n):
        closes.append(closes[-1] + rnd.uniform(-2.2, 2.2))
    bars = []
    for i in range(n):
        c = closes[i]
        wick_up = rnd.uniform(8.0, 16.0) if i % 5 == 2 else \
            rnd.uniform(0.0, 1.2)
        wick_dn = rnd.uniform(8.0, 16.0) if i % 7 == 3 else \
            rnd.uniform(0.0, 1.2)
        bars.append((c + wick_up, c - wick_dn, c))
    return bars


# ----------------------------------------------------------------------
# Params / plumbing
# ----------------------------------------------------------------------

def test_defaults_and_descriptor_params():
    series = ReversalTrap(
        id="rt", kind="ReversalTrap", level=0, primary=False, overlay=True,
        params={},
    )
    assert series.envelope_len == 55
    assert series.multiplier == 4.0
    assert series.trap_window == 10
    assert series.signal_gap == 10
    assert series.rsi_len == 20
    assert series.stop_mult == 0.5
    assert series.max_bars == 5000
    assert series.max_trades == 500
    assert series.atr_len == 100
    assert series.target_source == "Basis Line"
    assert series._warmup == 99

    desc = ReversalTrap(
        id="rt", kind="ReversalTrap", level=0, primary=False, overlay=True,
        params={
            "signal_gap": {"value": 5},
            "multiplier": {"value": 1.5},
            "envelope_len": {"value": 12},
        },
    )
    assert desc.signal_gap == 5
    assert desc.multiplier == 1.5
    assert desc.envelope_len == 12
    assert desc._warmup == max(11, 20, 99, 1)


@pytest.mark.parametrize("key", ["envelope_len", "rsi_len", "atr_len",
                                 "trap_window", "signal_gap"])
def test_invalid_length_params_raise(key):
    with pytest.raises(ValueError):
        ReversalTrap(
            id="rt", kind="ReversalTrap", level=0, primary=False,
            overlay=True, params={key: 0},
        )


@pytest.mark.parametrize("key", ["multiplier", "stop_mult"])
def test_invalid_positive_params_raise(key):
    with pytest.raises(ValueError):
        ReversalTrap(
            id="rt", kind="ReversalTrap", level=0, primary=False,
            overlay=True, params={key: 0},
        )


def test_registry_and_package_export():
    assert SERIES_REGISTRY["ReversalTrap"] is ReversalTrap
    assert ReversalTrap.__name__ == "ReversalTrap"


# ----------------------------------------------------------------------
# Warm-up: live suppressed until every chain is seeded.
# ----------------------------------------------------------------------

def test_warmup_suppresses_live_only():
    h = Harness()
    bars = [Harness.bar(10.0, 10.0, 10.0, i) for i in range(120)]
    lives = h.feed(bars)

    assert all(v is None for v in lives[:99])
    assert lives[99] is not None

    # The final bar is only provisional until a flush confirms it.
    assert h.series._closed.bar_index == 118
    assert len(h.series.history) == 118

    # Confirmed state is never suppressed by the warm-up period.
    h.flush()
    assert h.series.live.bar_index == 119
    assert h.series._closed.bar_index == 119
    assert len(h.series.history) == 119
    assert h.series.history[0].bar_index == 0


# ----------------------------------------------------------------------
# RSI edge cases: 100 when no downward momentum, 0 when none upward.
# ----------------------------------------------------------------------

def test_rsi_boundary_buckets():
    up_h = Harness(envelope_len=2, rsi_len=2, atr_len=2)
    up = up_h.feed([Harness.bar(10, 10, 10, 0),
                    Harness.bar(11, 11, 11, 1),
                    Harness.bar(12, 12, 12, 2)])
    assert up[2].rsi == pytest.approx(100.0)
    assert up[2].rsi_bucket == 10

    down_h = Harness(envelope_len=2, rsi_len=2, atr_len=2)
    down = down_h.feed([Harness.bar(10, 10, 10, 0),
                        Harness.bar(9, 8, 8, 1),
                        Harness.bar(7, 6, 6, 2)])
    assert down[2].rsi == pytest.approx(0.0)
    assert down[2].rsi_bucket == 0


# ----------------------------------------------------------------------
# Hand-verified trap sequence, cooldown and shared last_signal_bar.
# ----------------------------------------------------------------------

def test_trap_sequence_cooldown_and_exits():
    h = Harness(envelope_len=2, multiplier=1.0, trap_window=10,
                signal_gap=2, rsi_len=2, stop_mult=0.5, atr_len=2)
    bars = [
        Harness.bar(10, 10, 10, 0),
        Harness.bar(11, 11, 11, 1),
        Harness.bar(10, 10, 10, 2),   # bear trap
        Harness.bar(5, 5, 5, 3),      # raw bear suppressed (cooldown)
        Harness.bar(7, 2, 6, 4),      # bull trap; resets last_signal_bar
        Harness.bar(10, 9, 9, 5),     # gap still 1 -> no exits
        Harness.bar(8, 7, 7, 6),      # both trackers hit target -> wins
    ]
    lives = h.feed(bars)

    assert lives[0] is None
    assert lives[1] is None

    # Bar 2: bear trap fires (RSI == 50 -> bucket 5).
    v2 = lives[2]
    assert v2.bear_trap is True
    assert v2.bull_trap is False
    assert v2.raw_bear_trap is True
    assert v2.active_bear is True
    assert v2.bear_entry_bar == 2
    assert v2.bear_bucket == 5
    assert v2.rsi == pytest.approx(50.0)
    assert v2.rsi_bucket == 5
    assert v2.bear_target_price == pytest.approx(10.166666666666666)
    assert v2.bear_stop_price == pytest.approx(11.375)
    assert v2.last_signal_bar == 2
    assert v2.active_bull is False
    assert v2.total_trades_count == 1

    # Bar 3: raw bear is still true but the signal-gap cooldown applies.
    v3 = lives[3]
    assert v3.raw_bear_trap is True
    assert v3.bear_trap is False
    assert v3.active_bear is True
    assert v3.bear_entry_bar == 2
    assert v3.bull_trap is False

    # Bar 4: bull trap fires; both trackers stay active because the bull
    # trap just reset last_signal_bar to 4 (gap == 0 for both engines).
    v4 = lives[4]
    assert v4.bull_trap is True
    assert v4.bear_trap is False
    assert v4.active_bull is True
    assert v4.active_bear is True
    assert v4.bull_entry_bar == 4
    assert v4.bear_entry_bar == 2
    assert v4.bull_bucket == 3
    assert v4.rsi == pytest.approx(31.25)
    assert v4.rsi_bucket == 3
    assert v4.bull_target_price == pytest.approx(6.240740740740741)
    assert v4.bull_stop_price == pytest.approx(0.03125)
    assert v4.last_signal_bar == 4
    assert v4.total_trades_count == 2

    # Bar 5: gap == 1 for both trackers -> no exit can happen yet.
    v5 = lives[5]
    assert v5.active_bull is True
    assert v5.active_bear is True

    # Bar 6: gap == 2 -> both hit their targets and record wins.
    v6 = lives[6]
    assert v6.active_bull is False
    assert v6.active_bear is False
    assert v6.bull_entry_bar is None
    assert v6.bear_entry_bar is None
    assert v6.bull_total[3] == 1
    assert v6.bull_wins[3] == 1
    assert v6.bear_total[5] == 1
    assert v6.bear_wins[5] == 1
    assert v6.total_trades_count == 2


# ----------------------------------------------------------------------
# White-box exits: stop-loss branch and target-before-stop priority.
# ----------------------------------------------------------------------

def test_bull_loss_via_stop():
    h = Harness(envelope_len=2, multiplier=1.0, trap_window=10,
                signal_gap=2, rsi_len=2, stop_mult=0.5, atr_len=2)
    bars = [
        Harness.bar(10, 10, 10, 0),
        Harness.bar(11, 11, 11, 1),
        Harness.bar(12, 8, 11, 2),    # bull trap (RSI == 100 -> bucket 10)
        Harness.bar(14, 4, 9, 3),     # gap 1: no exit
        Harness.bar(9, 5, 8, 4),      # gap 2: hits stop only -> loss
    ]
    lives = h.feed(bars)

    v2 = lives[2]
    assert v2.bull_trap is True
    assert v2.active_bull is True
    assert v2.bull_bucket == 10
    assert v2.bull_target_price == pytest.approx(10.833333333333334)
    assert v2.bull_stop_price == pytest.approx(6.875)
    assert v2.total_trades_count == 1

    assert lives[3].active_bull is True

    v4 = lives[4]
    assert v4.active_bull is False
    assert v4.bull_entry_bar is None
    assert v4.bull_total[10] == 1
    assert v4.bull_wins[10] == 0
    assert v4.active_bear is False
    assert v4.total_trades_count == 1


def test_bull_target_wins_over_stop_on_same_bar():
    h = Harness(envelope_len=2, multiplier=1.0, trap_window=10,
                signal_gap=2, rsi_len=2, stop_mult=0.5, atr_len=2)
    bars = [
        Harness.bar(10, 10, 10, 0),
        Harness.bar(11, 11, 11, 1),
        Harness.bar(12, 8, 11, 2),    # bull trap (bucket 10)
        Harness.bar(14, 4, 9, 3),     # gap 1: no exit
        Harness.bar(14, 4, 9, 4),     # gap 2: target AND stop both hit
    ]
    lives = h.feed(bars)

    assert lives[2].bull_trap is True

    v4 = lives[4]
    # Target is evaluated before the stop, so a double-hit resolves as a
    # win even though the low also pierced the stop price.
    assert v4.active_bull is False
    assert v4.bull_total[10] == 1
    assert v4.bull_wins[10] == 1
    assert v4.active_bear is False
    assert v4.total_trades_count == 1


# ----------------------------------------------------------------------
# Equivalence against the clean-room reference on noisy data.
# ----------------------------------------------------------------------

REGRESSION_PARAMS = dict(
    envelope_len=10,
    multiplier=1.2,
    trap_window=8,
    signal_gap=3,
    rsi_len=5,
    stop_mult=0.5,
    atr_len=10,
    max_trades=100000,
)


def _check_full_equivalence(params, bars, target_source):
    ref = _reference(
        [(b.high, b.low, b.close) for b in bars],
        envelope_len=params["envelope_len"],
        multiplier=params["multiplier"],
        trap_window=params["trap_window"],
        signal_gap=params["signal_gap"],
        rsi_len=params["rsi_len"],
        stop_mult=params["stop_mult"],
        atr_len=params["atr_len"],
        target_source=target_source,
        max_bars=params.get("max_bars", 5000),
        max_trades=params["max_trades"],
    )

    h = Harness(target_source=target_source, **params)
    lives = h.feed(bars)

    # Every visible provisional live value must equal the reference bar.
    for i, (live, rv) in enumerate(zip(lives, ref)):
        if live is not None:
            assert live.bar_index == i
            assert_value_matches(live, rv)

    # After the final flush the whole confirmed history must match too.
    h.flush()
    assert len(h.series.history) == len(lives) - 1
    for value, rv in zip(h.series.history, ref[:-1]):
        assert_value_matches(value, rv)
    assert_value_matches(h.series.live, ref[-1])

    return h, ref


def test_pine_equivalence_basis_target():
    bars = [Harness.bar(*ohlc, i)
            for i, ohlc in enumerate(make_bars(260, seed=7))]
    _check_full_equivalence(REGRESSION_PARAMS, bars, "Basis Line")


def test_pine_equivalence_envelope_target():
    bars = [Harness.bar(*ohlc, i)
            for i, ohlc in enumerate(make_bars(260, seed=11))]
    _check_full_equivalence(REGRESSION_PARAMS, bars, "Envelope")


def test_max_trades_guard_matches_reference():
    params = dict(REGRESSION_PARAMS, max_trades=3)
    bars = [Harness.bar(*ohlc, i)
            for i, ohlc in enumerate(make_bars(260, seed=23))]
    h, ref = _check_full_equivalence(params, bars, "Basis Line")

    assert h.series.live.total_trades_count == 3

    # Once the guardrail is saturated, candidate traps with no active
    # position and no cooldown must still be blocked by the limits.
    blocked = [
        rv for rv in ref[1:]
        if (rv["bull_trap"] or rv["bear_trap"])
        and not rv["active_bull"] and not rv["active_bear"]
        and rv["total_trades_count"] == 3
    ]
    assert blocked
    assert all(not rv["allowed_by_limits"] for rv in blocked)
    assert all(rv["bull_entry_bar"] is None and rv["bear_entry_bar"] is None
               for rv in blocked)


# ----------------------------------------------------------------------
# Serialization round-trip.
# ----------------------------------------------------------------------

def test_state_round_trip_through_json():
    h = Harness(envelope_len=2, multiplier=1.0, trap_window=10,
                signal_gap=2, rsi_len=2, stop_mult=0.5, atr_len=2)
    bars = [Harness.bar(*ohlc, i) for i, ohlc in enumerate([
        (10, 10, 10), (11, 11, 11), (10, 10, 10), (5, 5, 5),
        (7, 2, 6), (10, 9, 9), (8, 7, 7),
    ])]
    h.feed(bars)
    h.flush()

    original = h.series.to_dict()
    wire = json.loads(json.dumps(original))

    restored = ReversalTrap(
        id="rt", kind="ReversalTrap", level=0, primary=False, overlay=True,
        params=wire["params"],
    )
    restored.set_state(wire)

    assert restored.live.start_ts == h.series.live.start_ts
    assert restored.history[0].bar_index == h.series.history[0].bar_index
    assert restored.history[-1].bar_index == h.series.history[-1].bar_index
    assert restored.live.bull_total == h.series.live.bull_total
    assert restored.live.bear_total == h.series.live.bear_total
    assert restored.to_dict() == original