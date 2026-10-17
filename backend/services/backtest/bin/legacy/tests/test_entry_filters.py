from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.config import Config
from strategy.indicators import atr, htf_trend
from strategy.strategy import EMACrossStrategy


def _bars(closes, rng):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="30min")
    o = [float(x) for x in closes]
    return pd.DataFrame(
        {
            "open": o,
            "high": [x * (1 + rng) for x in o],
            "low": [x * (1 - rng) for x in o],
            "close": o,
            "volume": 1.0,
        },
        index=idx,
    )


def test_entry_filters_disabled_by_default():
    c = Config()
    # `entry_htf_trend_enabled` es un toggle (su default puede cambiarse sin
    # romper el contrato): no se fija aquí.
    assert c.entry_max_dist_slow_pct == 0.0
    assert c.entry_atr_max_pct == 0.0
    assert c.entry_atr_len == 14


def test_no_chase_blocks_when_far_but_not_when_close():
    idx = pd.date_range("2020-01-01", periods=120, freq="30min")
    closes = [100] * 60 + [130] * 60
    o, h, l = (
        [1.001 * x for x in closes],
        [1.002 * x for x in closes],
        [0.999 * x for x in closes],
    )
    bars = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": l,
            "close": closes,
            "volume": np.ones(len(closes)),
        },
        index=idx,
    )
    c = Config(
        ema_fast=5,
        ema_slow=20,
        confirm_bars=0,
        entry_adx_enabled=False,
        entry_max_dist_slow_pct=0.03,
    )
    sig = EMACrossStrategy(c).generate_signals(bars)
    assert sig["target"].sum() == 0
    c2 = Config(
        ema_fast=5,
        ema_slow=20,
        confirm_bars=0,
        entry_adx_enabled=False,
        entry_max_dist_slow_pct=0.3,
    )
    sig2 = EMACrossStrategy(c2).generate_signals(bars)
    assert sig2["target"].sum() > 0


def test_htf_trend_helper_on_same_tf():
    idx = pd.date_range("2020-01-01 00:00", periods=10, freq="30min")
    closes = np.arange(100, 100 + 10)
    bars = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": 1,
        },
        index=idx,
    )
    t = htf_trend(bars, "30min", 2, 3, "30min")
    assert t is not None
    assert (t == 1).any()


def test_atr_helper_is_aligned_and_positive():
    bars = _bars([100] * 40, 0.002)
    a = atr(bars, 14)
    assert len(a) == len(bars)
    assert a.index.equals(bars.index)
    assert (a.to_numpy()[14:] > 0).all()


def test_atr_filter_blocks_high_vol_and_allows_low_vol():
    closes = [100.0] * 60 + [100.0 + 0.2 * i for i in range(1, 61)]

    def targets(bars, max_pct):
        c = Config(
            ema_fast=5,
            ema_slow=20,
            confirm_bars=0,
            entry_adx_enabled=False,
            entry_atr_max_pct=max_pct,
        )
        return int(EMACrossStrategy(c).generate_signals(bars)["target"].sum())

    high_vol = _bars(closes, 0.02)  # ATR/close ~ 4%
    low_vol = _bars(closes, 0.0005)  # ATR/close ~ 0.2%

    # Desactivado (0): la serie de alta volatilidad sí entra.
    assert targets(high_vol, 0.0) > 0
    # Con umbral 1%: la alta volatilidad queda bloqueada y la baja entra.
    assert targets(high_vol, 0.01) == 0
    assert targets(low_vol, 0.01) > 0
