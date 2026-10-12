from __future__ import annotations
from typing import Optional
from strategy.utils import source_minutes, resample_ohlcv
from strategy.config import Config
import numpy as np
import pandas as pd

def ema(values: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("El periodo EMA debe ser positivo.")
    return values.ewm(alpha=2 / (period + 1), adjust=False).mean()


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev = np.r_[np.nan, close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    if len(tr):
        tr[0] = high[0] - low[0]
    return tr


def adx(bars: pd.DataFrame, dilen: int = 14, adxlen: int = 14) -> pd.DataFrame:
    """ADX/DI con suavizado Wilder; salida alineada con el índice de entrada."""
    if dilen <= 0 or adxlen <= 0:
        raise ValueError("Las longitudes ADX deben ser positivas.")
    n = len(bars)
    out = pd.DataFrame(np.nan, index=bars.index, columns=["adx", "plus_di", "minus_di"])
    if not n:
        return out
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    up = np.r_[0.0, np.diff(h)]
    down = np.r_[0.0, -np.diff(l)]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    smooth = lambda x, a: pd.Series(x).ewm(alpha=a, adjust=False).mean().to_numpy()
    tr_rma = smooth(true_range(h, l, c), 1 / dilen)
    p_rma, m_rma = smooth(plus_dm, 1 / dilen), smooth(minus_dm, 1 / dilen)
    safe = np.where(tr_rma != 0, tr_rma, 1.0)
    pdi = np.where(tr_rma != 0, 100 * p_rma / safe, 0.0)
    mdi = np.where(tr_rma != 0, 100 * m_rma / safe, 0.0)
    denom = np.where(pdi + mdi != 0, pdi + mdi, 1.0)
    dx = 100 * np.abs(pdi - mdi) / denom
    av = smooth(dx, 1 / adxlen)
    first = dilen + adxlen - 2
    if first < n:
        out.iloc[first:, :] = np.column_stack([av[first:], pdi[first:], mdi[first:]])
    return out


def confirmed_crosses(cross: np.ndarray, side_ok: np.ndarray, bars: int) -> np.ndarray:
    cross, side_ok = np.asarray(cross, bool), np.asarray(side_ok, bool)
    if bars <= 0:
        return cross.copy()
    out = np.zeros(len(cross), dtype=bool)
    for idx in np.flatnonzero(cross):
        end = idx + bars
        if end < len(cross) and side_ok[idx:end + 1].all():
            out[end] = True
    return out


def adx_reversal_flags(values: np.ndarray, key_level: float) -> np.ndarray:
    values = np.asarray(values, float)
    flags = np.zeros(len(values), dtype=bool)
    if len(values) >= 3:
        flags[2:] = (values[2:] < values[1:-1]) & (values[1:-1] > values[:-2]) & (values[1:-1] > key_level)
    return flags


def ema_weakening_flags(fast: np.ndarray, slow: np.ndarray, gap_pct: float) -> np.ndarray:
    fast, slow = np.asarray(fast, float), np.asarray(slow, float)
    flags = np.zeros(len(fast), dtype=bool)
    if len(fast) < 2 or gap_pct <= 0:
        return flags
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.abs(fast - slow) / slow
    flags[1:] = (gap[:-1] > gap_pct) & (gap[1:] <= gap_pct)
    return flags


def align_closed_flags(source_index: pd.DatetimeIndex, flags: np.ndarray,
                       target_index: pd.DatetimeIndex, source_tf: str,
                       target_tf: str) -> np.ndarray:
    """Para cada cierre del timeframe destino, toma la última barra fuente ya cerrada."""
    if not len(target_index) or not len(source_index):
        return np.zeros(len(target_index), dtype=bool)
    source_end = (source_index + pd.Timedelta(source_tf)).asi8
    target_end = (target_index + pd.Timedelta(target_tf)).asi8
    loc = np.searchsorted(source_end, target_end, side="right") - 1
    out = np.zeros(len(target_index), dtype=bool)
    valid = (loc >= 0) & (loc < len(flags))
    out[valid] = np.asarray(flags, bool)[loc[valid]]
    return out


def build_adx_exit(raw: pd.DataFrame, bars: pd.DataFrame, cfg: Config) -> Optional[np.ndarray]:
    if not cfg.adx_exit_enabled or bars.empty:
        return None
    if pd.Timedelta(minutes=source_minutes(raw)) >= pd.Timedelta(cfg.adx_timeframe):
        return None
    low = resample_ohlcv(raw, cfg.adx_timeframe)
    if len(low) < cfg.adx_dilen + cfg.adx_adxlen + 2:
        return None
    flags = adx_reversal_flags(adx(low, cfg.adx_dilen, cfg.adx_adxlen)["adx"].to_numpy(), cfg.adx_key_level)
    return align_closed_flags(low.index, flags, bars.index, cfg.adx_timeframe, cfg.timeframe)


def build_weakening_exit(raw: pd.DataFrame, bars: pd.DataFrame, cfg: Config) -> Optional[np.ndarray]:
    if not cfg.ema_weakening_exit_enabled or bars.empty:
        return None
    tf = cfg.ema_weakening_timeframe or cfg.timeframe
    if pd.Timedelta(tf) == pd.Timedelta(cfg.timeframe) or pd.Timedelta(minutes=source_minutes(raw)) >= pd.Timedelta(tf):
        source_bars, source_tf = bars, cfg.timeframe
    else:
        source_bars, source_tf = resample_ohlcv(raw, tf), tf
    if len(source_bars) < cfg.ema_slow + 2:
        return None
    flags = ema_weakening_flags(ema(source_bars.close, cfg.ema_fast).to_numpy(),
                                ema(source_bars.close, cfg.ema_slow).to_numpy(),
                                cfg.ema_weakening_gap_pct)
    if source_bars.index.equals(bars.index):
        return flags
    return align_closed_flags(source_bars.index, flags, bars.index, source_tf, cfg.timeframe)