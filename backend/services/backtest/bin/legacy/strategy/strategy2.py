#!/usr/bin/env python3
"""
Strategy 2 — EMA Cross + ADX / EMA Weakening
============================================

Gestión de salida por defecto: take-profit parcial a 1R (50%) y stop a
breakeven del resto (`partial_tp_enabled`). Sube el win rate y recorta la
racha de pérdidas en whipsaw; puede desactivarse con `partial_tp_enabled=False`.

Archivo único con API reutilizable para backtesting y bots.

Uso:
    python strategy2_refactored.py data.parquet --output-dir backtest_out
    python strategy2_refactored.py data.parquet --no-entry-adx-filter
    python strategy2_refactored.py data.parquet --help

API para bots:
    cfg = Config()
    bars = load_parquet("data.parquet")
    strategy = EMACrossStrategy(cfg)
    signals = strategy.generate_signals(bars)
    # signals contiene señales conocidas al cierre de cada vela.
    # En ejecución real, actuar en la siguiente apertura/iteración.

Dependencias:
    pip install "pandas>=2" pyarrow numpy matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración y modelos
# ---------------------------------------------------------------------------

@dataclass
class Config:
    ema_fast: int = 55
    ema_slow: int = 200
    timeframe: str = "30min"
    allow_long: bool = True
    allow_short: bool = True

    initial_capital: float = 10_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    risk_pct: float = 0.01
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.01

    confirm_bars: int = 4
    entry_adx_enabled: bool = True
    entry_adx_len: int = 14
    entry_adx_min: float = 20.0

    adx_exit_enabled: bool = True
    adx_timeframe: str = "1D"
    adx_dilen: int = 14
    adx_adxlen: int = 14
    adx_key_level: float = 23.0
    adx_exit_pct: float = 100.0

    ema_weakening_exit_enabled: bool = True
    ema_weakening_timeframe: str = "20min"
    ema_weakening_gap_pct: float = 0.001
    ema_weakening_exit_pct: float = 50.0

    # Gestión de salida orientada a reducir la racha de pérdidas en whipsaw:
    # al alcanzar `partial_tp_r` veces el riesgo inicial se asegura beneficio
    # cerrando `partial_tp_pct` de la posición y, opcionalmente, se mueve el
    # stop del resto a breakeven. Sube el win rate real (rompe rachas de
    # stop-loss consecutivos) a cambio de recortar parte del retorno.
    partial_tp_enabled: bool = True
    partial_tp_r: float = 1.0
    partial_tp_pct: float = 50.0
    move_stop_to_breakeven: bool = True

    force_close_at_end: bool = True
    timestamp_is_close: bool = False

    def __post_init__(self):
        if min(self.ema_fast, self.ema_slow) <= 0 or self.ema_fast >= self.ema_slow:
            raise ValueError("Se requiere 0 < ema_fast < ema_slow.")
        for name in ("timeframe", "adx_timeframe"):
            if pd.Timedelta(getattr(self, name)) <= pd.Timedelta(0):
                raise ValueError(f"{name} debe ser positivo.")
        if self.ema_weakening_timeframe and pd.Timedelta(self.ema_weakening_timeframe) <= pd.Timedelta(0):
            raise ValueError("ema_weakening_timeframe debe ser positivo.")
        if self.initial_capital <= 0 or self.max_leverage <= 0:
            raise ValueError("Capital y leverage deben ser positivos.")
        if not 0 < self.risk_pct <= 0.2:
            raise ValueError("risk_pct debe estar en (0, 0.2].")
        if min(self.commission_bps, self.slippage_bps, self.stop_loss_pct) < 0:
            raise ValueError("Comisiones, slippage y stop no pueden ser negativos.")
        if self.confirm_bars < 0:
            raise ValueError("confirm_bars no puede ser negativo.")
        if self.entry_adx_enabled and (self.entry_adx_len <= 0 or self.entry_adx_min < 0):
            raise ValueError("Configuración de ADX de entrada inválida.")
        if self.adx_exit_enabled:
            if min(self.adx_dilen, self.adx_adxlen) <= 0 or self.adx_key_level < 0:
                raise ValueError("Configuración de ADX de salida inválida.")
            if not 0 < self.adx_exit_pct <= 100:
                raise ValueError("adx_exit_pct debe estar en (0, 100].")
        if self.ema_weakening_exit_enabled:
            if self.ema_weakening_gap_pct <= 0 or not 0 < self.ema_weakening_exit_pct <= 100:
                raise ValueError("Configuración de weakening inválida.")
        if self.partial_tp_enabled:
            if self.partial_tp_r <= 0 or not 0 < self.partial_tp_pct <= 100:
                raise ValueError("Configuración de take-profit parcial inválida.")
        if not (self.allow_long or self.allow_short):
            raise ValueError("Debe habilitarse LONG, SHORT o ambos.")


@dataclass
class Position:
    position_id: int
    side: int
    quantity: float
    initial_quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_fee: float
    stop_pct: float
    risk_amount: float
    breakeven: bool = False
    partial_done: bool = False


@dataclass
class Trade:
    trade_id: int
    position_id: int
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl_gross: float
    fees: float
    pnl_net: float
    return_pct: float
    reason: str
    position_risk: float


TRADE_COLUMNS = list(Trade.__dataclass_fields__)
CROSS_REASON = "EMA_CROSS_EXIT"
DEFAULT_OUTPUT_DIR = "backtest_out"


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TRADE_COLUMNS})


# ---------------------------------------------------------------------------
# Datos OHLCV
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza aliases OHLCV y devuelve un DataFrame indexado por timestamp UTC."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    aliases = {
        "timestamp": ("timestamp", "open_time", "datetime", "date", "time", "ts"),
        "open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"),
        "close": ("close", "c"), "volume": ("volume", "vol", "v"),
    }
    rename = {}
    for canonical, choices in aliases.items():
        found = next((c for c in choices if c in df.columns), None)
        if found is not None:
            rename[found] = canonical
    missing = [c for c in ("timestamp", "open", "high", "low", "close") if c not in rename.values()]
    if missing:
        raise ValueError(f"Faltan columnas {missing}; disponibles: {list(df.columns)}")
    out = df.rename(columns=rename).copy()
    if "volume" not in out:
        out["volume"] = 0.0

    ts = out["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        nums = pd.to_numeric(ts, errors="coerce")
        finite = nums.dropna()
        magnitude = float(finite.abs().median()) if len(finite) else 0
        unit = "ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e9 else None
        ts = pd.to_datetime(nums, unit=unit, utc=True, errors="coerce") if unit else pd.to_datetime(ts, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Hay timestamps inválidos.")
    out["timestamp"] = ts
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    return out[["open", "high", "low", "close", "volume"]]


def load_parquet(path: str | Path) -> pd.DataFrame:
    start = time.perf_counter()
    print(f"[LOAD] {path}", flush=True)
    data = normalize_columns(pd.read_parquet(path))
    print(f"[LOAD] {len(data):,} velas en {time.perf_counter()-start:.2f}s | {data.index.min()} -> {data.index.max()}", flush=True)
    return data


def source_minutes(data: pd.DataFrame) -> float:
    delta = data.index.to_series().diff().dropna().dt.total_seconds() / 60
    value = float(delta.median()) if len(delta) else 1.0
    return value if np.isfinite(value) and value > 0 else 1.0


def resample_ohlcv(data: pd.DataFrame, rule: str, source_interval_minutes: Optional[float] = None) -> pd.DataFrame:
    """Resamplea OHLCV y descarta buckets con menos barras que las esperadas."""
    if data.empty:
        return data.copy()
    src = source_interval_minutes or source_minutes(data)
    td = pd.Timedelta(rule)
    target_minutes = td.total_seconds() / 60
    if target_minutes < src:
        raise ValueError(f"No se puede remuestrear de {src:g}m a un timeframe menor ({rule}).")
    grouped = data.resample(rule, label="left", closed="left", origin="start_day")
    out = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = grouped["close"].count()
    expected = max(1, int(round(target_minutes / src)))
    out = out[counts.reindex(out.index, fill_value=0) >= expected]
    return out.dropna(subset=["open", "high", "low", "close"])


def to_timeframe(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if data.empty or pd.Timedelta(minutes=source_minutes(data)) >= pd.Timedelta(cfg.timeframe):
        return data.copy()
    out = resample_ohlcv(data, cfg.timeframe)
    print(f"[RESAMPLE] {source_minutes(data):g}m -> {cfg.timeframe}: {len(out):,} velas", flush=True)
    return out


# ---------------------------------------------------------------------------
# Indicadores puros
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Estrategia: compartida por backtest y bots
# ---------------------------------------------------------------------------

class EMACrossStrategy:
    """Generador de señales sin estado de cartera ni ejecución de órdenes."""

    def __init__(self, config: Config):
        self.cfg = config

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        fast = ema(bars.close, cfg.ema_fast).to_numpy(float)
        slow = ema(bars.close, cfg.ema_slow).to_numpy(float)
        prev_fast, prev_slow = np.r_[np.nan, fast[:-1]], np.r_[np.nan, slow[:-1]]
        up = (fast > slow) & (prev_fast <= prev_slow)
        down = (fast < slow) & (prev_fast >= prev_slow)
        valid = np.isfinite(fast) & np.isfinite(slow) & np.isfinite(prev_fast) & np.isfinite(prev_slow)
        if cfg.confirm_bars:
            up = confirmed_crosses(up, fast > slow, cfg.confirm_bars)
            down = confirmed_crosses(down, fast < slow, cfg.confirm_bars)
        entry_adx = adx(bars, cfg.entry_adx_len, cfg.entry_adx_len)["adx"].to_numpy(float) if cfg.entry_adx_enabled else np.full(len(bars), np.nan)
        if cfg.entry_adx_enabled:
            allowed = np.nan_to_num(entry_adx, nan=0.0) >= cfg.entry_adx_min
            up, down = up & allowed, down & allowed
        up &= valid
        down &= valid
        return pd.DataFrame({
            "ema_fast": fast, "ema_slow": slow, "adx_entry": entry_adx,
            "cross_up": up, "cross_down": down, "valid": valid,
            "target": np.select([up, down], [1, -1], default=0),
        }, index=bars.index)

    def prepare_exits(self, raw: pd.DataFrame, bars: pd.DataFrame) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return build_adx_exit(raw, bars, self.cfg), build_weakening_exit(raw, bars, self.cfg)


# ---------------------------------------------------------------------------
# Motor de ejecución
# ---------------------------------------------------------------------------

class Backtester:
    """Motor de ejecución. Las señales se evalúan al cierre y se ejecutan en la apertura siguiente."""

    def __init__(self, bars: pd.DataFrame, config: Config, signals: Optional[pd.DataFrame] = None,
                 adx_exit: Optional[np.ndarray] = None, weak_exit: Optional[np.ndarray] = None,
                 verbose: bool = False):
        self.bars, self.cfg, self.verbose = bars, config, verbose
        self.signals = signals if signals is not None else EMACrossStrategy(config).generate_signals(bars)
        self.adx_exit = np.asarray(adx_exit, bool) if adx_exit is not None else np.zeros(len(bars), bool)
        self.weak_exit = np.asarray(weak_exit, bool) if weak_exit is not None else np.zeros(len(bars), bool)
        if len(self.adx_exit) != len(bars): self.adx_exit = np.zeros(len(bars), bool)
        if len(self.weak_exit) != len(bars): self.weak_exit = np.zeros(len(bars), bool)
        self.o, self.h, self.l, self.c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.times = bars.index
        self.cash = config.initial_capital
        self.position: Optional[Position] = None
        self.pending_target: Optional[int] = None
        self.pending_exit: Optional[tuple[str, float]] = None
        self.trades: list[Trade] = []
        self.trade_id = self.position_id = 0
        self.eq_time, self.eq_cash, self.eq_value, self.eq_side, self.eq_qty = [], [], [], [], []

    def _price(self, price: float, side: int) -> float:
        slip = self.cfg.slippage_bps / 10_000
        return price * (1 + slip if side == 1 else 1 - slip)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.cfg.commission_bps / 10_000

    def _stop_price(self, side: int, entry: float, pct: float) -> float:
        return entry * (1 - pct if side == 1 else 1 + pct)

    def _risk_per_unit(self, side: int, entry: float, pct: float) -> float:
        stop_exec = self._price(self._stop_price(side, entry, pct), -side)
        return abs(stop_exec - entry) + (entry + abs(stop_exec)) * self.cfg.commission_bps / 10_000

    def open_position(self, side: int, raw_price: float, ts: pd.Timestamp):
        if self.position is not None or self.cash <= 0:
            return
        price = self._price(raw_price, side)
        if not np.isfinite(price) or price <= 0:
            return
        pct = self.cfg.stop_loss_pct
        risk_unit = self._risk_per_unit(side, price, pct) if pct > 0 else 0.0
        by_risk = self.cash * self.cfg.risk_pct / risk_unit if risk_unit > 0 else np.inf
        by_lev = self.cash * self.cfg.max_leverage / price
        qty = min(by_risk, by_lev)
        if not np.isfinite(qty) or qty <= 0:
            return
        fee = self._fee(price * qty)
        self.cash -= fee
        self.position_id += 1
        self.position = Position(self.position_id, side, qty, qty, price, ts, fee, pct, qty * risk_unit)

    def close_position(self, percent: float, raw_price: float, ts: pd.Timestamp, reason: str):
        pos = self.position
        if pos is None:
            return
        if not 0 < percent <= 100:
            raise ValueError("El porcentaje de cierre debe estar en (0, 100].")
        qty = pos.quantity * percent / 100
        side = pos.side
        price = self._price(raw_price, -side)
        gross = (price - pos.entry_price) * qty * side
        exit_fee = self._fee(price * qty)
        # La comisión de entrada se distribuye proporcionalmente a la cantidad original restante.
        entry_fee = pos.entry_fee * (qty / pos.quantity)
        fees = entry_fee + exit_fee
        net = gross - fees
        self.cash += gross - exit_fee
        self.trade_id += 1
        self.trades.append(Trade(self.trade_id, pos.position_id, "LONG" if side == 1 else "SHORT",
            pos.entry_time, ts, pos.entry_price, price, qty, gross, fees, net,
            net / (pos.entry_price * qty) * 100 if pos.entry_price * qty else 0.0,
            reason, pos.risk_amount))
        remaining = pos.quantity - qty
        if percent >= 100 or remaining <= max(1e-12, pos.initial_quantity * 1e-12):
            self.position = None
        else:
            pos.quantity = remaining
            pos.entry_fee -= entry_fee

    def _risk_unit(self, pos: Position) -> float:
        return pos.risk_amount / pos.initial_quantity if pos.initial_quantity else 0.0

    def _manage(self, i: int):
        """Take-profit parcial y stop a breakeven para recortar rachas de pérdidas."""
        cfg, pos = self.cfg, self.position
        if pos is None or not cfg.partial_tp_enabled or pos.partial_done:
            return
        risk_unit = self._risk_unit(pos)
        if risk_unit <= 0:
            return
        trigger = pos.entry_price + pos.side * cfg.partial_tp_r * risk_unit
        hit = self.h[i] >= trigger if pos.side == 1 else self.l[i] <= trigger
        if not hit:
            return
        fill = max(trigger, self.o[i]) if pos.side == 1 else min(trigger, self.o[i])
        self.close_position(cfg.partial_tp_pct, fill, self.times[i], f"PARTIAL_TP_{cfg.partial_tp_r:g}R")
        pos = self.position
        if pos is not None:
            pos.partial_done = True
            if cfg.move_stop_to_breakeven:
                pos.breakeven = True

    def _stop(self, i: int) -> bool:
        pos = self.position
        if pos is None or pos.stop_pct <= 0: return False
        stop = pos.entry_price if pos.breakeven else self._stop_price(pos.side, pos.entry_price, pos.stop_pct)
        hit = self.l[i] <= stop if pos.side == 1 else self.h[i] >= stop
        if not hit: return False
        fill = min(self.o[i], stop) if pos.side == 1 else max(self.o[i], stop)
        reason = "BREAKEVEN" if pos.breakeven else f"STOP_LOSS_{pos.stop_pct*100:.2f}%"
        self.close_position(100, fill, self.times[i], reason)
        if self.verbose: print(f"[STOP] {self.times[i]} fill={fill:.8f}", flush=True)
        return True

    def _execute_target(self, i: int, target: int):
        if self.position is not None and self.position.side != target:
            self.close_position(100, self.o[i], self.times[i], CROSS_REASON)
        if self.position is None:
            self.open_position(target, self.o[i], self.times[i])

    def _mark(self, ts: pd.Timestamp, close: float):
        pos = self.position
        value = self.cash
        if pos is not None:
            liq = self._price(close, -pos.side)
            value += (liq - pos.entry_price) * pos.quantity * pos.side - self._fee(liq * pos.quantity)
        self.eq_time.append(ts); self.eq_cash.append(self.cash); self.eq_value.append(value)
        self.eq_side.append("FLAT" if pos is None else ("LONG" if pos.side == 1 else "SHORT"))
        self.eq_qty.append(0.0 if pos is None else pos.quantity)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        n, cfg = len(self.bars), self.cfg
        if not n:
            return empty_trades(), pd.DataFrame(columns=["timestamp", "cash", "equity", "position_side", "position_quantity"])
        adx_tf_min = int(pd.Timedelta(cfg.adx_timeframe).total_seconds() / 60)
        adx_base = f"ADX_{adx_tf_min}M_REVERSAL"
        adx_reason = adx_base if cfg.adx_exit_pct >= 100 else f"{adx_base}_{cfg.adx_exit_pct:g}%"
        weak_tf = cfg.ema_weakening_timeframe or cfg.timeframe
        weak_base = f"EMA_WEAKENING_{weak_tf.upper()}"
        weak_reason = weak_base if cfg.ema_weakening_exit_pct >= 100 else f"{weak_base}_{cfg.ema_weakening_exit_pct:g}%"
        started = time.perf_counter()
        print(f"[BACKTEST] {n:,} velas {cfg.timeframe}", flush=True)
        for i in range(n):
            if self.pending_target is not None:
                self._execute_target(i, self.pending_target)
                self.pending_target = None
                self.pending_exit = None
            elif self.pending_exit is not None:
                reason, pct = self.pending_exit
                self.close_position(pct, self.o[i], self.times[i], reason)
                self.pending_exit = None

            stop_hit = self._stop(i)
            if not stop_hit:
                self._manage(i)
            if not stop_hit and bool(self.signals["valid"].iloc[i]):
                target = int(self.signals["target"].iloc[i])
                if target == 1 and not cfg.allow_long: target = 0
                if target == -1 and not cfg.allow_short: target = 0
                current = self.position.side if self.position else 0
                if target and target != current:
                    self.pending_target = target
                elif self.position is not None and self.weak_exit[i]:
                    self.pending_exit = (weak_reason, cfg.ema_weakening_exit_pct)
                elif self.position is not None and self.adx_exit[i]:
                    self.pending_exit = (adx_reason, cfg.adx_exit_pct)

            self._mark(self.times[i] + pd.Timedelta(cfg.timeframe), self.c[i])
            if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
                print(f"[BACKTEST] {(i+1)/n:6.1%} | {i+1:,}/{n:,} velas | trades={len(self.trades)}", flush=True)

        if self.position is not None and cfg.force_close_at_end:
            ts = self.times[-1] + pd.Timedelta(cfg.timeframe)
            self.close_position(100, self.c[-1], ts, "END_OF_DATA")
            self._mark(ts, self.c[-1])
        trades = pd.DataFrame([asdict(t) for t in self.trades]) if self.trades else empty_trades()
        equity = pd.DataFrame({"timestamp": self.eq_time, "cash": self.eq_cash, "equity": self.eq_value,
                               "position_side": self.eq_side, "position_quantity": self.eq_qty})
        print(f"[BACKTEST] Finalizado en {time.perf_counter()-started:.2f}s | tramos={len(trades):,}", flush=True)
        return trades, equity


# ---------------------------------------------------------------------------
# Métricas, reportes y archivos
# ---------------------------------------------------------------------------

def positions_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    cols = ["position_id", "side", "entry_time", "exit_time", "pnl_net", "fees", "risk", "legs", "reason", "r_multiple"]
    if trades.empty: return pd.DataFrame(columns=cols)
    pos = trades.groupby("position_id", sort=True).agg(
        side=("side", "first"), entry_time=("entry_time", "first"), exit_time=("exit_time", "last"),
        pnl_net=("pnl_net", "sum"), fees=("fees", "sum"), risk=("position_risk", "first"),
        legs=("trade_id", "count"), reason=("reason", "last")).reset_index()
    pos["r_multiple"] = np.where(pos.risk > 0, pos.pnl_net / pos.risk, np.nan)
    return pos[cols]


def position_stats(pos: pd.DataFrame) -> dict:
    if pos.empty:
        return {"positions": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "profit_factor": None,
                "net_pnl": 0.0, "avg_pnl": 0.0, "expectancy_r": None, "max_consecutive_losses": 0}
    pnl = pos.pnl_net.astype(float)
    gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    streak = best = 0
    for value in pnl:
        streak = streak + 1 if value < 0 else 0
        best = max(best, streak)
    r = pos.r_multiple.dropna()
    return {"positions": int(len(pnl)), "wins": int((pnl > 0).sum()), "losses": int((pnl < 0).sum()),
            "win_rate_pct": float((pnl > 0).mean() * 100), "profit_factor": gp / gl if gl else None,
            "net_pnl": float(pnl.sum()), "avg_pnl": float(pnl.mean()),
            "expectancy_r": float(r.mean()) if len(r) else None, "max_consecutive_losses": int(best)}


def equity_curve_from_trades(trades: pd.DataFrame, capital: float) -> pd.DataFrame:
    cols = ["timestamp", "trade_id", "pnl_net", "cum_pnl", "equity"]
    if trades.empty: return pd.DataFrame(columns=cols)
    out = trades.sort_values(["exit_time", "trade_id"], kind="stable").copy()
    out["cum_pnl"] = out.pnl_net.astype(float).cumsum()
    out["equity"] = capital + out.cum_pnl
    return out[["exit_time", "trade_id", "pnl_net", "cum_pnl", "equity"]].rename(columns={"exit_time": "timestamp"})[cols].reset_index(drop=True)


def calculate_metrics(trades: pd.DataFrame, equity: pd.DataFrame, capital: float) -> dict:
    stats = position_stats(positions_from_trades(trades))
    if equity.empty:
        final, max_dd, min_eq, min_pct = capital, 0.0, capital, 0.0
        peak_eq = trough_eq = capital
        peak_pct = trough_pct = sharpe = sortino = 0.0
    else:
        eq = equity.equity.astype(float)
        final = float(eq.iloc[-1])
        peak = eq.cummax()
        dd = (eq / peak - 1) * 100
        max_dd = float(dd.min())
        min_eq = float(eq.min())
        min_pct = float((eq / capital - 1).min() * 100)
        idx = int(np.argmin(dd.to_numpy()))
        peak_eq, trough_eq = float(peak.iloc[idx]), float(eq.iloc[idx])
        peak_pct, trough_pct = (peak_eq / capital - 1) * 100, (trough_eq / capital - 1) * 100
        daily = equity.set_index("timestamp").equity.resample("1D").last().dropna()
        rets = daily.pct_change().dropna()
        vol = rets.std(ddof=1) if len(rets) > 1 else 0
        sharpe = float(rets.mean() / vol * math.sqrt(365)) if len(rets) > 1 and np.isfinite(vol) and vol > 0 else 0.0
        downside = math.sqrt(float(np.mean(np.minimum(rets.to_numpy(float), 0) ** 2))) if len(rets) else 0
        sortino = float(rets.mean() / downside * math.sqrt(365)) if downside > 0 else 0.0
    net = final - capital
    return {"initial_capital": capital, "final_equity": final, "net_pnl": net,
            "return_pct": net / capital * 100, "max_drawdown_pct": max_dd,
            "max_dd_peak_equity": peak_eq, "max_dd_trough_equity": trough_eq,
            "max_dd_peak_vs_capital_pct": peak_pct, "max_dd_trough_vs_capital_pct": trough_pct,
            "min_equity": min_eq, "min_equity_vs_capital_pct": min_pct,
            "return_over_maxdd": (net / capital * 100) / abs(max_dd) if max_dd < 0 else None,
            "sharpe_daily_365": sharpe, "sortino_daily_365": sortino, "legs": int(len(trades)), **stats}


def jsonable(obj):
    if isinstance(obj, dict): return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)): return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    if isinstance(obj, Path): return str(obj)
    return obj


def dataset_quality(data: pd.DataFrame) -> dict:
    if len(data) < 2: return {"bars": len(data), "gaps_gt_1bar": 0, "max_gap_minutes": 0.0}
    delta = data.index.to_series().diff().dropna().dt.total_seconds() / 60
    median = float(delta.median())
    gaps = delta[delta > median]
    return {"bars": int(len(data)), "gaps_gt_1bar": int(len(gaps)),
            "max_gap_minutes": float(gaps.max()) if len(gaps) else median}


def build_report(cfg: Config, data: pd.DataFrame, trades: pd.DataFrame, metrics: dict, source: str) -> dict:
    sides = {}
    for side in ("LONG", "SHORT"):
        subset = trades[trades.side == side] if not trades.empty else trades
        sides[side] = position_stats(positions_from_trades(subset))
    reasons = {}
    if not trades.empty:
        grouped = trades.groupby("reason", sort=False).agg(legs=("trade_id", "count"), positions=("position_id", "nunique"), pnl_net=("pnl_net", "sum"))
        reasons = {str(k): {"legs": int(v.legs), "positions": int(v.positions), "pnl_net": float(v.pnl_net)} for k, v in grouped.iterrows()}
    return jsonable({"strategy": "strategy2_ema_cross", "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"source": source, "bars": len(data), "start": data.index[0].isoformat(), "end": data.index[-1].isoformat(), "quality": dataset_quality(data)},
        "config": asdict(cfg), "metrics": metrics, "by_side": sides, "exit_reasons": reasons})


def write_outputs(out_dir: str | Path, trades: pd.DataFrame, equity: pd.DataFrame, curve: pd.DataFrame,
                  report: dict, write_equity_csv: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    positions_from_trades(trades).to_csv(out / "positions.csv", index=False)
    curve.to_csv(out / "equity_curve.csv", index=False)
    if write_equity_csv: equity.to_csv(out / "equity.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metrics_path = out / "metrics.json"
    current = jsonable(report["metrics"])
    previous = None
    if metrics_path.exists():
        try: previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): pass
    delta = {}
    if previous is not None:
        for key in sorted(set(previous) | set(current)):
            a, b = previous.get(key), current.get(key)
            numeric = isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool)
            diff = b - a if numeric else None
            delta[key] = {"previous": a, "current": b, "delta": diff,
                          "delta_pct": diff / abs(a) * 100 if numeric and a else None,
                          "changed": diff != 0 if numeric else a != b}
    (out / "metrics_delta.json").write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
        "previous_available": previous is not None, "delta": delta}, indent=2, allow_nan=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(current, indent=2, allow_nan=False), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(report["config"], indent=2, allow_nan=False), encoding="utf-8")
    print(f"[OUT] Archivos guardados en {out.resolve()}", flush=True)


def print_report(metrics: dict, cfg: Config, data: pd.DataFrame):
    print("\n" + "=" * 68 + "\nSTRATEGY 2 — EMA CROSS\n" + "=" * 68)
    print(f"Periodo       : {data.index[0]} -> {data.index[-1]}")
    print(f"Timeframe     : {cfg.timeframe} | EMA{cfg.ema_fast}/{cfg.ema_slow}")
    print(f"Capital       : {metrics['initial_capital']:,.2f} -> {metrics['final_equity']:,.2f}")
    print(f"PnL neto      : {metrics['net_pnl']:,.2f} ({metrics['return_pct']:.2f}%)")
    print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe/Sortino: {metrics['sharpe_daily_365']:.2f} / {metrics['sortino_daily_365']:.2f}")
    print(f"Posiciones    : {metrics['positions']} | tramos: {metrics['legs']}")
    print(f"Win rate      : {metrics['win_rate_pct']:.2f}% | PF: {metrics['profit_factor']}")
    print(f"Expectancy R  : {metrics['expectancy_r']} | racha pérdidas: {metrics['max_consecutive_losses']}")
    print(f"Costes        : comisión {cfg.commission_bps} bps | slippage {cfg.slippage_bps} bps")
    print("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    d = Config()
    p = argparse.ArgumentParser(description="Backtest EMA cross con salidas ADX y EMA weakening.")
    p.add_argument("parquet", help="Dataset OHLCV en Parquet")
    p.add_argument("--capital", type=float, default=d.initial_capital)
    p.add_argument("--ema-fast", type=int, default=d.ema_fast)
    p.add_argument("--ema-slow", type=int, default=d.ema_slow)
    p.add_argument("--timeframe", default=d.timeframe)
    p.add_argument("--commission-bps", type=float, default=d.commission_bps)
    p.add_argument("--slippage-bps", type=float, default=d.slippage_bps)
    p.add_argument("--risk-pct", type=float, default=d.risk_pct)
    p.add_argument("--max-leverage", type=float, default=d.max_leverage)
    p.add_argument("--stop-loss-pct", type=float, default=d.stop_loss_pct)
    p.add_argument("--confirm-bars", type=int, default=d.confirm_bars)
    p.add_argument("--entry-adx-filter", action=argparse.BooleanOptionalAction, default=d.entry_adx_enabled)
    p.add_argument("--entry-adx-len", type=int, default=d.entry_adx_len)
    p.add_argument("--entry-adx-min", type=float, default=d.entry_adx_min)
    p.add_argument("--adx-exit", action=argparse.BooleanOptionalAction, default=d.adx_exit_enabled)
    p.add_argument("--adx-timeframe", default=d.adx_timeframe)
    p.add_argument("--adx-dilen", type=int, default=d.adx_dilen)
    p.add_argument("--adx-adxlen", type=int, default=d.adx_adxlen)
    p.add_argument("--adx-key-level", type=float, default=d.adx_key_level)
    p.add_argument("--adx-exit-pct", type=float, default=d.adx_exit_pct)
    p.add_argument("--ema-weakening-exit", action=argparse.BooleanOptionalAction, default=d.ema_weakening_exit_enabled)
    p.add_argument("--ema-weakening-timeframe", default=d.ema_weakening_timeframe)
    p.add_argument("--ema-weakening-gap-pct", type=float, default=d.ema_weakening_gap_pct)
    p.add_argument("--ema-weakening-exit-pct", type=float, default=d.ema_weakening_exit_pct)
    p.add_argument("--partial-tp", action=argparse.BooleanOptionalAction, default=d.partial_tp_enabled)
    p.add_argument("--partial-tp-r", type=float, default=d.partial_tp_r)
    p.add_argument("--partial-tp-pct", type=float, default=d.partial_tp_pct)
    p.add_argument("--breakeven-stop", action=argparse.BooleanOptionalAction, default=d.move_stop_to_breakeven)
    p.add_argument("--no-long", action="store_true")
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--no-force-close", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-output", action="store_true")
    p.add_argument("--equity-csv", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot", default=None)
    p.add_argument("--plot-dpi", type=int, default=150)
    return p.parse_args()


def config_from_args(a) -> Config:
    return Config(ema_fast=a.ema_fast, ema_slow=a.ema_slow, timeframe=a.timeframe,
        allow_long=not a.no_long, allow_short=not a.no_short, initial_capital=a.capital,
        commission_bps=a.commission_bps, slippage_bps=a.slippage_bps, risk_pct=a.risk_pct,
        max_leverage=a.max_leverage, stop_loss_pct=a.stop_loss_pct, confirm_bars=a.confirm_bars,
        entry_adx_enabled=a.entry_adx_filter, entry_adx_len=a.entry_adx_len, entry_adx_min=a.entry_adx_min,
        adx_exit_enabled=a.adx_exit, adx_timeframe=a.adx_timeframe, adx_dilen=a.adx_dilen,
        adx_adxlen=a.adx_adxlen, adx_key_level=a.adx_key_level, adx_exit_pct=a.adx_exit_pct,
        ema_weakening_exit_enabled=a.ema_weakening_exit, ema_weakening_timeframe=a.ema_weakening_timeframe,
        ema_weakening_gap_pct=a.ema_weakening_gap_pct, ema_weakening_exit_pct=a.ema_weakening_exit_pct,
        partial_tp_enabled=a.partial_tp, partial_tp_r=a.partial_tp_r, partial_tp_pct=a.partial_tp_pct,
        move_stop_to_breakeven=a.breakeven_stop,
        force_close_at_end=not a.no_force_close)


def plot_equity(curve: pd.DataFrame, path: str | Path, capital: float, dpi: int = 150):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib no disponible; se omite el gráfico.")
        return
    if curve.empty:
        print("[PLOT] Sin cierres; no se genera gráfico.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = pd.to_datetime(curve.timestamp, utc=True)
    y = curve.equity.to_numpy(float)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, y, linewidth=1.4, label="Equity")
    ax.axhline(capital, linestyle="--", linewidth=0.9, label=f"Capital inicial ({capital:,.2f})")
    ax.set(title="Strategy 2 — Equity curve", xlabel="Fecha (UTC)", ylabel="Equity")
    ax.grid(True, alpha=.3); ax.legend()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def main():
    args = parse_args()
    cfg = config_from_args(args)
    raw = load_parquet(args.parquet)
    bars = to_timeframe(raw, cfg)
    if len(bars) < cfg.ema_slow + 2:
        raise ValueError(f"Dataset insuficiente para EMA{cfg.ema_slow}: {len(bars)} velas.")
    strategy = EMACrossStrategy(cfg)
    signals = strategy.generate_signals(bars)
    adx_exit, weak_exit = strategy.prepare_exits(raw, bars)
    if cfg.adx_exit_enabled:
        print(f"[ADX] {'disponible' if adx_exit is not None else 'sin datos suficientes/timeframe incompatible'}", flush=True)
    if cfg.ema_weakening_exit_enabled:
        print(f"[WEAKENING] {'disponible' if weak_exit is not None else 'sin datos suficientes'}", flush=True)
    engine = Backtester(bars, cfg, signals, adx_exit, weak_exit, args.verbose)
    trades, equity = engine.run()
    metrics = calculate_metrics(trades, equity, cfg.initial_capital)
    print_report(metrics, cfg, bars)
    curve = equity_curve_from_trades(trades, cfg.initial_capital)
    if not args.no_output:
        report = build_report(cfg, bars, trades, metrics, args.parquet)
        write_outputs(args.output_dir, trades, equity, curve, report, args.equity_csv)
    if not args.no_plot:
        plot_equity(curve, args.plot or str(Path(args.output_dir) / "equity_curve.png"), cfg.initial_capital, args.plot_dpi)


if __name__ == "__main__":
    main()