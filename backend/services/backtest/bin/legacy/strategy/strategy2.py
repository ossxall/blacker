#!/usr/bin/env python3

"""
Backtesting Strategy2 — EMA CROSS 55/200 (30m)
==============================================

Estrategia básica de cruce de medias sobre velas de 30 minutos.

Reglas:

    LONG  -> EMA55 cruza AL ALZA a EMA200 (golden cross).
    SHORT -> EMA55 cruza A LA BAJA a EMA200 (death cross).
    SALIDA -> cruce contrario (la posición se revierte).
    SALIDA ADICIONAL -> reversión del ADX calculado en 5m: cuando el
        ADX deja de subir tras un pico por encima de `key_level`, se
        cierra la posición (total o parcial según `adx_exit_pct`).
        Funciona igual para LONG y SHORT.
    SALIDA PARCIAL RSI (solo SHORT) -> en el timeframe de la
        estrategia (30m), cuando el RSI cruza A LA BAJA el nivel
        `rsi_exit_level` (32) se cierra el % `rsi_exit_pct` del
        SHORT abierto. No aplica a LONG.

La entrada sigue siendo SOLO el cruce de EMAs. No hay ATR.

Convención anti-lookahead:

    Las señales se calculan al cierre de la vela 30m y se ejecutan
    en la apertura de la siguiente vela 30m.

Entrada:

    OHLCV en .parquet. Si el dataset es de 1m se remuestrea a 30m;
    si ya está en 30m se usa tal cual.

Dependencias:

    pip install "pandas>=2" pyarrow numpy matplotlib

Ejemplo:

    python strategy2.py data.parquet \
        --capital 100000 \
        --commission-bps 2 \
        --slippage-bps 1 \
        --output-dir ./backtest_out
"""

from __future__ import annotations

import argparse
import json
import math
import time

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Cruce de EMAs.
    ema_fast: int = 55
    ema_slow: int = 200

    # Timeframe de la estrategia.
    timeframe: str = "30min"

    # Lados habilitados.
    allow_long: bool = True
    allow_short: bool = True

    initial_capital: float = 10_000.0

    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    risk_pct: float = 0.01
    max_leverage: float = 1.0

    # Stop opcional. 0 => deshabilitado (solo EMACROSS).
    stop_loss_pct: float = 0.03

    # Salida por reversión del ADX en un timeframe menor (ej. 5m).
    # adx_exit_pct = 100 -> cierre total; < 100 -> salida parcial.
    adx_exit_enabled: bool = True
    adx_timeframe: str = "5min"
    adx_dilen: int = 14
    adx_adxlen: int = 14
    adx_key_level: float = 23.0
    adx_exit_pct: float = 50.0

    # Salida parcial por RSI en el timeframe de la estrategia (30m).
    # Solo SHORT: cruce a la baja de `rsi_exit_level`.
    rsi_exit_enabled: bool = True
    rsi_len: int = 14
    rsi_exit_level: float = 32.0
    rsi_exit_pct: float = 50.0

    force_close_at_end: bool = True

    timestamp_is_close: bool = False

    def __post_init__(self):

        if self.ema_fast <= 0 or self.ema_slow <= 0:
            raise ValueError("Los periodos EMA deben ser > 0.")

        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast debe ser menor que ema_slow.")

        if pd.Timedelta(self.timeframe) <= pd.Timedelta(0):
            raise ValueError("timeframe debe ser > 0.")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital debe ser > 0.")

        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError(
                "commission_bps y slippage_bps no pueden ser negativos."
            )

        if not (0.0 < self.risk_pct <= 0.2):
            raise ValueError(
                "risk_pct debe estar entre 0 (excl.) y 0.2."
            )

        if self.max_leverage <= 0:
            raise ValueError("max_leverage debe ser > 0.")

        if self.stop_loss_pct < 0:
            raise ValueError("stop_loss_pct no puede ser negativo.")

        if self.adx_exit_enabled:

            if self.adx_dilen <= 0 or self.adx_adxlen <= 0:
                raise ValueError(
                    "adx_dilen y adx_adxlen deben ser > 0."
                )

            if pd.Timedelta(self.adx_timeframe) <= pd.Timedelta(0):
                raise ValueError("adx_timeframe debe ser > 0.")

            if self.adx_key_level < 0:
                raise ValueError("adx_key_level no puede ser negativo.")

            if not (0.0 < self.adx_exit_pct <= 100.0):
                raise ValueError(
                    "adx_exit_pct debe estar entre 0 (excl.) y 100."
                )

        if self.rsi_exit_enabled:

            if self.rsi_len <= 0:
                raise ValueError("rsi_len debe ser > 0.")

            if not (0.0 < self.rsi_exit_level < 100.0):
                raise ValueError(
                    "rsi_exit_level debe estar entre 0 y 100."
                )

            if not (0.0 < self.rsi_exit_pct <= 100.0):
                raise ValueError(
                    "rsi_exit_pct debe estar entre 0 (excl.) y 100."
                )

        if not self.allow_long and not self.allow_short:
            raise ValueError(
                "Debe habilitarse al menos un lado (long o short)."
            )


DEFAULT_OUTPUT_DIR = "backtest_out"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    position_id: int
    side: int                  # +1 LONG, -1 SHORT
    quantity: float
    initial_quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_fee: float
    stop_pct: float
    risk_amount: float
    stop_price_estimate: float


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


def empty_trades() -> pd.DataFrame:
    """DataFrame de trades sin filas pero con el esquema completo."""
    return pd.DataFrame({
        c: pd.Series(dtype="object")
        for c in TRADE_COLUMNS
    })


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y localiza OHLCV."""

    df = df.rename(
        columns={
            c: str(c).strip().lower()
            for c in df.columns
        }
    )

    aliases = {
        "timestamp": [
            "timestamp",
            "open_time",
            "datetime",
            "date",
            "time",
            "ts",
        ],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v"],
    }

    result = {}

    for canonical, candidates in aliases.items():
        found = next(
            (c for c in candidates if c in df.columns),
            None,
        )

        if found is not None:
            result[canonical] = found

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        x for x in required
        if x not in result
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    out = df.rename(
        columns={
            v: k
            for k, v in result.items()
        }
    ).copy()

    if "volume" not in out.columns:
        out["volume"] = 0.0

    raw_ts = out["timestamp"]

    # Timestamp en UTC.
    if pd.api.types.is_numeric_dtype(raw_ts):

        numeric_ts = pd.to_numeric(raw_ts, errors="coerce")

        finite = numeric_ts.dropna()

        unit = None

        if not finite.empty:

            magnitude = float(finite.abs().median())

            if magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            elif magnitude >= 1e9:
                unit = "s"

        if unit is not None:

            ts = pd.to_datetime(
                numeric_ts,
                unit=unit,
                utc=True,
                errors="coerce",
            )

        else:

            ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")

    else:

        ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")

    if ts.isna().any():
        raise ValueError("Hay timestamps inválidos en el dataset.")

    out["timestamp"] = ts

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    out = out.sort_values("timestamp")

    out = out.drop_duplicates("timestamp", keep="last")

    out = out.set_index("timestamp")

    return out[["open", "high", "low", "close", "volume"]]


def load_parquet(path: str | Path) -> pd.DataFrame:

    print(f"[LOAD] Leyendo Parquet: {path}", flush=True)

    t0 = time.perf_counter()

    df = pd.read_parquet(path)

    print(
        f"[LOAD] {len(df):,} filas en "
        f"{time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    out = normalize_columns(df)

    print(
        f"[LOAD] {len(out):,} velas | "
        f"{out.index.min()} -> {out.index.max()}",
        flush=True,
    )

    return out


def resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
    source_minutes: float,
) -> pd.DataFrame:

    """Remuestrea a `rule` descartando velas incompletas."""

    agg = df.resample(
        rule,
        label="left",
        closed="left",
        origin="start_day",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    counts = df["close"].resample(
        rule,
        label="left",
        closed="left",
        origin="start_day",
    ).count()

    target_minutes = (
        pd.Timedelta(rule) / pd.Timedelta(minutes=1)
    )

    expected = max(
        1,
        int(round(target_minutes / source_minutes)),
    )

    complete = (
        counts
        .reindex(agg.index, fill_value=0)
        .to_numpy()
        >= expected
    )

    agg = agg[complete]

    return agg.dropna(
        subset=["open", "high", "low", "close"]
    )


def to_timeframe(
    data: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:

    """Lleva el dataset al timeframe de la estrategia."""

    if len(data) < 2:
        return data

    source_minutes = float(
        data.index
        .to_series()
        .diff()
        .dropna()
        .dt.total_seconds()
        .median()
        / 60.0
    )

    if not np.isfinite(source_minutes) or source_minutes <= 0:
        source_minutes = 1.0

    target = pd.Timedelta(cfg.timeframe)

    if pd.Timedelta(minutes=source_minutes) >= target:
        return data

    out = resample_ohlcv(
        data,
        cfg.timeframe,
        source_minutes,
    )

    print(
        f"[RESAMPLE] {source_minutes:.0f}m -> "
        f"{cfg.timeframe} | {len(out):,} velas",
        flush=True,
    )

    return out


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def ema_closed(
    values: pd.Series,
    period: int,
) -> pd.Series:

    """EMA con seed = primer valor."""

    if period <= 0:
        raise ValueError("EMA period must be > 0")

    return values.ewm(
        alpha=2.0 / (period + 1.0),
        adjust=False,
    ).mean()


def rsi_vec(
    close: pd.Series,
    length: int = 14,
) -> pd.Series:

    """RSI de Wilder (smoothing alpha = 1/length)."""

    if length <= 0:
        raise ValueError("RSI length must be > 0")

    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()

    zero_loss = avg_loss.to_numpy() == 0.0
    zero_gain = avg_gain.to_numpy() == 0.0

    safe_loss = np.where(zero_loss, 1.0, avg_loss.to_numpy())

    with np.errstate(invalid="ignore"):
        rs = avg_gain.to_numpy() / safe_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    rsi = np.where(zero_loss & ~zero_gain, 100.0, rsi)
    rsi = np.where(zero_loss & zero_gain, 50.0, rsi)

    out = pd.Series(rsi, index=close.index, dtype=float)

    out.iloc[:length] = np.nan

    return out



def build_features(
    bars: pd.DataFrame,
    cfg: Config,
) -> dict:

    """EMAs 55/200 y detección de cruces (sin lookahead)."""

    close = bars["close"]

    ef = ema_closed(close, cfg.ema_fast).to_numpy(float)
    es = ema_closed(close, cfg.ema_slow).to_numpy(float)

    prev_ef = np.r_[np.nan, ef[:-1]]
    prev_es = np.r_[np.nan, es[:-1]]

    with np.errstate(invalid="ignore"):

        cross_up = (ef > es) & (prev_ef <= prev_es)
        cross_down = (ef < es) & (prev_ef >= prev_es)

    valid = (
        np.isfinite(ef)
        & np.isfinite(es)
        & np.isfinite(prev_ef)
        & np.isfinite(prev_es)
    )

    return {
        "ef": ef,
        "es": es,
        "cross_up": cross_up,
        "cross_down": cross_down,
        "valid": valid,
    }


def _true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:

    prev_close = np.r_[
        np.nan,
        close[:-1],
    ]

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    if len(tr):
        tr[0] = high[0] - low[0]

    return tr


def adx_vec(
    bars: pd.DataFrame,
    dilen: int = 14,
    adxlen: int = 14,
) -> pd.DataFrame:

    """ADX/DI de Wilder (misma implementación que strategy1)."""

    if dilen <= 0 or adxlen <= 0:
        raise ValueError("ADX lengths must be > 0")

    n = len(bars)

    out = pd.DataFrame(
        {
            "adx": np.nan,
            "plus_di": np.nan,
            "minus_di": np.nan,
        },
        index=bars.index,
    )

    if n == 0:
        return out

    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)

    up = np.r_[
        0.0,
        high[1:] - high[:-1],
    ]

    down = np.r_[
        0.0,
        low[:-1] - low[1:],
    ]

    plus_dm = np.where(
        (up > down) & (up > 0),
        up,
        0.0,
    )

    minus_dm = np.where(
        (down > up) & (down > 0),
        down,
        0.0,
    )

    tr = _true_range(high, low, close)

    a_di = 1.0 / dilen

    tr_rma = pd.Series(tr).ewm(
        alpha=a_di,
        adjust=False,
    ).mean().to_numpy()

    p_rma = pd.Series(plus_dm).ewm(
        alpha=a_di,
        adjust=False,
    ).mean().to_numpy()

    m_rma = pd.Series(minus_dm).ewm(
        alpha=a_di,
        adjust=False,
    ).mean().to_numpy()

    safe_tr = np.where(tr_rma != 0, tr_rma, 1.0)

    plus_di = np.where(
        tr_rma != 0,
        100.0 * p_rma / safe_tr,
        0.0,
    )

    minus_di = np.where(
        tr_rma != 0,
        100.0 * m_rma / safe_tr,
        0.0,
    )

    summ = plus_di + minus_di

    dx = np.abs(plus_di - minus_di) / np.where(
        summ != 0,
        summ,
        1.0,
    )

    adx = pd.Series(100.0 * dx).ewm(
        alpha=1.0 / adxlen,
        adjust=False,
    ).mean().to_numpy()

    visible_from = dilen + adxlen - 2

    if visible_from < n:

        out.iloc[visible_from:, 0] = adx[visible_from:]
        out.iloc[visible_from:, 1] = plus_di[visible_from:]
        out.iloc[visible_from:, 2] = minus_di[visible_from:]

    return out


def adx_reversal_flags(
    adx: np.ndarray,
    key_level: float,
) -> np.ndarray:

    """
    Marca la vela donde el ADX revierte.

    Es la misma regla que usa la clase `ADX` (strategy/ADX/ADX.py):
    el ADX gira a la baja justo después de un pico por encima de
    `key_level`.
    """

    adx = np.asarray(adx, dtype=float)

    flags = np.zeros(len(adx), dtype=bool)

    if len(adx) < 3:
        return flags

    a0 = adx[:-2]
    a1 = adx[1:-1]
    a2 = adx[2:]

    with np.errstate(invalid="ignore"):

        rule1 = a2 < a1
        rule2 = a1 > a0
        rule3 = a1 > key_level

    flags[2:] = rule1 & rule2 & rule3

    return flags


def rsi_exit_flags(
    rsi: np.ndarray,
    level: float,
) -> np.ndarray:

    """
    Marca la vela donde el RSI cruza A LA BAJA `level`.

    Se usa como take-profit parcial de SHORT: el RSI entra en la
    zona de sobreventa y se reduce la posición.
    """

    rsi = np.asarray(rsi, dtype=float)

    flags = np.zeros(len(rsi), dtype=bool)

    if len(rsi) < 2:
        return flags

    prev = rsi[:-1]
    cur = rsi[1:]

    with np.errstate(invalid="ignore"):

        flags[1:] = (prev > level) & (cur <= level)

    return flags



def _source_minutes(data: pd.DataFrame) -> float:

    minutes = float(
        data.index
        .to_series()
        .diff()
        .dropna()
        .dt.total_seconds()
        .median()
        / 60.0
    )

    if not np.isfinite(minutes) or minutes <= 0:
        return 1.0

    return minutes


def build_5m_adx_exit(
    raw: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: Config,
) -> Optional[np.ndarray]:

    """
    Reversiones del ADX en `cfg.adx_timeframe` alineadas a las velas
    de la estrategia.

    Para cada vela de la estrategia se toma la última vela de
    `adx_timeframe` ya cerrada en su cierre, sin lookahead. Devuelve
    `None` si el dataset de origen es más grueso que `adx_timeframe`.
    """

    if not cfg.adx_exit_enabled or len(bars) == 0:
        return None

    target = pd.Timedelta(cfg.adx_timeframe)

    source_minutes = _source_minutes(raw)

    if pd.Timedelta(minutes=source_minutes) >= target:
        return None

    bars_low = resample_ohlcv(
        raw,
        cfg.adx_timeframe,
        source_minutes,
    )

    if len(bars_low) < cfg.adx_dilen + cfg.adx_adxlen + 2:
        return None

    adx = adx_vec(
        bars_low,
        cfg.adx_dilen,
        cfg.adx_adxlen,
    )["adx"].to_numpy(float)

    flags = adx_reversal_flags(adx, cfg.adx_key_level)

    low_end = (bars_low.index + target).asi8
    bar_close = (bars.index + pd.Timedelta(cfg.timeframe)).asi8

    pos = np.searchsorted(low_end, bar_close, side="right") - 1

    out = np.zeros(len(bars), dtype=bool)

    ok = pos >= 0

    out[ok] = flags[pos[ok]]

    return out


def build_rsi_exit(
    bars: pd.DataFrame,
    cfg: Config,
) -> Optional[np.ndarray]:

    """
    Cruces a la baja del RSI en el timeframe de la estrategia (30m),
    alineados 1:1 con las velas. Devuelve `None` si está deshabilitado.
    """

    if not cfg.rsi_exit_enabled or len(bars) == 0:
        return None

    rsi = rsi_vec(
        bars["close"],
        cfg.rsi_len,
    ).to_numpy(float)

    return rsi_exit_flags(rsi, cfg.rsi_exit_level)



# ---------------------------------------------------------------------------
# Reporte / métricas (mismo contrato que strategy1)
# ---------------------------------------------------------------------------

def positions_from_trades(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    """Agrupa los tramos en una fila por posición."""

    if trades.empty:

        return pd.DataFrame(
            columns=[
                "position_id",
                "side",
                "entry_time",
                "exit_time",
                "pnl_net",
                "fees",
                "risk",
                "legs",
                "reason",
                "r_multiple",
            ]
        )

    pos = (
        trades
        .groupby("position_id", sort=True)
        .agg(
            side=("side", "first"),
            entry_time=("entry_time", "first"),
            exit_time=("exit_time", "last"),
            pnl_net=("pnl_net", "sum"),
            fees=("fees", "sum"),
            risk=("position_risk", "first"),
            legs=("trade_id", "count"),
            reason=("reason", "last"),
        )
        .reset_index()
    )

    pos["r_multiple"] = np.where(
        pos["risk"] > 0,
        pos["pnl_net"] / pos["risk"],
        np.nan,
    )

    return pos


def position_stats(
    pos: pd.DataFrame,
) -> dict:

    if pos.empty:

        return {
            "positions": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "net_pnl": 0.0,
            "avg_pnl": 0.0,
            "expectancy_r": None,
            "max_consecutive_losses": 0,
        }

    pnl = pos["pnl_net"].astype(float)

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())

    streak = 0
    best_streak = 0

    for v in pnl.to_numpy():

        streak = streak + 1 if v < 0 else 0
        best_streak = max(best_streak, streak)

    r = pos["r_multiple"].dropna()

    return {
        "positions": int(len(pnl)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else None
        ),
        "net_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "expectancy_r": (
            float(r.mean())
            if len(r)
            else None
        ),
        "max_consecutive_losses": int(best_streak),
    }


def calculate_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
) -> dict:

    pos = positions_from_trades(trades)

    stats = position_stats(pos)

    if equity.empty:

        final_equity = initial_capital
        max_dd = 0.0
        sharpe = 0.0
        sortino = 0.0
        min_equity = initial_capital
        min_vs_capital_pct = 0.0
        dd_peak_equity = initial_capital
        dd_trough_equity = initial_capital
        dd_peak_vs_capital_pct = 0.0
        dd_trough_vs_capital_pct = 0.0

    else:

        eq = equity["equity"].astype(float)

        final_equity = float(eq.iloc[-1])

        peak = eq.cummax()
        dd = (eq / peak - 1.0) * 100.0
        max_dd = float(dd.min())

        min_equity = float(eq.min())

        min_vs_capital_pct = (
            float((eq / initial_capital - 1.0).min() * 100.0)
            if initial_capital
            else 0.0
        )

        trough_pos = int(dd.idxmin())

        dd_trough_equity = float(eq.loc[trough_pos])
        dd_peak_equity = float(peak.loc[trough_pos])

        dd_peak_vs_capital_pct = (
            float((dd_peak_equity / initial_capital - 1.0) * 100.0)
            if initial_capital
            else 0.0
        )

        dd_trough_vs_capital_pct = (
            float((dd_trough_equity / initial_capital - 1.0) * 100.0)
            if initial_capital
            else 0.0
        )

        daily = (
            equity
            .set_index("timestamp")["equity"]
            .resample("1D")
            .last()
            .dropna()
        )

        rets = daily.pct_change().dropna()

        sharpe = 0.0
        sortino = 0.0

        vol = (
            rets.std(ddof=1)
            if len(rets) > 1
            else 0.0
        )

        if len(rets) > 1 and np.isfinite(vol) and vol > 0:
            sharpe = float(
                rets.mean() / vol * math.sqrt(365)
            )

        downside_sq = (
            np.minimum(rets.to_numpy(float), 0.0) ** 2
        )

        downside_dev = (
            math.sqrt(float(np.mean(downside_sq)))
            if len(rets)
            else 0.0
        )

        if downside_dev > 0:
            sortino = float(
                rets.mean() / downside_dev * math.sqrt(365)
            )

    net = final_equity - initial_capital

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "net_pnl": net,
        "return_pct": (
            net / initial_capital * 100.0
            if initial_capital
            else 0.0
        ),
        "max_drawdown_pct": max_dd,
        "max_dd_peak_equity": dd_peak_equity,
        "max_dd_trough_equity": dd_trough_equity,
        "max_dd_peak_vs_capital_pct": dd_peak_vs_capital_pct,
        "max_dd_trough_vs_capital_pct": dd_trough_vs_capital_pct,
        "min_equity": min_equity,
        "min_equity_vs_capital_pct": min_vs_capital_pct,
        "return_over_maxdd": (
            (net / initial_capital * 100.0) / abs(max_dd)
            if (initial_capital and max_dd < 0)
            else None
        ),
        "sharpe_daily_365": sharpe,
        "sortino_daily_365": sortino,
        "legs": int(len(trades)),
        **stats,
    }


def equity_curve_from_trades(
    trades: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:

    columns = [
        "timestamp",
        "trade_id",
        "pnl_net",
        "cum_pnl",
        "equity",
    ]

    if trades.empty or "exit_time" not in trades.columns:
        return pd.DataFrame(columns=columns)

    df = (
        trades
        .sort_values(
            ["exit_time", "trade_id"],
            kind="stable",
        )
        .copy()
    )

    df["cum_pnl"] = df["pnl_net"].astype(float).cumsum()
    df["equity"] = initial_capital + df["cum_pnl"]

    out = (
        df[
            [
                "trade_id",
                "exit_time",
                "pnl_net",
                "cum_pnl",
                "equity",
            ]
        ]
        .rename(columns={"exit_time": "timestamp"})
    )

    return out.reset_index(drop=True)[columns]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_equity(
    curve: pd.DataFrame,
    output_path: str | Path,
    initial_capital: float,
    title: str = "Equity curve",
    dpi: int = 150,
) -> Optional[Path]:

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

    except Exception as exc:

        print(
            f"[PLOT] matplotlib no disponible "
            f"({exc}); se omite el gráfico.",
            flush=True,
        )

        return None

    if curve.empty:

        print(
            "[PLOT] No hay trades: "
            "no se genera la curva de equity.",
            flush=True,
        )

        return None

    out = Path(output_path)

    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_datetime(curve["timestamp"], utc=True)
    equity = curve["equity"].to_numpy(float)
    pnl = curve["pnl_net"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        x,
        equity,
        color="#1f77b4",
        linewidth=1.4,
        label="Equity",
    )

    ax.scatter(
        x[pnl >= 0],
        equity[pnl >= 0],
        s=14,
        color="#2ca02c",
        alpha=0.7,
        zorder=3,
        label="Cerrar ganadora",
    )

    ax.scatter(
        x[pnl < 0],
        equity[pnl < 0],
        s=14,
        color="#d62728",
        alpha=0.7,
        zorder=3,
        label="Cerrar perdedora",
    )

    ax.axhline(
        initial_capital,
        color="grey",
        linestyle="--",
        linewidth=0.9,
        label=f"Capital inicial ({initial_capital:,.2f})",
    )

    with np.errstate(invalid="ignore"):

        peak = np.maximum.accumulate(equity)
        dd = (equity / peak - 1.0) * 100.0

    dd_text = (
        f" | max DD {dd.min():.2f}%"
        if len(dd)
        else ""
    )

    ax.set_title(
        f"{title} | {len(curve):,} cierres{dd_text}"
    )

    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Equity")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    locator = mdates.AutoDateLocator()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)

    print(f"[PLOT] Curva de equity guardada en: {out}", flush=True)

    return out


# ---------------------------------------------------------------------------
# Reporte en consola / disco
# ---------------------------------------------------------------------------

def _fmt(v, spec=".3f"):
    return "N/A" if v is None else format(v, spec)


def print_report(
    m: dict,
    cfg: Config,
    data: pd.DataFrame,
):

    print("\n" + "=" * 72)
    print("STRATEGY 2 — EMA CROSS 55/200 (30m)")
    print("=" * 72)

    print(
        f"Periodo         : "
        f"{data.index[0]} -> {data.index[-1]}"
    )

    print(
        f"Timeframe       : {cfg.timeframe}"
    )

    print(
        f"EMAs            : "
        f"fast={cfg.ema_fast} slow={cfg.ema_slow}"
    )

    print(
        f"Capital         : "
        f"{m['initial_capital']:,.2f} -> "
        f"{m['final_equity']:,.2f}"
    )

    print(
        f"PnL neto        : "
        f"{m['net_pnl']:,.2f} ({m['return_pct']:.2f}%)"
    )

    print(
        f"Max drawdown    : "
        f"{m['max_drawdown_pct']:.2f}% "
        f"(pico {m['max_dd_peak_equity']:,.2f} "
        f"[{m['max_dd_peak_vs_capital_pct']:+.2f}%] "
        f"-> valle {m['max_dd_trough_equity']:,.2f} "
        f"[{m['max_dd_trough_vs_capital_pct']:+.2f}%])"
    )

    print(
        f"Equity mínimo   : "
        f"{m['min_equity']:,.2f} "
        f"({m['min_equity_vs_capital_pct']:.2f}% "
        f"vs capital inicial)"
    )

    print(
        f"Retorno / MaxDD : {_fmt(m['return_over_maxdd'], '.2f')}"
    )

    print(
        f"Sharpe / Sortino: "
        f"{m['sharpe_daily_365']:.2f} / "
        f"{m['sortino_daily_365']:.2f}"
    )

    print(
        f"Posiciones      : "
        f"{m['positions']:,} (tramos: {m['legs']:,})"
    )

    print(
        f"Win rate        : "
        f"{m['win_rate_pct']:.2f}% "
        f"({m['wins']} ganadoras / {m['losses']} perdedoras)"
    )

    print(f"Profit factor   : {_fmt(m['profit_factor'])}")
    print(f"Expectancy (R)  : {_fmt(m['expectancy_r'])}")
    print(f"Avg PnL/posición: {m['avg_pnl']:,.4f}")
    print(f"Racha pérdidas  : {m['max_consecutive_losses']}")

    print(
        f"Costos          : "
        f"{cfg.commission_bps} bps comisión | "
        f"{cfg.slippage_bps} bps slippage"
    )

    if cfg.stop_loss_pct > 0:

        print(
            f"Sizing / stop   : "
            f"riesgo {cfg.risk_pct:.2%} "
            f"(lev máx {cfg.max_leverage}x) / "
            f"stop {cfg.stop_loss_pct:.2%}"
        )

    else:

        print(
            f"Sizing          : "
            f"lev máx {cfg.max_leverage}x / stop OFF"
        )

    print(
        f"Lados           : "
        f"{'LONG ' if cfg.allow_long else ''}"
        f"{'SHORT' if cfg.allow_short else ''}"
    )

    if cfg.rsi_exit_enabled:

        print(
            f"RSI exit        : "
            f"RSI({cfg.rsi_len}) cruza <= {cfg.rsi_exit_level:g} "
            f"en {cfg.timeframe} -> "
            f"cierra {cfg.rsi_exit_pct:g}% (solo SHORT)"
        )

    print("=" * 72)


def side_stats(trades: pd.DataFrame) -> dict:

    out = {}

    for side in ("LONG", "SHORT"):

        pos = positions_from_trades(
            trades[trades["side"] == side]
            if len(trades)
            else trades
        )

        out[side] = position_stats(pos)

    return out


def exit_reason_breakdown(trades: pd.DataFrame) -> dict:

    if trades.empty:
        return {}

    grouped = (
        trades
        .groupby("reason", sort=False)
        .agg(
            legs=("trade_id", "count"),
            positions=("position_id", "nunique"),
            pnl_net=("pnl_net", "sum"),
        )
    )

    return {
        str(reason): {
            "legs": int(row["legs"]),
            "positions": int(row["positions"]),
            "pnl_net": float(row["pnl_net"]),
        }
        for reason, row in grouped.iterrows()
    }


def dataset_quality(data: pd.DataFrame) -> dict:

    if len(data) < 2:

        return {
            "bars": int(len(data)),
            "gaps_gt_1bar": 0,
            "max_gap_minutes": 0.0,
        }

    delta = (
        data.index
        .to_series()
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    median = float(delta.median())

    gaps = delta[delta > median]

    return {
        "bars": int(len(data)),
        "gaps_gt_1bar": int(len(gaps)),
        "max_gap_minutes": (
            float(gaps.max())
            if len(gaps)
            else median
        ),
    }


def oos_split(
    trades: pd.DataFrame,
    oos_start: str,
) -> Optional[dict]:

    pos = positions_from_trades(trades)

    if pos.empty:
        return None

    cut = pd.Timestamp(oos_start)

    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    else:
        cut = cut.tz_convert("UTC")

    return {
        "cut": str(cut),
        "before_cut": position_stats(
            pos[pos["entry_time"] < cut]
        ),
        "from_cut": position_stats(
            pos[pos["entry_time"] >= cut]
        ),
    }


def print_oos_split(
    trades: pd.DataFrame,
    oos_start: str,
):

    split = oos_split(trades, oos_start)

    if split is None:
        return

    print(
        f"\nSplit por fecha de entrada (corte {oos_start}):"
    )

    print(
        f"  {'':<14}"
        f"{'posiciones':>11}"
        f"{'win%':>8}"
        f"{'PF':>8}"
        f"{'exp.R':>8}"
        f"{'PnL neto':>14}"
    )

    for key, name in (
        ("before_cut", "antes"),
        ("from_cut", "desde el corte"),
    ):

        s = split[key]

        print(
            f"  {name:<14}"
            f"{s['positions']:>11}"
            f"{s['win_rate_pct']:>8.1f}"
            f"{_fmt(s['profit_factor'], '.2f'):>8}"
            f"{_fmt(s['expectancy_r'], '.2f'):>8}"
            f"{s['net_pnl']:>14,.2f}"
        )


def build_report(
    cfg: Config,
    data: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict,
    source: Optional[str] = None,
    oos_start: Optional[str] = None,
) -> dict:

    report = {

        "strategy": "strategy2",

        "generated_at": (
            datetime.now(timezone.utc).isoformat()
        ),

        "dataset": {

            "source": source,

            "bars": int(len(data)),

            "start": data.index[0].isoformat(),

            "end": data.index[-1].isoformat(),

            "quality": dataset_quality(data),
        },

        "config": asdict(cfg),

        "metrics": metrics,

        "by_side": side_stats(trades),

        "exit_reasons": exit_reason_breakdown(trades),
    }

    if oos_start:
        report["oos_split"] = oos_split(trades, oos_start)

    return _jsonable(report)


def write_outputs(
    output_dir: str | Path,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    curve: pd.DataFrame,
    report: dict,
    write_equity_csv: bool = False,
) -> dict[str, Path]:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades.to_csv(out / "trades.csv", index=False)

    positions_from_trades(trades).to_csv(
        out / "positions.csv",
        index=False,
    )

    curve.to_csv(out / "equity_curve.csv", index=False)

    if write_equity_csv:
        equity.to_csv(out / "equity.csv", index=False)

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(report["metrics"]), f, indent=2)

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(report["config"], f, indent=2)

    written = [
        "trades.csv",
        "report.json",
        "metrics.json",
        "config.json",
        "positions.csv",
        "equity_curve.csv",
    ]

    if write_equity_csv:
        written.append("equity.csv")

    print(
        f"[OUT] {out.resolve()} -> {', '.join(written)}",
        flush=True,
    )

    return out


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class Strategy2Backtester:

    """
    Backtester de cruce de EMAs 55/200 sobre velas de 30m.

    Señales calculadas al cierre de la vela y ejecutadas en la
    apertura de la vela siguiente.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        verbose: bool = False,
        adx_exit: Optional[np.ndarray] = None,
        rsi_exit: Optional[np.ndarray] = None,
    ):

        self.data = data
        self.cfg = config
        self.verbose = verbose

        if adx_exit is None or len(adx_exit) != len(data):
            adx_exit = np.zeros(len(data), dtype=bool)

        self.adx_exit = np.asarray(adx_exit, dtype=bool)

        if rsi_exit is None or len(rsi_exit) != len(data):
            rsi_exit = np.zeros(len(data), dtype=bool)

        self.rsi_exit = np.asarray(rsi_exit, dtype=bool)

        self.sig = build_features(data, config)

        self.o = data["open"].to_numpy(float)
        self.h = data["high"].to_numpy(float)
        self.l = data["low"].to_numpy(float)
        self.c = data["close"].to_numpy(float)

        self.times = data.index

        self.cash = config.initial_capital

        self.position: Optional[Position] = None

        # Objetivo pendiente de ejecutar: +1 LONG, -1 SHORT.
        self.pending_target: Optional[int] = None

        # Salida pendiente de ejecutar (ADX o RSI): (motivo, %).
        self.pending_exit: Optional[tuple[str, float]] = None

        self.trades: list[Trade] = []

        self.trade_counter = 0
        self.position_counter = 0

        self.eq_time = []
        self.eq_cash = []
        self.eq_equity = []
        self.eq_side = []
        self.eq_qty = []

    # ------------------------------------------------------------------
    # Precios y costes
    # ------------------------------------------------------------------

    def _execution_price(
        self,
        raw_price: float,
        side: int,
    ) -> float:

        bps = self.cfg.slippage_bps / 10_000.0

        if side == 1:
            return raw_price * (1.0 + bps)

        return raw_price * (1.0 - bps)

    def _fee(self, notional: float) -> float:

        return (
            abs(notional)
            * self.cfg.commission_bps
            / 10_000.0
        )

    def _initial_stop_price(
        self,
        side: int,
        entry_price: float,
        stop_pct: float,
    ) -> float:

        if side == 1:
            return entry_price * (1.0 - stop_pct)

        return entry_price * (1.0 + stop_pct)

    def _estimated_stop_risk_per_unit(
        self,
        side: int,
        entry_price: float,
        stop_pct: float,
    ) -> float:

        stop_price = self._initial_stop_price(
            side,
            entry_price,
            stop_pct,
        )

        stop_exec = self._execution_price(stop_price, -side)

        price_loss = abs(stop_exec - entry_price)

        fee_rate = self.cfg.commission_bps / 10_000.0

        return (
            price_loss
            + (entry_price + abs(stop_exec)) * fee_rate
        )

    # ------------------------------------------------------------------
    # Apertura
    # ------------------------------------------------------------------

    def _open(
        self,
        side: int,
        price: float,
        time: pd.Timestamp,
        stop_pct: float,
    ):

        if self.position is not None:
            return

        cfg = self.cfg

        exec_price = self._execution_price(price, side)

        if (
            self.cash <= 0
            or not np.isfinite(exec_price)
            or exec_price <= 0
        ):
            return

        if stop_pct > 0:

            risk_per_unit = (
                self._estimated_stop_risk_per_unit(
                    side,
                    exec_price,
                    stop_pct,
                )
            )

            if risk_per_unit <= 0:
                return

            qty_by_risk = (
                self.cash * cfg.risk_pct / risk_per_unit
            )

        else:

            risk_per_unit = 0.0
            qty_by_risk = np.inf

        qty_by_leverage = (
            self.cash * cfg.max_leverage / exec_price
        )

        qty = min(qty_by_risk, qty_by_leverage)

        if qty <= 0:
            return

        fee = self._fee(exec_price * qty)

        self.cash -= fee

        self.position_counter += 1

        stop_price = (
            self._initial_stop_price(
                side,
                exec_price,
                stop_pct,
            )
            if stop_pct > 0
            else 0.0
        )

        self.position = Position(
            position_id=self.position_counter,
            side=side,
            quantity=qty,
            initial_quantity=qty,
            entry_price=exec_price,
            entry_time=time,
            entry_fee=fee,
            stop_pct=stop_pct,
            risk_amount=qty * risk_per_unit,
            stop_price_estimate=stop_price,
        )

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------

    def close_position(
        self,
        percent: float,
        price: float,
        time: pd.Timestamp,
        reason: str,
    ):

        pos = self.position

        if pos is None:
            return

        if (
            not np.isfinite(percent)
            or percent <= 0
            or percent > 100
        ):
            raise ValueError("percent debe estar entre 0 y 100.")

        fraction = float(percent) / 100.0

        qty = pos.quantity * fraction

        if qty <= 0:
            return

        side = pos.side

        exec_price = self._execution_price(price, -side)

        gross = (
            (exec_price - pos.entry_price) * qty * side
        )

        exit_fee = self._fee(exec_price * qty)

        entry_fee_alloc = (
            pos.entry_fee * (qty / pos.quantity)
        )

        total_fees = entry_fee_alloc + exit_fee

        pnl_net = gross - total_fees

        self.cash += gross - exit_fee

        self.trade_counter += 1

        notional_entry = pos.entry_price * qty

        return_pct = (
            pnl_net / notional_entry * 100.0
            if notional_entry != 0
            else 0.0
        )

        self.trades.append(
            Trade(
                trade_id=self.trade_counter,
                position_id=pos.position_id,
                side="LONG" if side == 1 else "SHORT",
                entry_time=pos.entry_time,
                exit_time=time,
                entry_price=pos.entry_price,
                exit_price=exec_price,
                quantity=qty,
                pnl_gross=gross,
                fees=total_fees,
                pnl_net=pnl_net,
                return_pct=return_pct,
                reason=reason,
                position_risk=pos.risk_amount,
            )
        )

        remaining = pos.quantity - qty

        eps = max(1e-12, pos.initial_quantity * 1e-12)

        if fraction >= 1.0 or remaining <= eps:

            self.position = None

        else:

            pos.quantity = remaining
            pos.entry_fee -= entry_fee_alloc

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def _check_stop_loss(self, i: int) -> Optional[str]:

        pos = self.position

        if pos is None or pos.stop_pct <= 0:
            return None

        bo = self.o[i]
        bh = self.h[i]
        bl = self.l[i]

        e = pos.entry_price
        side = pos.side

        if side == 1:

            stop = e * (1.0 - pos.stop_pct)

            if bl > stop:
                return None

            fill = bo if bo <= stop else stop

        else:

            stop = e * (1.0 + pos.stop_pct)

            if bh < stop:
                return None

            fill = bo if bo >= stop else stop

        reason = f"STOP_LOSS_{pos.stop_pct * 100:.2f}%"

        self.close_position(
            100.0,
            fill,
            self.times[i],
            reason,
        )

        if self.verbose:
            print(
                f"[STOP] {self.times[i]} | "
                f"{'LONG' if side == 1 else 'SHORT'} | "
                f"stop={stop:.8f} | fill={fill:.8f}",
                flush=True,
            )

        return reason

    # ------------------------------------------------------------------
    # Ejecutar objetivo pendiente
    # ------------------------------------------------------------------

    def _execute_target(self, i: int, target: int):

        bar_open = self.o[i]
        ts = self.times[i]

        # Reversión: cerrar la posición contraria.
        if (
            self.position is not None
            and self.position.side != target
        ):

            self.close_position(
                100.0,
                bar_open,
                ts,
                "EMA_CROSS_EXIT",
            )

        # Abrir en la dirección del cruce.
        if self.position is None:
            self._open(
                target,
                bar_open,
                ts,
                self.cfg.stop_loss_pct,
            )

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    def _mark(
        self,
        time: pd.Timestamp,
        close: float,
    ):

        pos = self.position
        equity = self.cash

        if pos is not None:

            liquidation_price = self._execution_price(
                close,
                -pos.side,
            )

            gross = (
                (liquidation_price - pos.entry_price)
                * pos.quantity
                * pos.side
            )

            exit_fee = self._fee(
                liquidation_price * pos.quantity
            )

            equity += gross - exit_fee

        self.eq_time.append(time)
        self.eq_cash.append(self.cash)
        self.eq_equity.append(equity)

        self.eq_side.append(
            "FLAT"
            if pos is None
            else ("LONG" if pos.side == 1 else "SHORT")
        )

        self.eq_qty.append(
            0.0 if pos is None else pos.quantity
        )

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:

        cfg = self.cfg
        sig = self.sig

        n = len(self.data)

        cross_up = sig["cross_up"]
        cross_down = sig["cross_down"]
        valid = sig["valid"]

        adx_minutes = int(
            pd.Timedelta(cfg.adx_timeframe)
            / pd.Timedelta(minutes=1)
        )

        adx_reason = f"ADX_{adx_minutes}M_REVERSAL"

        adx_exit_pct = float(cfg.adx_exit_pct)

        adx_exit_reason = (
            adx_reason
            if adx_exit_pct >= 100.0
            else f"{adx_reason}_{adx_exit_pct:g}%"
        )

        rsi_reason = (
            f"RSI{cfg.rsi_len}_{cfg.rsi_exit_level:g}_EXIT"
        )

        rsi_exit_pct = float(cfg.rsi_exit_pct)

        rsi_exit_reason = (
            rsi_reason
            if rsi_exit_pct >= 100.0
            else f"{rsi_reason}_{rsi_exit_pct:g}%"
        )

        started = time.perf_counter()

        last_log = -10

        print(f"[BACKTEST] {n:,} velas {cfg.timeframe}", flush=True)

        for i in range(n):

            # -------------------------------------------------------
            # 1) Ejecutar objetivo del cierre anterior.
            # -------------------------------------------------------

            if self.pending_target is not None:

                self._execute_target(i, self.pending_target)
                self.pending_target = None

            if self.pending_exit is not None:

                exit_reason, exit_pct = self.pending_exit

                if self.verbose and self.position is not None:
                    print(
                        f"[EXIT] {self.times[i]} | EXIT "
                        f"{exit_pct:g}% | {exit_reason}",
                        flush=True,
                    )

                self.close_position(
                    exit_pct,
                    self.o[i],
                    self.times[i],
                    exit_reason,
                )

                self.pending_exit = None

            # -------------------------------------------------------
            # 2) Stop intrabar (si está habilitado).
            # -------------------------------------------------------

            stop_reason = self._check_stop_loss(i)

            stop_hit = stop_reason is not None

            # -------------------------------------------------------
            # 3) Señal de cruce al cierre de esta vela.
            #    Se ejecuta en la próxima apertura.
            # -------------------------------------------------------

            new_target = None
            new_exit = None

            if not stop_hit and valid[i]:

                target = None

                if cross_up[i] and cfg.allow_long:
                    target = 1

                elif cross_down[i] and cfg.allow_short:
                    target = -1

                if target is not None:

                    current = (
                        self.position.side
                        if self.position is not None
                        else None
                    )

                    if current != target:
                        new_target = target

                # Salidas parciales (ADX 5m / RSI 30m): solo si no
                # hay un cruce contrario que ya revertiría.
                if (
                    new_target is None
                    and self.position is not None
                    and self.adx_exit[i]
                ):

                    new_exit = (adx_exit_reason, adx_exit_pct)

                # Salida parcial por RSI en el timeframe de la
                # estrategia (30m): SOLO cuando la posición es SHORT
                # y el RSI cruza a la baja el nivel (sobreventa).
                if (
                    new_exit is None
                    and new_target is None
                    and self.position is not None
                    and self.position.side == -1
                    and self.rsi_exit[i]
                ):

                    new_exit = (rsi_exit_reason, rsi_exit_pct)

            if new_target is not None:
                self.pending_target = new_target

            elif new_exit is not None:
                self.pending_exit = new_exit

            # -------------------------------------------------------
            # Equity
            # -------------------------------------------------------

            self._mark(
                self.times[i] + pd.Timedelta(cfg.timeframe),
                self.c[i],
            )

            pct = int((i + 1) / n * 100)

            if pct >= last_log + 10 or i == n - 1:

                elapsed = time.perf_counter() - started

                rate = (
                    (i + 1) / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"[BACKTEST] {pct:3d}% | "
                    f"{rate:,.0f} velas/s | "
                    f"{elapsed:.1f}s | "
                    f"tramos {len(self.trades)}",
                    flush=True,
                )

                last_log = pct

        # -----------------------------------------------------------
        # Cierre al final del dataset.
        # -----------------------------------------------------------

        if (
            self.position is not None
            and cfg.force_close_at_end
        ):

            last_ts = (
                self.times[-1] + pd.Timedelta(cfg.timeframe)
            )

            self.close_position(
                100.0,
                self.c[-1],
                last_ts,
                "END_OF_DATA",
            )

            self._mark(last_ts, self.c[-1])

        trades = (
            pd.DataFrame(
                [asdict(t) for t in self.trades]
            )
            if self.trades
            else empty_trades()
        )

        equity = pd.DataFrame({
            "timestamp": self.eq_time,
            "cash": self.eq_cash,
            "equity": self.eq_equity,
            "position_side": self.eq_side,
            "position_quantity": self.eq_qty,
        })

        print(
            f"[BACKTEST] Finalizado en "
            f"{time.perf_counter() - started:.2f}s | "
            f"tramos={len(trades):,}",
            flush=True,
        )

        return trades, equity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():

    d = Config()

    p = argparse.ArgumentParser(
        description=(
            "Backtester Strategy2 (EMA CROSS 55/200) "
            "sobre OHLCV en Parquet."
        )
    )

    p.add_argument(
        "parquet",
        help="Ruta al dataset .parquet",
    )

    p.add_argument(
        "--capital",
        type=float,
        default=d.initial_capital,
    )

    p.add_argument(
        "--ema-fast",
        type=int,
        default=d.ema_fast,
    )

    p.add_argument(
        "--ema-slow",
        type=int,
        default=d.ema_slow,
    )

    p.add_argument(
        "--timeframe",
        default=d.timeframe,
        help="Timeframe de la estrategia (default: 30min).",
    )

    p.add_argument(
        "--commission-bps",
        type=float,
        default=d.commission_bps,
    )

    p.add_argument(
        "--slippage-bps",
        type=float,
        default=d.slippage_bps,
    )

    p.add_argument(
        "--risk-pct",
        type=float,
        default=d.risk_pct,
    )

    p.add_argument(
        "--max-leverage",
        type=float,
        default=d.max_leverage,
    )

    p.add_argument(
        "--stop-loss-pct",
        type=float,
        default=d.stop_loss_pct,
        help=(
            "Stop fijo (0 = deshabilitado; "
            "solo EMACROSS)."
        ),
    )

    p.add_argument(
        "--no-adx-exit",
        action="store_true",
        help="Deshabilita la salida por reversión del ADX 5m.",
    )

    p.add_argument(
        "--no-rsi-exit",
        action="store_true",
        help=(
            "Deshabilita la salida parcial por RSI "
            "(solo SHORT)."
        ),
    )

    p.add_argument(
        "--rsi-len",
        type=int,
        default=d.rsi_len,
        help="Periodo del RSI de salida (default: 14).",
    )

    p.add_argument(
        "--rsi-exit-level",
        type=float,
        default=d.rsi_exit_level,
        help=(
            "Nivel del RSI que dispara la salida parcial del "
            "SHORT (default: 32)."
        ),
    )

    p.add_argument(
        "--rsi-exit-pct",
        type=float,
        default=d.rsi_exit_pct,
        help=(
            "Porcentaje del SHORT a cerrar en cada cruce del RSI "
            "por debajo del nivel (100 = cierre total; "
            "<100 = salida parcial; default: 50)."
        ),
    )

    p.add_argument(
        "--adx-timeframe",
        default=d.adx_timeframe,
        help="Timeframe del ADX de salida (default: 5min).",
    )

    p.add_argument(
        "--adx-dilen",
        type=int,
        default=d.adx_dilen,
    )

    p.add_argument(
        "--adx-adxlen",
        type=int,
        default=d.adx_adxlen,
    )

    p.add_argument(
        "--adx-key-level",
        type=float,
        default=d.adx_key_level,
        help="Nivel del pico de ADX que habilita la reversión.",
    )

    p.add_argument(
        "--adx-exit-pct",
        type=float,
        default=d.adx_exit_pct,
        help=(
            "Porcentaje de la posición a cerrar en cada reversión "
            "del ADX (100 = cierre total; <100 = salida parcial)."
        ),
    )

    p.add_argument(
        "--no-long",
        action="store_true",
        help="Deshabilita entradas LONG.",
    )

    p.add_argument(
        "--no-short",
        action="store_true",
        help="Deshabilita entradas SHORT.",
    )

    p.add_argument(
        "--no-force-close",
        action="store_true",
    )

    p.add_argument(
        "--oos-start",
        default=None,
        help="Fecha YYYY-MM-DD para split OOS.",
    )

    p.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime cada stop.",
    )

    p.add_argument(
        "--timestamp-is-close",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Indica que el timestamp es cierre, no apertura.",
    )

    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    p.add_argument(
        "--no-output",
        action="store_true",
    )

    p.add_argument(
        "--equity-csv",
        action="store_true",
    )

    p.add_argument(
        "--plot",
        default=None,
        metavar="PATH",
    )

    p.add_argument(
        "--no-plot",
        action="store_true",
    )

    p.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
    )

    return p.parse_args()


def make_config(args) -> Config:

    return Config(
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        timeframe=args.timeframe,
        allow_long=not args.no_long,
        allow_short=not args.no_short,
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        risk_pct=args.risk_pct,
        max_leverage=args.max_leverage,
        stop_loss_pct=args.stop_loss_pct,
        adx_exit_enabled=not args.no_adx_exit,
        adx_timeframe=args.adx_timeframe,
        adx_dilen=args.adx_dilen,
        adx_adxlen=args.adx_adxlen,
        adx_key_level=args.adx_key_level,
        adx_exit_pct=args.adx_exit_pct,
        rsi_exit_enabled=not args.no_rsi_exit,
        rsi_len=args.rsi_len,
        rsi_exit_level=args.rsi_exit_level,
        rsi_exit_pct=args.rsi_exit_pct,
        force_close_at_end=not args.no_force_close,
        timestamp_is_close=(
            Config().timestamp_is_close
            if args.timestamp_is_close is None
            else args.timestamp_is_close
        ),
    )


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _jsonable(obj):

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (np.floating, float)):
        return (
            None
            if not math.isfinite(float(obj))
            else float(obj)
        )

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]

    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    cfg = make_config(args)

    raw = load_parquet(args.parquet)

    data = to_timeframe(raw, cfg)

    print(
        f"[DATA] {len(data):,} velas {cfg.timeframe} | "
        f"{data.index.min()} -> {data.index.max()}",
        flush=True,
    )

    if len(data) < cfg.ema_slow + 2:

        raise ValueError(
            "El dataset tiene muy pocas velas "
            f"para EMA{cfg.ema_slow} en {cfg.timeframe}."
        )

    adx_exit = build_5m_adx_exit(raw, data, cfg)

    if cfg.adx_exit_enabled:

        if adx_exit is None:

            print(
                f"[ADX] salida por ADX {cfg.adx_timeframe} "
                "deshabilitada: dataset de origen más grueso "
                "que el timeframe del ADX.",
                flush=True,
            )

        else:

            print(
                f"[ADX] {cfg.adx_timeframe} "
                f"ADX({cfg.adx_dilen}/{cfg.adx_adxlen}) "
                f"key_level={cfg.adx_key_level:g} | "
                f"exit={cfg.adx_exit_pct:g}% | "
                f"reversiones={int(adx_exit.sum()):,}",
                flush=True,
            )

    rsi_exit = build_rsi_exit(data, cfg)

    if cfg.rsi_exit_enabled:

        if rsi_exit is None:

            print(
                "[RSI] salida por RSI deshabilitada.",
                flush=True,
            )

        else:

            print(
                f"[RSI] RSI({cfg.rsi_len}) {cfg.timeframe} | "
                f"nivel={cfg.rsi_exit_level:g} | "
                f"exit={cfg.rsi_exit_pct:g}% SOLO SHORT | "
                f"cruces={int(rsi_exit.sum()):,}",
                flush=True,
            )

    engine = Strategy2Backtester(
        data,
        cfg,
        verbose=args.verbose,
        adx_exit=adx_exit,
        rsi_exit=rsi_exit,
    )

    trades, equity = engine.run()

    metrics = calculate_metrics(
        trades,
        equity,
        cfg.initial_capital,
    )

    print_report(metrics, cfg, data)

    if args.oos_start:
        print_oos_split(trades, args.oos_start)

    curve = equity_curve_from_trades(
        trades,
        cfg.initial_capital,
    )

    if not args.no_output:

        report = build_report(
            cfg,
            data,
            trades,
            metrics,
            source=str(args.parquet),
            oos_start=args.oos_start,
        )

        write_outputs(
            args.output_dir,
            trades,
            equity,
            curve,
            report,
            write_equity_csv=args.equity_csv,
        )

    if not args.no_plot:

        plot_path = (
            args.plot
            or str(
                Path(args.output_dir or ".")
                / "equity_curve.png"
            )
        )

        plot_equity(
            curve,
            plot_path,
            cfg.initial_capital,
            title=(
                f"Strategy2 EMA{cfg.ema_fast}/"
                f"EMA{cfg.ema_slow} {cfg.timeframe} | "
                f"{metrics['initial_capital']:,.2f} -> "
                f"{metrics['final_equity']:,.2f}"
            ),
            dpi=args.plot_dpi,
        )


if __name__ == "__main__":
    main()
