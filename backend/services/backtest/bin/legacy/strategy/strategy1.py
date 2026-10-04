#!/usr/bin/env python3

"""
Backtesting Strategy1 — IMPROVED + RSI DE SALIDA
================================================

Dataset de entrada: OHLCV de 1 minuto en .parquet.

Reglas de entrada de la estrategia (sin cambios):

    1H  -> tendencia: EMA21/EMA55 + ADX14 >= 25 + DI

    30m -> confirmación: EMA21 vs EMA55

    15m -> pullback: EMA21/EMA55

    5m  -> setup: EMA21 vs EMA55

    1m  -> trigger: EMA21/EMA55 + cierre sobre/bajo EMA21 +
           ruptura del máximo/mínimo de la vela anterior

Valores default:

    - EMA 21/55 en todos los timeframes.
    - ADX 14/14, umbral 25.
    - Riesgo 1% del capital, apalancamiento máximo 1x.
    - Stop 3.0 x ATR(14) de 15m, acotado entre 0.4% y 1.5%.
    - Cierre parcial 50% de la posición RESTANTE ante contra-señal 30m.
    - Cooldown 0 min.
    - ADX máximo 40 y ADX creciente.
    - Gap mínimo EMA 1H 0.1%.
    - Rango mínimo 1m 0.5 ATR.

RSI DE SALIDA:

    - RSI(14) calculado en 1m.
    - LONG  -> cierre total si RSI >= 70.
    - SHORT -> cierre total si RSI <= 30.
    - El RSI SOLO afecta a posiciones abiertas.
    - El RSI NO participa en ninguna condición de entrada.
    - La señal RSI se genera al cierre de la vela 1m.
    - La ejecución ocurre en la apertura de la siguiente vela 1m.

Prioridad de salidas:

    1. Stop intrabar.
    2. Reversal 1H -> cierre total.
    3. Reversal 30m -> cierre parcial.
    4. RSI -> cierre total.

Convención anti-lookahead:

    Las señales usan solo velas cerradas y se ejecutan en la apertura
    de la siguiente vela 1m.

Si hay gap a través del stop, el fill es el open.

Dependencias:

    pip install "pandas>=2" pyarrow numpy matplotlib

Ejemplo:

    python backtest_strategy1_improved.py data.parquet \
        --capital 100000 \
        --commission-bps 2 \
        --slippage-bps 1 \
        --oos-start 2026-01-01 \
        --exit-rsi-len 14 \
        --exit-rsi-long 70 \
        --exit-rsi-short 30 \
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
    ema_fast: int = 21
    ema_slow: int = 55

    adx_dilen: int = 14
    adx_len: int = 14
    adx_threshold: float = 25.0

    atr_len: int = 14

    close_on_30m_pct: float = 50.0

    initial_capital: float = 10_000.0

    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    # Stop fijo de respaldo.
    stop_loss_pct: float = 0.005

    force_close_at_end: bool = True

    # --- parámetros improved ---

    risk_pct: float = 0.01
    max_leverage: float = 1.0

    atr_stop_mult: float = 3.0
    atr_stop_min_pct: float = 0.004
    atr_stop_max_pct: float = 0.015

    cooldown_min: int = 0

    adx_max: float = 40.0
    adx_rising: bool = True

    min_ema_gap_1h_pct: float = 0.001
    min_trigger_range_atr: float = 0.5

    block_hours_utc: str = ""

    timestamp_is_close: bool = False

    # ---------------------------------------------------------------
    # RSI EXCLUSIVAMENTE PARA SALIDAS
    # ---------------------------------------------------------------

    exit_rsi_enabled: bool = True
    exit_rsi_len: int = 14

    # LONG: salida si RSI >= nivel
    exit_rsi_long: float = 70.0

    # SHORT: salida si RSI <= nivel
    exit_rsi_short: float = 30.0

    def __post_init__(self):

        if self.ema_fast <= 0 or self.ema_slow <= 0:
            raise ValueError("Los periodos EMA deben ser > 0.")

        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast debe ser menor que ema_slow.")

        if self.adx_dilen <= 0 or self.adx_len <= 0 or self.atr_len <= 0:
            raise ValueError("Los periodos ADX/ATR deben ser > 0.")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital debe ser > 0.")

        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError(
                "commission_bps y slippage_bps no pueden ser negativos."
            )

        if not (0.0 < self.close_on_30m_pct <= 100.0):
            raise ValueError(
                "close_on_30m_pct debe estar entre 0 y 100."
            )

        if not (0.0 < self.risk_pct <= 0.2):
            raise ValueError(
                "risk_pct debe estar entre 0 (excl.) y 0.2."
            )

        if self.max_leverage <= 0:
            raise ValueError("max_leverage debe ser > 0.")

        if self.atr_stop_mult < 0:
            raise ValueError(
                "atr_stop_mult no puede ser negativo."
            )

        if self.atr_stop_min_pct < 0 or self.atr_stop_max_pct < 0:
            raise ValueError(
                "Los límites de stop ATR no pueden ser negativos."
            )

        if self.atr_stop_min_pct > self.atr_stop_max_pct:
            raise ValueError(
                "atr_stop_min_pct no puede superar atr_stop_max_pct."
            )

        if self.stop_loss_pct <= 0 and self.atr_stop_mult <= 0:
            raise ValueError(
                "Se requiere un stop (stop_loss_pct o atr_stop_mult)."
            )

        # RSI
        if self.exit_rsi_len <= 0:
            raise ValueError(
                "exit_rsi_len debe ser > 0."
            )

        if not (0.0 <= self.exit_rsi_long <= 100.0):
            raise ValueError(
                "exit_rsi_long debe estar entre 0 y 100."
            )

        if not (0.0 <= self.exit_rsi_short <= 100.0):
            raise ValueError(
                "exit_rsi_short debe estar entre 0 y 100."
            )

        self.blocked_hours()

    def blocked_hours(self) -> list[int]:

        txt = (self.block_hours_utc or "").strip()

        if not txt:
            return []

        try:
            hours = [
                int(x)
                for x in txt.split(",")
                if x.strip() != ""
            ]
        except ValueError:
            raise ValueError(
                f"block_hours_utc inválido: {txt!r}"
            )

        if any(h < 0 or h > 23 for h in hours):
            raise ValueError(
                "block_hours_utc debe contener horas entre 0 y 23."
            )

        return hours


# Parámetros expuestos por CLI.
CLI_KEYS = [
    "risk_pct",
    "max_leverage",
    "atr_stop_mult",
    "atr_stop_min_pct",
    "atr_stop_max_pct",
    "cooldown_min",
    "adx_max",
    "adx_rising",
    "min_ema_gap_1h_pct",
    "min_trigger_range_atr",
    "block_hours_utc",

    # RSI salida
    "exit_rsi_enabled",
    "exit_rsi_len",
    "exit_rsi_long",
    "exit_rsi_short",
]

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

        numeric_ts = pd.to_numeric(
            raw_ts,
            errors="coerce",
        )

        finite = numeric_ts.dropna()

        unit = None

        if not finite.empty:

            magnitude = float(
                finite.abs().median()
            )

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

            ts = pd.to_datetime(
                raw_ts,
                utc=True,
                errors="coerce",
            )

    else:

        ts = pd.to_datetime(
            raw_ts,
            utc=True,
            errors="coerce",
        )

    if ts.isna().any():
        raise ValueError(
            "Hay timestamps inválidos en el dataset."
        )

    out["timestamp"] = ts

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce",
        )

    out = out.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    out = out.sort_values("timestamp")

    out = out.drop_duplicates(
        "timestamp",
        keep="last",
    )

    out = out.set_index("timestamp")

    return out[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def load_parquet(path: str | Path) -> pd.DataFrame:

    print(
        f"[LOAD] Leyendo Parquet: {path}",
        flush=True,
    )

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
) -> pd.DataFrame:

    out = df.resample(
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

    return out.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    """Epoch en nanosegundos."""

    return idx.as_unit("ns").asi8


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def ema_closed(
    values: pd.Series,
    period: int,
) -> pd.Series:

    """EMA con seed = primer valor."""

    if period <= 0:
        raise ValueError(
            "EMA period must be > 0"
        )

    return values.ewm(
        alpha=2.0 / (period + 1.0),
        adjust=False,
    ).mean()


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


def atr_vec(
    bars: pd.DataFrame,
    length: int,
) -> np.ndarray:

    """ATR de Wilder."""

    tr = _true_range(
        bars["high"].to_numpy(float),
        bars["low"].to_numpy(float),
        bars["close"].to_numpy(float),
    )

    return pd.Series(tr).ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean().to_numpy()


def adx_vec(
    bars: pd.DataFrame,
    dilen: int = 14,
    adxlen: int = 14,
) -> pd.DataFrame:

    """
    ADX/DI con RMA.
    """

    if dilen <= 0 or adxlen <= 0:
        raise ValueError(
            "ADX lengths must be > 0"
        )

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

    tr = _true_range(
        high,
        low,
        close,
    )

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

    safe_tr = np.where(
        tr_rma != 0,
        tr_rma,
        1.0,
    )

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

    dx = np.abs(
        plus_di - minus_di
    ) / np.where(
        summ != 0,
        summ,
        1.0,
    )

    adx = pd.Series(
        100.0 * dx
    ).ewm(
        alpha=1.0 / adxlen,
        adjust=False,
    ).mean().to_numpy()

    visible_from = dilen + adxlen - 2

    if visible_from < n:

        out.iloc[
            visible_from:,
            0
        ] = adx[visible_from:]

        out.iloc[
            visible_from:,
            1
        ] = plus_di[visible_from:]

        out.iloc[
            visible_from:,
            2
        ] = minus_di[visible_from:]

    return out


def rsi_vec(
    values: pd.Series,
    length: int = 14,
) -> np.ndarray:

    """
    RSI de Wilder mediante RMA.

    IMPORTANTE:
    Este indicador se utiliza exclusivamente para SALIDAS.
    No se utiliza en ninguna condición de entrada.
    """

    if length <= 0:
        raise ValueError(
            "RSI length must be > 0"
        )

    close = values.to_numpy(float)

    if len(close) == 0:
        return np.array(
            [],
            dtype=float,
        )

    delta = np.r_[
        np.nan,
        np.diff(close),
    ]

    gain = np.maximum(
        delta,
        0.0,
    )

    loss = np.maximum(
        -delta,
        0.0,
    )

    # Primera vela.
    gain[0] = 0.0
    loss[0] = 0.0

    avg_gain = pd.Series(
        gain
    ).ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean().to_numpy()

    avg_loss = pd.Series(
        loss
    ).ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean().to_numpy()

    rsi = np.full(
        len(close),
        np.nan,
        dtype=float,
    )

    # Caso normal.
    valid = avg_loss > 0

    rs = np.divide(
        avg_gain,
        avg_loss,
        out=np.zeros_like(avg_gain),
        where=valid,
    )

    rsi[valid] = (
        100.0
        - 100.0 / (1.0 + rs[valid])
    )

    # Sin pérdidas y con ganancias.
    no_loss = (
        (avg_loss == 0)
        & (avg_gain > 0)
    )

    rsi[no_loss] = 100.0

    # Mercado completamente plano.
    flat = (
        (avg_loss == 0)
        & (avg_gain == 0)
    )

    rsi[flat] = 50.0

    return rsi


# ---------------------------------------------------------------------------
# Precálculo
# ---------------------------------------------------------------------------

def _map_tf(
    source_1m_open: pd.DataFrame,
    rule: str,
    cfg: Config,
    decision_ns: np.ndarray,
    adx: bool = False,
    atr: bool = False,
) -> dict:

    """
    Proyecta exclusivamente HTF cerradas y completas.
    """

    bars = resample_ohlcv(
        source_1m_open,
        rule,
    )

    expected = {
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1h": 60,
    }[rule]

    counts = (
        source_1m_open["close"]
        .resample(
            rule,
            label="left",
            closed="left",
            origin="start_day",
        )
        .count()
        .reindex(
            bars.index,
            fill_value=0,
        )
        .to_numpy()
    )

    complete = counts == expected

    ends = _ns(
        bars.index + pd.Timedelta(rule)
    )

    pos = np.searchsorted(
        ends,
        decision_ns,
        side="right",
    ) - 1

    ok = pos >= 0

    j = np.where(
        ok,
        pos,
        0,
    )

    ok &= complete[j]

    ef = ema_closed(
        bars["close"],
        cfg.ema_fast,
    ).to_numpy()

    es = ema_closed(
        bars["close"],
        cfg.ema_slow,
    ).to_numpy()

    close = bars[
        "close"
    ].to_numpy(float)

    out = {
        "ok": ok,
        "close": np.where(
            ok,
            close[j],
            np.nan,
        ),
        "ef": np.where(
            ok,
            ef[j],
            np.nan,
        ),
        "es": np.where(
            ok,
            es[j],
            np.nan,
        ),
    }

    if adx:

        a = adx_vec(
            bars,
            cfg.adx_dilen,
            cfg.adx_len,
        )

        adx_arr = a[
            "adx"
        ].to_numpy()

        plus_arr = a[
            "plus_di"
        ].to_numpy()

        minus_arr = a[
            "minus_di"
        ].to_numpy()

        out["adx"] = np.where(
            ok,
            adx_arr[j],
            np.nan,
        )

        prev = np.r_[
            np.nan,
            adx_arr[:-1],
        ]

        out["adx_prev"] = np.where(
            ok,
            prev[j],
            np.nan,
        )

        out["plus_di"] = np.where(
            ok,
            plus_arr[j],
            np.nan,
        )

        out["minus_di"] = np.where(
            ok,
            minus_arr[j],
            np.nan,
        )

    if atr:

        atr_arr = atr_vec(
            bars,
            cfg.atr_len,
        )

        out["atr"] = np.where(
            ok,
            atr_arr[j],
            np.nan,
        )

    return out


def build_features(
    data: pd.DataFrame,
    cfg: Config,
) -> dict:

    """
    Construye features sin lookahead.

    El RSI se calcula en 1m y se utiliza exclusivamente
    para salidas.
    """

    n = len(data)

    if cfg.timestamp_is_close:

        open_index = (
            data.index
            - pd.Timedelta(minutes=1)
        )

        decision_ns = _ns(
            data.index
        )

    else:

        open_index = data.index

        decision_ns = _ns(
            data.index
            + pd.Timedelta(minutes=1)
        )

    source_1m_open = data.copy()

    source_1m_open.index = open_index

    source_1m_open = (
        source_1m_open
        .sort_index()
    )

    feats = {

        "decision_ns": decision_ns,

        "1h": _map_tf(
            source_1m_open,
            "1h",
            cfg,
            decision_ns,
            adx=True,
        ),

        "30m": _map_tf(
            source_1m_open,
            "30min",
            cfg,
            decision_ns,
        ),

        "15m": _map_tf(
            source_1m_open,
            "15min",
            cfg,
            decision_ns,
            atr=True,
        ),

        "5m": _map_tf(
            source_1m_open,
            "5min",
            cfg,
            decision_ns,
        ),
    }

    c1 = data["close"]

    high = data[
        "high"
    ].to_numpy(float)

    low = data[
        "low"
    ].to_numpy(float)

    feats["1m"] = {

        "ef": ema_closed(
            c1,
            cfg.ema_fast,
        ).to_numpy(),

        "es": ema_closed(
            c1,
            cfg.ema_slow,
        ).to_numpy(),

        "close": c1.to_numpy(float),

        "high": high,
        "low": low,

        "prev_high": np.r_[
            np.nan,
            high[:-1],
        ],

        "prev_low": np.r_[
            np.nan,
            low[:-1],
        ],

        "atr": atr_vec(
            data,
            cfg.atr_len,
        ),

        # -----------------------------------------------------------
        # RSI SOLO PARA SALIDAS
        # -----------------------------------------------------------
        "rsi": rsi_vec(
            c1,
            cfg.exit_rsi_len,
        ),
    }

    feats["n"] = n

    return feats


# ---------------------------------------------------------------------------
# Señales
# ---------------------------------------------------------------------------

def build_signals(
    f: dict,
    cfg: Config,
) -> dict:

    """
    Condiciones de Strategy1 + filtros improved.

    IMPORTANTE:
    El RSI NO participa en las entradas.
    """

    n = f["n"]

    h1 = f["1h"]
    m30 = f["30m"]
    m15 = f["15m"]
    m5 = f["5m"]
    m1 = f["1m"]

    with np.errstate(
        invalid="ignore"
    ):

        # -----------------------------------------------------------
        # 1H
        # -----------------------------------------------------------

        bull1h = (
            (h1["ef"] > h1["es"])
            & (h1["adx"] >= cfg.adx_threshold)
            & (
                h1["plus_di"]
                > h1["minus_di"]
            )
        )

        bear1h = (
            (h1["ef"] < h1["es"])
            & (h1["adx"] >= cfg.adx_threshold)
            & (
                h1["minus_di"]
                > h1["plus_di"]
            )
        )

        # -----------------------------------------------------------
        # 30m
        # -----------------------------------------------------------

        bull30 = (
            m30["ef"]
            > m30["es"]
        )

        bear30 = (
            m30["ef"]
            < m30["es"]
        )

        # -----------------------------------------------------------
        # 15m
        # -----------------------------------------------------------

        long_pb = (
            m15["ef"]
            > m15["es"]
        )

        short_pb = (
            m15["ef"]
            < m15["es"]
        )

        # -----------------------------------------------------------
        # 5m
        # -----------------------------------------------------------

        bull5 = (
            m5["ef"]
            > m5["es"]
        )

        bear5 = (
            m5["ef"]
            < m5["es"]
        )

        # -----------------------------------------------------------
        # 1m TRIGGER
        # -----------------------------------------------------------

        bull1m = (
            (m1["ef"] > m1["es"])
            & (m1["close"] > m1["ef"])
            & (
                m1["close"]
                > m1["prev_high"]
            )
        )

        bear1m = (
            (m1["ef"] < m1["es"])
            & (m1["close"] < m1["ef"])
            & (
                m1["close"]
                < m1["prev_low"]
            )
        )

        # -----------------------------------------------------------
        # ENTRADAS
        #
        # RSI NO ESTÁ AQUÍ.
        # -----------------------------------------------------------

        entry_long = (
            bull1h
            & bull30
            & long_pb
            & bull5
            & bull1m
        )

        entry_short = (
            bear1h
            & bear30
            & short_pb
            & bear5
            & bear1m
        )

        # -----------------------------------------------------------
        # Filtros improved.
        # SOLO afectan a entradas.
        # -----------------------------------------------------------

        mask = np.ones(
            n,
            dtype=bool,
        )

        if cfg.adx_max > 0:

            mask &= (
                h1["adx"]
                <= cfg.adx_max
            )

        if cfg.adx_rising:

            mask &= (
                h1["adx"]
                > h1["adx_prev"]
            )

        if cfg.min_ema_gap_1h_pct > 0:

            mask &= (
                np.abs(
                    h1["ef"]
                    - h1["es"]
                )
                / h1["es"]
                >= cfg.min_ema_gap_1h_pct
            )

        if cfg.min_trigger_range_atr > 0:

            mask &= (
                (
                    m1["high"]
                    - m1["low"]
                )
                >= (
                    cfg.min_trigger_range_atr
                    * m1["atr"]
                )
            )

        blocked = cfg.blocked_hours()

        if blocked:

            hours = (
                f["decision_ns"]
                // 3_600_000_000_000
            ) % 24

            mask &= ~np.isin(
                hours,
                blocked,
            )

    # ---------------------------------------------------------------
    # Valididad MTF.
    # ---------------------------------------------------------------

    valid = (
        h1["ok"]
        & m30["ok"]
        & m15["ok"]
        & m5["ok"]

        & np.isfinite(h1["adx"])
        & np.isfinite(h1["adx_prev"])
        & np.isfinite(h1["plus_di"])
        & np.isfinite(h1["minus_di"])

        & np.isfinite(m15["atr"])
        & np.isfinite(m1["atr"])

        & (
            (~cfg.exit_rsi_enabled)
            | np.isfinite(m1["rsi"])
        )

        & (np.arange(n) >= 1)
    )

    return {

        "valid": valid,

        "bull1h": bull1h,
        "bear1h": bear1h,

        "bull30": bull30,
        "bear30": bear30,

        # RSI NO afecta estas señales salvo los filtros
        # existentes en "mask".
        "entry_long": (
            entry_long
            & mask
        ),

        "entry_short": (
            entry_short
            & mask
        ),

        "atr15": m15["atr"],

        # RSI exclusivamente para salidas.
        "exit_rsi": m1["rsi"],
    }


# ---------------------------------------------------------------------------
# Abstracciones reutilizables
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Construye features MTF."""

    def __init__(
        self,
        config: Config,
    ):
        self.cfg = config

    def build(
        self,
        data: pd.DataFrame,
    ) -> dict:

        return build_features(
            data,
            self.cfg,
        )


class SignalEngine:
    """
    Contrato mínimo para una estrategia.
    """

    def build(
        self,
        features: dict,
    ) -> dict:

        raise NotImplementedError


class Strategy1SignalEngine(
    SignalEngine
):

    def __init__(
        self,
        config: Config,
    ):
        self.cfg = config

    def build(
        self,
        features: dict,
    ) -> dict:

        return build_signals(
            features,
            self.cfg,
        )


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class Strategy1Backtester:

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        verbose: bool = False,
        feature_builder: Optional[
            FeatureBuilder
        ] = None,
        signal_engine: Optional[
            SignalEngine
        ] = None,
    ):

        self.data = data
        self.cfg = config
        self.verbose = verbose

        self.feature_builder = (
            feature_builder
            or FeatureBuilder(config)
        )

        self.signal_engine = (
            signal_engine
            or Strategy1SignalEngine(config)
        )

        feats = self.feature_builder.build(
            data
        )

        self.sig = (
            self.signal_engine.build(feats)
        )

        self.o = data[
            "open"
        ].to_numpy(float)

        self.h = data[
            "high"
        ].to_numpy(float)

        self.l = data[
            "low"
        ].to_numpy(float)

        self.c = data[
            "close"
        ].to_numpy(float)

        self.times = data.index

        self.cash = config.initial_capital

        self.position: Optional[
            Position
        ] = None

        self.pending_signal: Optional[
            dict
        ] = None

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

        bps = (
            self.cfg.slippage_bps
            / 10_000.0
        )

        if side == 1:
            return raw_price * (
                1.0 + bps
            )

        return raw_price * (
            1.0 - bps
        )

    def _fee(
        self,
        notional: float,
    ) -> float:

        return (
            abs(notional)
            * self.cfg.commission_bps
            / 10_000.0
        )

    def _stop_pct_for(
        self,
        i: int,
        ref_price: float,
    ) -> float:

        cfg = self.cfg

        atr = self.sig[
            "atr15"
        ][i]

        if (
            cfg.atr_stop_mult > 0
            and np.isfinite(atr)
            and ref_price > 0
        ):

            pct = (
                cfg.atr_stop_mult
                * atr
                / ref_price
            )

            return min(
                max(
                    pct,
                    cfg.atr_stop_min_pct,
                ),
                cfg.atr_stop_max_pct,
            )

        return cfg.stop_loss_pct

    def _initial_stop_price(
        self,
        side: int,
        entry_price: float,
        stop_pct: float,
    ) -> float:

        if side == 1:
            return (
                entry_price
                * (1.0 - stop_pct)
            )

        return (
            entry_price
            * (1.0 + stop_pct)
        )

    def _estimated_stop_risk_per_unit(
        self,
        side: int,
        entry_price: float,
        stop_pct: float,
    ) -> float:

        stop_price = (
            self._initial_stop_price(
                side,
                entry_price,
                stop_pct,
            )
        )

        stop_exec = (
            self._execution_price(
                stop_price,
                -side,
            )
        )

        price_loss = abs(
            stop_exec
            - entry_price
        )

        fee_rate = (
            self.cfg.commission_bps
            / 10_000.0
        )

        return (
            price_loss
            + (
                entry_price
                + abs(stop_exec)
            )
            * fee_rate
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

        exec_price = (
            self._execution_price(
                price,
                side,
            )
        )

        if (
            stop_pct <= 0
            or self.cash <= 0
            or not np.isfinite(exec_price)
            or exec_price <= 0
        ):
            return

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
            self.cash
            * cfg.risk_pct
            / risk_per_unit
        )

        qty_by_leverage = (
            self.cash
            * cfg.max_leverage
            / exec_price
        )

        qty = min(
            qty_by_risk,
            qty_by_leverage,
        )

        if qty <= 0:
            return

        fee = self._fee(
            exec_price * qty
        )

        self.cash -= fee

        self.position_counter += 1

        stop_price = (
            self._initial_stop_price(
                side,
                exec_price,
                stop_pct,
            )
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
            risk_amount=(
                qty * risk_per_unit
            ),
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

        """
        Cierra un porcentaje de la POSICIÓN ACTUAL.

        Cada ejecución queda registrada como Trade independiente.
        """

        pos = self.position

        if pos is None:
            return

        if (
            not np.isfinite(percent)
            or percent <= 0
            or percent > 100
        ):
            raise ValueError(
                "percent debe estar entre 0 y 100."
            )

        fraction = (
            float(percent)
            / 100.0
        )

        qty = (
            pos.quantity
            * fraction
        )

        if qty <= 0:
            return

        side = pos.side

        exec_price = (
            self._execution_price(
                price,
                -side,
            )
        )

        gross = (
            (
                exec_price
                - pos.entry_price
            )
            * qty
            * side
        )

        exit_fee = self._fee(
            exec_price * qty
        )

        entry_fee_alloc = (
            pos.entry_fee
            * (qty / pos.quantity)
        )

        total_fees = (
            entry_fee_alloc
            + exit_fee
        )

        pnl_net = (
            gross
            - total_fees
        )

        self.cash += (
            gross
            - exit_fee
        )

        self.trade_counter += 1

        notional_entry = (
            pos.entry_price * qty
        )

        return_pct = (
            pnl_net
            / notional_entry
            * 100.0
            if notional_entry != 0
            else 0.0
        )

        self.trades.append(
            Trade(
                trade_id=self.trade_counter,
                position_id=pos.position_id,
                side=(
                    "LONG"
                    if side == 1
                    else "SHORT"
                ),
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

        remaining = (
            pos.quantity - qty
        )

        eps = max(
            1e-12,
            pos.initial_quantity * 1e-12,
        )

        if (
            fraction >= 1.0
            or remaining <= eps
        ):

            self.position = None

        else:

            pos.quantity = remaining

            pos.entry_fee -= (
                entry_fee_alloc
            )

    def _close_fraction(
        self,
        fraction: float,
        price: float,
        time: pd.Timestamp,
        reason: str,
    ):

        self.close_position(
            float(fraction) * 100.0,
            price,
            time,
            reason,
        )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def _check_stop_loss(
        self,
        i: int,
    ) -> Optional[str]:

        """
        Stop inicial fijo.

        No existe trailing.
        """

        pos = self.position

        if (
            pos is None
            or pos.stop_pct <= 0
        ):
            return None

        bo = self.o[i]
        bh = self.h[i]
        bl = self.l[i]

        e = pos.entry_price
        side = pos.side

        if side == 1:

            stop = (
                e
                * (1.0 - pos.stop_pct)
            )

            if bl > stop:
                return None

            fill = (
                bo
                if bo <= stop
                else stop
            )

        else:

            stop = (
                e
                * (1.0 + pos.stop_pct)
            )

            if bh < stop:
                return None

            fill = (
                bo
                if bo >= stop
                else stop
            )

        reason = (
            f"STOP_LOSS_"
            f"{pos.stop_pct * 100:.2f}%"
        )

        qty_before = pos.quantity

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
                f"entrada={e:.8f} | "
                f"stop={stop:.8f} | "
                f"fill={fill:.8f} | "
                f"qty={qty_before} | "
                f"{reason}",
                flush=True,
            )

        return reason

    # ------------------------------------------------------------------
    # Ejecutar señal pendiente
    # ------------------------------------------------------------------

    def _execute_pending(
        self,
        i: int,
    ):

        sig = self.pending_signal

        if sig is None:
            return

        self.pending_signal = None

        bar_open = self.o[i]
        ts = self.times[i]

        if sig["action"] == "BUY":

            self._open(
                1,
                bar_open,
                ts,
                sig["stop_pct"],
            )

        elif sig["action"] == "SELL":

            self._open(
                -1,
                bar_open,
                ts,
                sig["stop_pct"],
            )

        elif sig["action"] == "EXIT":

            self.close_position(
                float(sig["percent"]),
                bar_open,
                ts,
                sig["reason"],
            )

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    def _mark(
        self,
        time: pd.Timestamp,
        close: float,
    ):

        """
        Marca equity como valor de liquidación neto.
        """

        pos = self.position

        equity = self.cash

        if pos is not None:

            liquidation_price = (
                self._execution_price(
                    close,
                    -pos.side,
                )
            )

            gross = (
                (
                    liquidation_price
                    - pos.entry_price
                )
                * pos.quantity
                * pos.side
            )

            exit_fee = self._fee(
                liquidation_price
                * pos.quantity
            )

            equity += (
                gross
                - exit_fee
            )

        self.eq_time.append(time)
        self.eq_cash.append(self.cash)
        self.eq_equity.append(equity)

        self.eq_side.append(
            "FLAT"
            if pos is None
            else (
                "LONG"
                if pos.side == 1
                else "SHORT"
            )
        )

        self.eq_qty.append(
            0.0
            if pos is None
            else pos.quantity
        )

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def run(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        cfg = self.cfg
        sig = self.sig

        n = len(self.data)

        valid = sig["valid"]
        bull1h = sig["bull1h"]
        bear1h = sig["bear1h"]

        bull30 = sig["bull30"]
        bear30 = sig["bear30"]

        entry_long = sig["entry_long"]
        entry_short = sig["entry_short"]

        exit_rsi = sig["exit_rsi"]

        started = time.perf_counter()

        last_log = -10

        cooldown_until: Optional[
            pd.Timestamp
        ] = None

        one_min = pd.Timedelta(
            minutes=1
        )

        print(
            f"[BACKTEST] {n:,} velas 1m",
            flush=True,
        )

        for i in range(n):

            # -------------------------------------------------------
            # 1) Ejecutar señal del cierre anterior.
            # -------------------------------------------------------

            if self.pending_signal is not None:

                self._execute_pending(i)

            # -------------------------------------------------------
            # 2) Stop intrabar.
            #    Siempre tiene prioridad.
            # -------------------------------------------------------

            stop_reason = (
                self._check_stop_loss(i)
            )

            stop_hit = (
                stop_reason is not None
            )

            if (
                stop_hit
                and stop_reason.startswith(
                    "STOP_LOSS"
                )
            ):

                cooldown_until = (
                    self.times[i]
                    + pd.Timedelta(
                        minutes=1
                        + cfg.cooldown_min
                    )
                )

            # -------------------------------------------------------
            # 3) Señal al cierre de esta vela.
            #    Se ejecuta en la próxima apertura.
            # -------------------------------------------------------

            new_sig = None

            if (
                not stop_hit
                and valid[i]
            ):

                pos = self.position

                if pos is not None:

                    side = pos.side
                    rsi = exit_rsi[i]

                    # ------------------------------------------------
                    # PRIORIDAD 1:
                    # 1H reversal -> cierre total
                    # ------------------------------------------------

                    if (
                        (
                            side == 1
                            and bear1h[i]
                        )
                        or
                        (
                            side == -1
                            and bull1h[i]
                        )
                    ):

                        new_sig = {
                            "action": "EXIT",
                            "percent": 100.0,
                            "reason": "1H_REVERSAL",
                        }

                    # ------------------------------------------------
                    # PRIORIDAD 2:
                    # 30m reversal -> cierre parcial
                    #
                    # Mantiene la lógica original:
                    # solo se hace una vez sobre la posición inicial.
                    # ------------------------------------------------

                    elif (
                        pos.quantity
                        >= pos.initial_quantity
                        - 1e-12
                        and
                        (
                            (
                                side == 1
                                and bear30[i]
                            )
                            or
                            (
                                side == -1
                                and bull30[i]
                            )
                        )
                    ):

                        new_sig = {
                            "action": "EXIT",
                            "percent": cfg.close_on_30m_pct,
                            "reason": "30M_PARTIAL_CLOSE",
                        }

                    # ------------------------------------------------
                    # PRIORIDAD 3:
                    # RSI -> cierre TOTAL
                    #
                    # IMPORTANTE:
                    # el RSI SOLO puede cerrar una posición.
                    # NUNCA genera entradas.
                    # ------------------------------------------------

                    elif (
                        cfg.exit_rsi_enabled
                        and np.isfinite(rsi)
                    ):

                        # LONG:
                        # RSI >= 70 por defecto.
                        if (
                            side == 1
                            and rsi
                            >= cfg.exit_rsi_long
                        ):

                            new_sig = {
                                "action": "EXIT",
                                "percent": 100.0,
                                "reason": (
                                    f"RSI_EXIT_LONG_"
                                    f"{rsi:.2f}"
                                ),
                            }

                        # SHORT:
                        # RSI <= 30 por defecto.
                        elif (
                            side == -1
                            and rsi
                            <= cfg.exit_rsi_short
                        ):

                            new_sig = {
                                "action": "EXIT",
                                "percent": 100.0,
                                "reason": (
                                    f"RSI_EXIT_SHORT_"
                                    f"{rsi:.2f}"
                                ),
                            }

                # ---------------------------------------------------
                # No hay posición:
                # SOLO aquí se buscan entradas.
                #
                # El RSI no se consulta.
                # ---------------------------------------------------

                elif (
                    cooldown_until is None
                    or self.times[i]
                    >= cooldown_until
                ):

                    if entry_long[i]:

                        new_sig = {
                            "action": "BUY",
                            "reason": "MTF_LONG",
                            "stop_pct": (
                                self._stop_pct_for(
                                    i,
                                    self.c[i],
                                )
                            ),
                        }

                    elif entry_short[i]:

                        new_sig = {
                            "action": "SELL",
                            "reason": "MTF_SHORT",
                            "stop_pct": (
                                self._stop_pct_for(
                                    i,
                                    self.c[i],
                                )
                            ),
                        }

            if new_sig is not None:
                self.pending_signal = new_sig

            # -------------------------------------------------------
            # Equity
            # -------------------------------------------------------

            self._mark(
                self.times[i] + one_min,
                self.c[i],
            )

            pct = int(
                (i + 1)
                / n
                * 100
            )

            if (
                pct >= last_log + 10
                or i == n - 1
            ):

                elapsed = (
                    time.perf_counter()
                    - started
                )

                rate = (
                    (i + 1) / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"[BACKTEST] "
                    f"{pct:3d}% | "
                    f"{rate:,.0f} velas/s | "
                    f"{elapsed:.1f}s | "
                    f"tramos "
                    f"{len(self.trades)}",
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
                self.times[-1]
                + one_min
            )

            self.close_position(
                100.0,
                self.c[-1],
                last_ts,
                "END_OF_DATA",
            )

            self._mark(
                last_ts,
                self.c[-1],
            )

        trades = (
            pd.DataFrame(
                [
                    asdict(t)
                    for t in self.trades
                ]
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
# Métricas
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
        .groupby(
            "position_id",
            sort=True,
        )
        .agg(
            side=("side", "first"),
            entry_time=(
                "entry_time",
                "first",
            ),
            exit_time=(
                "exit_time",
                "last",
            ),
            pnl_net=(
                "pnl_net",
                "sum",
            ),
            fees=(
                "fees",
                "sum",
            ),
            risk=(
                "position_risk",
                "first",
            ),
            legs=(
                "trade_id",
                "count",
            ),
            reason=(
                "reason",
                "last",
            ),
        )
        .reset_index()
    )

    pos["r_multiple"] = np.where(
        pos["risk"] > 0,
        pos["pnl_net"]
        / pos["risk"],
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

    pnl = pos[
        "pnl_net"
    ].astype(float)

    gross_profit = float(
        pnl[pnl > 0].sum()
    )

    gross_loss = float(
        -pnl[pnl < 0].sum()
    )

    streak = 0
    best_streak = 0

    for v in pnl.to_numpy():

        streak = (
            streak + 1
            if v < 0
            else 0
        )

        best_streak = max(
            best_streak,
            streak,
        )

    r = pos[
        "r_multiple"
    ].dropna()

    return {
        "positions": int(len(pnl)),
        "wins": int(
            (pnl > 0).sum()
        ),
        "losses": int(
            (pnl < 0).sum()
        ),
        "win_rate_pct": float(
            (pnl > 0).mean()
            * 100.0
        ),
        "profit_factor": (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else None
        ),
        "net_pnl": float(
            pnl.sum()
        ),
        "avg_pnl": float(
            pnl.mean()
        ),
        "expectancy_r": (
            float(r.mean())
            if len(r)
            else None
        ),
        "max_consecutive_losses": int(
            best_streak
        ),
    }


def calculate_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
) -> dict:

    pos = positions_from_trades(
        trades
    )

    stats = position_stats(pos)

    if equity.empty:

        final_equity = initial_capital

        max_dd = 0.0
        sharpe = 0.0
        sortino = 0.0

        min_equity = initial_capital

        min_vs_capital_pct = 0.0

        dd_peak_equity = (
            initial_capital
        )

        dd_trough_equity = (
            initial_capital
        )

        dd_peak_vs_capital_pct = 0.0

        dd_trough_vs_capital_pct = 0.0

    else:

        eq = equity[
            "equity"
        ].astype(float)

        final_equity = float(
            eq.iloc[-1]
        )

        peak = eq.cummax()

        dd = (
            eq / peak - 1.0
        ) * 100.0

        max_dd = float(
            dd.min()
        )

        min_equity = float(
            eq.min()
        )

        min_vs_capital_pct = (
            float(
                (
                    eq
                    / initial_capital
                    - 1.0
                ).min()
                * 100.0
            )
            if initial_capital
            else 0.0
        )

        trough_pos = int(
            dd.idxmin()
        )

        dd_trough_equity = float(
            eq.loc[trough_pos]
        )

        dd_peak_equity = float(
            peak.loc[trough_pos]
        )

        dd_peak_vs_capital_pct = (
            float(
                (
                    dd_peak_equity
                    / initial_capital
                    - 1.0
                )
                * 100.0
            )
            if initial_capital
            else 0.0
        )

        dd_trough_vs_capital_pct = (
            float(
                (
                    dd_trough_equity
                    / initial_capital
                    - 1.0
                )
                * 100.0
            )
            if initial_capital
            else 0.0
        )

        daily = (
            equity
            .set_index("timestamp")[
                "equity"
            ]
            .resample("1D")
            .last()
            .dropna()
        )

        rets = (
            daily
            .pct_change()
            .dropna()
        )

        sharpe = 0.0
        sortino = 0.0

        vol = (
            rets.std(ddof=1)
            if len(rets) > 1
            else 0.0
        )

        if (
            len(rets) > 1
            and np.isfinite(vol)
            and vol > 0
        ):

            sharpe = float(
                rets.mean()
                / vol
                * math.sqrt(365)
            )

        downside_sq = (
            np.minimum(
                rets.to_numpy(float),
                0.0,
            )
            ** 2
        )

        downside_dev = (
            math.sqrt(
                float(
                    np.mean(
                        downside_sq
                    )
                )
            )
            if len(rets)
            else 0.0
        )

        if downside_dev > 0:

            sortino = float(
                rets.mean()
                / downside_dev
                * math.sqrt(365)
            )

    net = (
        final_equity
        - initial_capital
    )

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "net_pnl": net,

        "return_pct": (
            net
            / initial_capital
            * 100.0
            if initial_capital
            else 0.0
        ),

        "max_drawdown_pct": max_dd,

        "max_dd_peak_equity": (
            dd_peak_equity
        ),

        "max_dd_trough_equity": (
            dd_trough_equity
        ),

        "max_dd_peak_vs_capital_pct": (
            dd_peak_vs_capital_pct
        ),

        "max_dd_trough_vs_capital_pct": (
            dd_trough_vs_capital_pct
        ),

        "min_equity": min_equity,

        "min_equity_vs_capital_pct": (
            min_vs_capital_pct
        ),

        "return_over_maxdd": (
            (
                net
                / initial_capital
                * 100.0
            )
            / abs(max_dd)
            if (
                initial_capital
                and max_dd < 0
            )
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

    if (
        trades.empty
        or "exit_time" not in trades.columns
    ):
        return pd.DataFrame(
            columns=columns
        )

    df = (
        trades
        .sort_values(
            [
                "exit_time",
                "trade_id",
            ],
            kind="stable",
        )
        .copy()
    )

    df["cum_pnl"] = (
        df["pnl_net"]
        .astype(float)
        .cumsum()
    )

    df["equity"] = (
        initial_capital
        + df["cum_pnl"]
    )

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
        .rename(
            columns={
                "exit_time": "timestamp"
            }
        )
    )

    return out.reset_index(
        drop=True
    )[columns]


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
        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    x = pd.to_datetime(
        curve["timestamp"],
        utc=True,
    )

    equity = curve[
        "equity"
    ].to_numpy(float)

    pnl = curve[
        "pnl_net"
    ].to_numpy(float)

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

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
        label=(
            f"Capital inicial "
            f"({initial_capital:,.2f})"
        ),
    )

    peak = np.maximum.accumulate(
        equity
    )

    dd = (
        equity / peak - 1.0
    ) * 100.0

    dd_text = (
        f" | max DD {dd.min():.2f}%"
        if len(dd)
        else ""
    )

    if initial_capital:

        below_pct = (
            equity / initial_capital
            - 1.0
        ) * 100.0

        min_below_pct = float(
            np.min(below_pct)
        )

    else:

        min_below_pct = 0.0

    min_below_equity = (
        initial_capital
        * (
            1.0
            + min_below_pct / 100.0
        )
    )

    ax.axhline(
        min_below_equity,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.1,
        label=(
            f"Min equity bajo capital: "
            f"{min_below_pct:.2f}% "
            f"({min_below_equity:,.2f})"
        ),
    )

    if len(dd):

        trough_i = int(
            np.argmin(dd)
        )

        trough_equity = float(
            equity[trough_i]
        )

        peak_equity = float(
            peak[trough_i]
        )

        peak_pct = (
            (
                peak_equity
                / initial_capital
                - 1.0
            )
            * 100.0
            if initial_capital
            else 0.0
        )

        ax.axhline(
            trough_equity,
            color="#9467bd",
            linestyle="-.",
            linewidth=1.1,
            label=(
                f"Valle max DD: "
                f"{dd.min():.2f}% "
                f"desde pico "
                f"{peak_equity:,.2f} "
                f"[{peak_pct:+.2f}%] "
                f"-> "
                f"{trough_equity:,.2f}"
            ),
        )

        ax.plot(
            x[trough_i],
            trough_equity,
            marker="v",
            color="#9467bd",
            markersize=7,
            zorder=4,
            linestyle="none",
        )

    ax.set_title(
        f"{title} | "
        f"{len(curve):,} cierres"
        f"{dd_text} | "
        f"min bajo capital "
        f"{min_below_pct:.2f}%"
    )

    ax.set_xlabel(
        "Fecha (UTC)"
    )

    ax.set_ylabel(
        "Equity"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best",
        fontsize=8,
    )

    locator = (
        mdates.AutoDateLocator()
    )

    ax.xaxis.set_major_locator(
        locator
    )

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            locator
        )
    )

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=dpi,
    )

    plt.close(fig)

    print(
        f"[PLOT] Curva de equity guardada en: "
        f"{out}",
        flush=True,
    )

    return out


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _fmt(
    v,
    spec=".3f",
):

    return (
        "N/A"
        if v is None
        else format(v, spec)
    )


def print_report(
    m: dict,
    cfg: Config,
    data: pd.DataFrame,
):

    print(
        "\n" + "=" * 72
    )

    print(
        "STRATEGY 1 — IMPROVED + RSI EXIT"
    )

    print(
        "=" * 72
    )

    print(
        f"Periodo         : "
        f"{data.index[0]} -> "
        f"{data.index[-1]}"
    )

    print(
        f"Capital         : "
        f"{m['initial_capital']:,.2f} -> "
        f"{m['final_equity']:,.2f}"
    )

    print(
        f"PnL neto        : "
        f"{m['net_pnl']:,.2f} "
        f"({m['return_pct']:.2f}%)"
    )

    print(
        f"Max drawdown    : "
        f"{m['max_drawdown_pct']:.2f}% "
        f"(pico "
        f"{m['max_dd_peak_equity']:,.2f} "
        f"[{m['max_dd_peak_vs_capital_pct']:+.2f}%] "
        f"-> valle "
        f"{m['max_dd_trough_equity']:,.2f} "
        f"[{m['max_dd_trough_vs_capital_pct']:+.2f}%])"
    )

    print(
        f"Equity mínimo   : "
        f"{m['min_equity']:,.2f} "
        f"({m['min_equity_vs_capital_pct']:.2f}% "
        f"vs capital inicial)"
    )

    print(
        f"Retorno / MaxDD : "
        f"{_fmt(m['return_over_maxdd'], '.2f')}"
    )

    print(
        f"Sharpe / Sortino: "
        f"{m['sharpe_daily_365']:.2f} / "
        f"{m['sortino_daily_365']:.2f}"
    )

    print(
        f"Posiciones      : "
        f"{m['positions']:,} "
        f"(tramos: {m['legs']:,})"
    )

    print(
        f"Win rate        : "
        f"{m['win_rate_pct']:.2f}% "
        f"({m['wins']} ganadoras / "
        f"{m['losses']} perdedoras)"
    )

    print(
        f"Profit factor   : "
        f"{_fmt(m['profit_factor'])}"
    )

    print(
        f"Expectancy (R)  : "
        f"{_fmt(m['expectancy_r'])}"
    )

    print(
        f"Avg PnL/posición: "
        f"{m['avg_pnl']:,.4f}"
    )

    print(
        f"Racha pérdidas  : "
        f"{m['max_consecutive_losses']}"
    )

    print(
        f"Costos          : "
        f"{cfg.commission_bps} bps comisión | "
        f"{cfg.slippage_bps} bps slippage"
    )

    print(
        f"Sizing / stop   : "
        f"riesgo {cfg.risk_pct:.2%} "
        f"(lev máx {cfg.max_leverage}x) / "
        f"{cfg.atr_stop_mult}xATR15m "
        f"[{cfg.atr_stop_min_pct:.2%}-"
        f"{cfg.atr_stop_max_pct:.2%}]"
    )

    print(
        "Trailing        : OFF"
    )

    print(
        f"Filtros entrada : "
        f"cooldown={cfg.cooldown_min}m "
        f"adx_max={cfg.adx_max} "
        f"adx_rising={cfg.adx_rising} "
        f"gap1h={cfg.min_ema_gap_1h_pct} "
        f"rango1m={cfg.min_trigger_range_atr}xATR "
        f"horas_bloq={cfg.block_hours_utc or '-'}"
    )

    print(
        f"Salida RSI      : "
        f"{'ON' if cfg.exit_rsi_enabled else 'OFF'} "
        f"RSI({cfg.exit_rsi_len}) | "
        f"LONG >= {cfg.exit_rsi_long:.1f} | "
        f"SHORT <= {cfg.exit_rsi_short:.1f}"
    )

    print(
        "=" * 72
    )


def oos_split(
    trades: pd.DataFrame,
    oos_start: str,
) -> Optional[dict]:

    pos = positions_from_trades(
        trades
    )

    if pos.empty:
        return None

    cut = pd.Timestamp(
        oos_start
    )

    if cut.tzinfo is None:

        cut = cut.tz_localize(
            "UTC"
        )

    else:

        cut = cut.tz_convert(
            "UTC"
        )

    return {
        "cut": str(cut),

        "before_cut": position_stats(
            pos[
                pos["entry_time"] < cut
            ]
        ),

        "from_cut": position_stats(
            pos[
                pos["entry_time"] >= cut
            ]
        ),
    }


def print_oos_split(
    trades: pd.DataFrame,
    oos_start: str,
):

    split = oos_split(
        trades,
        oos_start,
    )

    if split is None:
        return

    print(
        f"\nSplit por fecha de entrada "
        f"(corte {oos_start}):"
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


# ---------------------------------------------------------------------------
# Reporte en disco
# ---------------------------------------------------------------------------

def side_stats(
    trades: pd.DataFrame,
) -> dict:

    """Métricas separadas por lado."""

    out = {}

    for side in (
        "LONG",
        "SHORT",
    ):

        pos = positions_from_trades(
            trades[
                trades["side"] == side
            ]
            if len(trades)
            else trades
        )

        out[side] = position_stats(
            pos
        )

    return out


def exit_reason_breakdown(
    trades: pd.DataFrame,
) -> dict:

    """
    Tramos y PnL por motivo de cierre.

    Esto incluye los nuevos:
        RSI_EXIT_LONG_xx
        RSI_EXIT_SHORT_xx
    """

    if trades.empty:
        return {}

    grouped = (
        trades
        .groupby(
            "reason",
            sort=False,
        )
        .agg(
            legs=("trade_id", "count"),
            positions=(
                "position_id",
                "nunique",
            ),
            pnl_net=(
                "pnl_net",
                "sum",
            ),
        )
    )

    return {
        str(reason): {
            "legs": int(
                row["legs"]
            ),
            "positions": int(
                row["positions"]
            ),
            "pnl_net": float(
                row["pnl_net"]
            ),
        }
        for reason, row
        in grouped.iterrows()
    }


def dataset_quality(
    data: pd.DataFrame,
) -> dict:

    """Controles básicos para detectar huecos de 1m."""

    if len(data) < 2:

        return {
            "bars": int(len(data)),
            "gaps_gt_1m": 0,
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

    gaps = delta[
        delta > 1.0
    ]

    return {
        "bars": int(len(data)),
        "gaps_gt_1m": int(len(gaps)),
        "max_gap_minutes": (
            float(gaps.max())
            if len(gaps)
            else 1.0
        ),
    }


def build_report(
    cfg: Config,
    data: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict,
    source: Optional[str] = None,
    oos_start: Optional[str] = None,
) -> dict:

    """Reporte completo y serializable."""

    report = {

        "strategy": "strategy1",

        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "dataset": {

            "source": source,

            "bars_1m": int(
                len(data)
            ),

            "start": (
                data.index[0]
                .isoformat()
            ),

            "end": (
                data.index[-1]
                .isoformat()
            ),

            "quality": (
                dataset_quality(data)
            ),
        },

        "config": asdict(cfg),

        "metrics": metrics,

        "by_side": side_stats(
            trades
        ),

        "exit_reasons": (
            exit_reason_breakdown(
                trades
            )
        ),
    }

    if oos_start:

        report["oos_split"] = (
            oos_split(
                trades,
                oos_start,
            )
        )

    return _jsonable(
        report
    )


def write_outputs(
    output_dir: str | Path,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    curve: pd.DataFrame,
    report: dict,
    write_equity_csv: bool = False,
) -> dict[str, Path]:

    """Escribe los artefactos de la corrida."""

    out = Path(
        output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        out / "trades.csv",
        index=False,
    )

    positions_from_trades(
        trades
    ).to_csv(
        out / "positions.csv",
        index=False,
    )

    curve.to_csv(
        out / "equity_curve.csv",
        index=False,
    )

    if write_equity_csv:

        equity.to_csv(
            out / "equity.csv",
            index=False,
        )

    with open(
        out / "report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    with open(
        out / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            _jsonable(
                report["metrics"]
            ),
            f,
            indent=2,
        )

    with open(
        out / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report["config"],
            f,
            indent=2,
        )

    written = [
        "trades.csv",
        "report.json",
        "metrics.json",
        "config.json",
        "positions.csv",
        "equity_curve.csv",
    ]

    if write_equity_csv:
        written.append(
            "equity.csv"
        )

    print(
        f"[OUT] {out.resolve()} -> "
        f"{', '.join(written)}",
        flush=True,
    )

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():

    d = Config()

    p = argparse.ArgumentParser(
        description=(
            "Backtester Strategy1 IMPROVED "
            "sobre OHLCV 1m Parquet."
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
        "--close-on-30m",
        type=float,
        default=d.close_on_30m_pct,
        help=(
            "Porcentaje de la posición RESTANTE "
            "a cerrar ante contra-señal 30m."
        ),
    )

    p.add_argument(
        "--adx-threshold",
        type=float,
        default=d.adx_threshold,
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
        "--stop-loss-pct",
        type=float,
        default=d.stop_loss_pct,
        help=(
            "Stop fijo de respaldo si no hay ATR "
            "(0.006 = 0.6%%)."
        ),
    )

    p.add_argument(
        "--no-force-close",
        action="store_true",
    )

    p.add_argument(
        "--oos-start",
        default=None,
        help=(
            "Fecha YYYY-MM-DD para separar "
            "el reporte antes/después."
        ),
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
        help=(
            f"Indica que el timestamp de 1m "
            f"es cierre, no apertura "
            f"(default: {d.timestamp_is_close})."
        ),
    )

    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Directorio de salida "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )

    p.add_argument(
        "--no-output",
        action="store_true",
        help=(
            "No escribe trades.csv ni report.json."
        ),
    )

    p.add_argument(
        "--equity-csv",
        action="store_true",
        help=(
            "Escribe además equity.csv."
        ),
    )

    p.add_argument(
        "--plot",
        default=None,
        metavar="PATH",
        help=(
            "Ruta del gráfico de equity."
        ),
    )

    p.add_argument(
        "--no-plot",
        action="store_true",
        help=(
            "No genera el gráfico."
        ),
    )

    p.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
    )

    # ---------------------------------------------------------------
    # Parámetros improved.
    # ---------------------------------------------------------------

    for key in CLI_KEYS:

        flag = (
            "--"
            + key.replace("_", "-")
        )

        default_value = getattr(
            d,
            key,
        )

        if isinstance(
            default_value,
            bool,
        ):

            p.add_argument(
                flag,
                action=(
                    argparse.BooleanOptionalAction
                ),
                default=None,
                help=(
                    f"(default: "
                    f"{default_value})"
                ),
            )

        else:

            p.add_argument(
                flag,
                type=type(
                    default_value
                ),
                default=None,
                help=(
                    f"(default: "
                    f"{default_value!r})"
                ),
            )

    return p.parse_args()


def make_config(args) -> Config:

    overrides = {
        k: getattr(args, k)
        for k in CLI_KEYS
        if getattr(args, k) is not None
    }

    return Config(

        initial_capital=args.capital,

        close_on_30m_pct=(
            args.close_on_30m
        ),

        adx_threshold=(
            args.adx_threshold
        ),

        commission_bps=(
            args.commission_bps
        ),

        slippage_bps=(
            args.slippage_bps
        ),

        stop_loss_pct=(
            args.stop_loss_pct
        ),

        force_close_at_end=(
            not args.no_force_close
        ),

        timestamp_is_close=(
            Config().timestamp_is_close
            if args.timestamp_is_close is None
            else args.timestamp_is_close
        ),

        **overrides,
    )


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _jsonable(obj):

    if isinstance(obj, dict):

        return {
            k: _jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(
        obj,
        (np.floating, float),
    ):

        return (
            None
            if not math.isfinite(
                float(obj)
            )
            else float(obj)
        )

    if isinstance(
        obj,
        np.integer,
    ):

        return int(obj)

    if isinstance(
        obj,
        (pd.Timestamp, datetime),
    ):

        return obj.isoformat()

    if isinstance(
        obj,
        Path,
    ):

        return str(obj)

    if isinstance(
        obj,
        (list, tuple),
    ):

        return [
            _jsonable(v)
            for v in obj
        ]

    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    cfg = make_config(
        args
    )

    data = load_parquet(
        args.parquet
    )

    if len(data) < 100:

        raise ValueError(
            "El dataset tiene muy pocas "
            "velas 1m para este backtest."
        )

    engine = Strategy1Backtester(
        data,
        cfg,
        verbose=args.verbose,
    )

    trades, equity = engine.run()

    metrics = calculate_metrics(
        trades,
        equity,
        cfg.initial_capital,
    )

    print_report(
        metrics,
        cfg,
        data,
    )

    if args.oos_start:

        print_oos_split(
            trades,
            args.oos_start,
        )

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
            source=str(
                args.parquet
            ),
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
                Path(
                    args.output_dir or "."
                )
                / "equity_curve.png"
            )
        )

        plot_equity(
            curve,
            plot_path,
            cfg.initial_capital,
            title=(
                "Strategy1 improved + RSI exit | "
                f"{metrics['initial_capital']:,.2f} -> "
                f"{metrics['final_equity']:,.2f}"
            ),
            dpi=args.plot_dpi,
        )


if __name__ == "__main__":
    main()

