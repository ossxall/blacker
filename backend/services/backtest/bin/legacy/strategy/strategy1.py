#!/usr/bin/env python3
"""
Backtesting Strategy1 — IMPROVED
================================

Dataset de entrada: OHLCV de 1 minuto en .parquet.

Reglas de la estrategia (Strategy1, sin cambios):
    1H  -> tendencia: EMA20/EMA50 + ADX14 >= 25 + DI
    30m -> confirmación: EMA20 vs EMA50
    15m -> pullback: estructura + cierre contra EMA20
    5m  -> setup: EMA20 vs EMA50
    1m  -> trigger: estructura + cierre sobre/bajo EMA20 +
           ruptura del máximo/mínimo de la vela anterior

Valores default reales (compatibilidad con la versión existente):
    - EMA 21/55 en todos los timeframes.
    - ADX 14/14, umbral 23.
    - Riesgo 1% del capital, apalancamiento máximo 1x.
    - Stop 3.0 x ATR(14) de 15m, acotado entre 0.4% y 1.5%.
    - Scale-out 99% y trailing 2.5 x ATR(15m) sobre el resto.
    - Cooldown 0 min; ADX máximo 40 y ADX creciente.
    - Gap mínimo EMA 1H 0.1%; rango mínimo 1m 0 ATR.

Arquitectura reutilizable:
    Config -> DataFeed -> FeatureBuilder -> SignalEngine -> ExecutionEngine
    -> Portfolio/Equity -> Metrics. Los bots posteriores pueden sustituir
    SignalEngine sin duplicar carga, indicadores, ejecución o reporting.

Salidas:
    Stop inicial (ATR)      -> cierre total.
    1H contra la posición   -> cierre total.
    30m contra la posición  -> scale-out parcial (50%) y el stop del resto
                               sube a breakeven; después rige el trailing.

Convención anti-lookahead:
    Las señales usan solo velas cerradas y se ejecutan en la apertura de la
    siguiente vela 1m. El trailing se actualiza al cierre de cada vela y rige
    desde la siguiente. Si hay gap a través del stop, el fill es el open.

Fidelidad: EMA, ADX y ATR se calculan vectorizados (ewm con adjust=False),
equivalentes a la recursión original (seed = primer valor, sin SMA; ADX
visible desde la vela dilen + adxlen - 2).

Dependencias:  pip install "pandas>=2" pyarrow numpy matplotlib

Ejemplo:
    python backtest_strategy1_improved.py data.parquet --capital 100000 \
        --commission-bps 2 --slippage-bps 1 --oos-start 2026-01-01 \
        --output-dir ./backtest_out

Salidas en disco (por defecto en ./backtest_out, o en --output-dir):
    trades.csv        un tramo por cierre (incluye scale-out y resto)
    report.json       reporte completo: dataset, config, métricas, por lado,
                      por motivo de cierre y split in-sample/out-of-sample
    metrics.json      solo métricas (compatibilidad)
    config.json       parámetros efectivos de la corrida
    positions.csv     una fila por posición (tramos agregados)
    equity_curve.csv  equity reconstruida, un punto por cierre
    equity.csv        marca a cada vela 1m (opcional: --equity-csv, es pesado)
    equity_curve.png  gráfico de la curva de equity (--no-plot lo desactiva)
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

    scale_out_fraction: float = 0.5

    initial_capital: float = 10_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    # Stop fijo de respaldo (si no hay ATR disponible) y para el cálculo base.
    stop_loss_pct: float = 0.005
    force_close_at_end: bool = True

    # --- parámetros de la versión improved ---
    risk_pct: float = 0.01              # riesgo por operación (fracción del capital)
    max_leverage: float = 1.0
    atr_stop_mult: float = 3.0          # stop = k * ATR(15m)
    atr_stop_min_pct: float = 0.004     # tope inferior del stop (fracción)
    atr_stop_max_pct: float = 0.015     # tope superior del stop (fracción)
    trail_atr_mult: float = 2.5         # trailing tras el scale-out (0 = off)
    cooldown_min: int = 0              # minutos sin entrar tras un stop inicial
    adx_max: float = 40.0               # techo de ADX 1H (0 = off)
    adx_rising: bool = True             # ADX 1H > ADX de la vela 1H previa
    min_ema_gap_1h_pct: float = 0.001  # |EMA20-EMA50|/EMA50 mínimo en 1H
    min_trigger_range_atr: float = 0.5  # rango mínimo de la vela 1m en ATR(1m)
    block_hours_utc: str = ""           # horas UTC sin entradas, p.ej. "0,1,2"
    timestamp_is_close: bool = False   # False: timestamp = apertura de vela 1m

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
            raise ValueError("commission_bps y slippage_bps no pueden ser negativos.")
        if not (0.0 < self.scale_out_fraction < 1.0):
            raise ValueError(
                "scale_out_fraction debe estar estrictamente entre 0 y 1 "
                f"(recibido: {self.scale_out_fraction})."
            )
        if not (0.0 < self.risk_pct <= 0.2):
            raise ValueError("risk_pct debe estar entre 0 (excl.) y 0.2.")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage debe ser > 0.")
        if self.atr_stop_mult < 0 or self.trail_atr_mult < 0:
            raise ValueError("Los multiplicadores de ATR no pueden ser negativos.")
        if self.atr_stop_min_pct < 0 or self.atr_stop_max_pct < 0:
            raise ValueError("Los límites de stop ATR no pueden ser negativos.")
        if self.atr_stop_min_pct > self.atr_stop_max_pct:
            raise ValueError("atr_stop_min_pct no puede superar atr_stop_max_pct.")
        if self.stop_loss_pct <= 0 and self.atr_stop_mult <= 0:
            raise ValueError("Se requiere un stop (stop_loss_pct o atr_stop_mult).")
        self.blocked_hours()  # valida el formato

    def blocked_hours(self) -> list[int]:
        txt = (self.block_hours_utc or "").strip()
        if not txt:
            return []
        try:
            hours = [int(x) for x in txt.split(",") if x.strip() != ""]
        except ValueError:
            raise ValueError(f"block_hours_utc inválido: {txt!r}")
        if any(h < 0 or h > 23 for h in hours):
            raise ValueError("block_hours_utc debe contener horas entre 0 y 23.")
        return hours


# Parámetros expuestos por CLI (default None = usa el valor de Config).
CLI_KEYS = [
    "risk_pct", "max_leverage", "atr_stop_mult", "atr_stop_min_pct",
    "atr_stop_max_pct", "trail_atr_mult", "cooldown_min", "adx_max",
    "adx_rising", "min_ema_gap_1h_pct", "min_trigger_range_atr",
    "block_hours_utc",
]

DEFAULT_OUTPUT_DIR = "backtest_out"


@dataclass
class Position:
    position_id: int
    side: int                  # +1 LONG, -1 SHORT
    quantity: float
    initial_quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_fee: float
    stop_pct: float            # stop inicial efectivo (fracción)
    risk_amount: float         # capital en riesgo al abrir (qty*entry*stop_pct)

    stop_at_entry: bool = False        # stop en breakeven tras scale-out
    trailing: bool = False             # trailing activo (tras scale-out)
    extreme: Optional[float] = None    # máx (long) / mín (short) desde el scale-out
    trail_level: Optional[float] = None


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
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TRADE_COLUMNS})


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y localiza OHLCV."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    aliases = {
        "timestamp": ["timestamp", "open_time", "datetime", "date", "time", "ts"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v"],
    }
    result = {}
    for canonical, candidates in aliases.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is not None:
            result[canonical] = found

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [x for x in required if x not in result]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    out = df.rename(columns={v: k for k, v in result.items()}).copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0

    # Timestamp en UTC. Si es numérico se detecta la unidad (epoch s/ms/us/ns).
    raw_ts = out["timestamp"]
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
            ts = pd.to_datetime(numeric_ts, unit=unit, utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")

    if ts.isna().any():
        raise ValueError("Hay timestamps inválidos en el dataset.")

    out["timestamp"] = ts
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")
    out = out.set_index("timestamp")
    return out[["open", "high", "low", "close", "volume"]]


def load_parquet(path: str | Path) -> pd.DataFrame:
    print(f"[LOAD] Leyendo Parquet: {path}", flush=True)
    t0 = time.perf_counter()
    df = pd.read_parquet(path)
    print(f"[LOAD] {len(df):,} filas en {time.perf_counter() - t0:.2f}s", flush=True)
    out = normalize_columns(df)
    print(
        f"[LOAD] {len(out):,} velas | {out.index.min()} -> {out.index.max()}",
        flush=True,
    )
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="left", closed="left", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    """Epoch en nanosegundos (independiente de la resolución del índice)."""
    return idx.as_unit("ns").asi8


# ---------------------------------------------------------------------------
# Indicadores vectorizados
# ---------------------------------------------------------------------------

def ema_closed(values: pd.Series, period: int) -> pd.Series:
    """EMA con seed = primer valor (sin SMA)."""
    if period <= 0:
        raise ValueError("EMA period must be > 0")
    return values.ewm(alpha=2.0 / (period + 1.0), adjust=False).mean()


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    if len(tr):
        tr[0] = high[0] - low[0]  # el primer TR es high-low
    return tr


def atr_vec(bars: pd.DataFrame, length: int) -> np.ndarray:
    """ATR de Wilder (RMA recursivo desde la primera vela)."""
    tr = _true_range(
        bars["high"].to_numpy(float), bars["low"].to_numpy(float),
        bars["close"].to_numpy(float),
    )
    return pd.Series(tr).ewm(alpha=1.0 / length, adjust=False).mean().to_numpy()


def adx_vec(bars: pd.DataFrame, dilen: int = 14, adxlen: int = 14) -> pd.DataFrame:
    """
    ADX/DI: primer TR = high-low; primer +DM/-DM = 0; RMA desde la primera
    vela; DI = 100*DM_RMA/TR_RMA; DX = |+DI - -DI| / (+DI + -DI);
    ADX = RMA(100*DX) con seed 100*DX_0. Visible desde dilen + adxlen - 2.
    """
    if dilen <= 0 or adxlen <= 0:
        raise ValueError("ADX lengths must be > 0")

    n = len(bars)
    out = pd.DataFrame(
        {"adx": np.nan, "plus_di": np.nan, "minus_di": np.nan}, index=bars.index
    )
    if n == 0:
        return out

    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)

    up = np.r_[0.0, high[1:] - high[:-1]]
    down = np.r_[0.0, low[:-1] - low[1:]]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(high, low, close)

    a_di = 1.0 / dilen
    tr_rma = pd.Series(tr).ewm(alpha=a_di, adjust=False).mean().to_numpy()
    p_rma = pd.Series(plus_dm).ewm(alpha=a_di, adjust=False).mean().to_numpy()
    m_rma = pd.Series(minus_dm).ewm(alpha=a_di, adjust=False).mean().to_numpy()

    safe_tr = np.where(tr_rma != 0, tr_rma, 1.0)
    plus_di = np.where(tr_rma != 0, 100.0 * p_rma / safe_tr, 0.0)
    minus_di = np.where(tr_rma != 0, 100.0 * m_rma / safe_tr, 0.0)

    summ = plus_di + minus_di
    dx = np.abs(plus_di - minus_di) / np.where(summ != 0, summ, 1.0)
    adx = pd.Series(100.0 * dx).ewm(alpha=1.0 / adxlen, adjust=False).mean().to_numpy()

    visible_from = dilen + adxlen - 2
    if visible_from < n:
        out.iloc[visible_from:, 0] = adx[visible_from:]
        out.iloc[visible_from:, 1] = plus_di[visible_from:]
        out.iloc[visible_from:, 2] = minus_di[visible_from:]
    return out


# ---------------------------------------------------------------------------
# Precálculo: features por vela 1m y señales
# ---------------------------------------------------------------------------

def _map_tf(
    data: pd.DataFrame,
    rule: str,
    cfg: Config,
    decision_ns: np.ndarray,
    adx: bool = False,
    atr: bool = False,
) -> dict:
    """Calcula HTF y proyecta exclusivamente velas ya cerradas.

    `decision_ns` representa el instante en que la vela 1m queda cerrada.
    La regla es deliberadamente conservadora: una HTF solo entra en una
    decisión si su timestamp de cierre es <= al instante de decisión.
    """
    bars = resample_ohlcv(data, rule)
    ends = _ns(bars.index + pd.Timedelta(rule))
    pos = np.searchsorted(ends, decision_ns, side="right") - 1
    ok = pos >= 0
    j = np.where(ok, pos, 0)

    ef = ema_closed(bars["close"], cfg.ema_fast).to_numpy()
    es = ema_closed(bars["close"], cfg.ema_slow).to_numpy()
    out = {
        "ok": ok,
        "close": bars["close"].to_numpy(float)[j],
        "ef": ef[j],
        "es": es[j],
    }
    if adx:
        a = adx_vec(bars, cfg.adx_dilen, cfg.adx_len)
        out["adx"] = a["adx"].to_numpy()[j]
        out["adx_prev"] = a["adx"].shift(1).to_numpy()[j]
        out["plus_di"] = a["plus_di"].to_numpy()[j]
        out["minus_di"] = a["minus_di"].to_numpy()[j]
    if atr:
        out["atr"] = atr_vec(bars, cfg.atr_len)[j]
    return out


def build_features(data: pd.DataFrame, cfg: Config) -> dict:
    """Indicadores alineados al instante real de decisión de cada vela 1m.

    Por defecto el dataset usa timestamps de apertura, por lo que se suma un
    minuto para obtener el cierre. Si el proveedor entrega timestamps de cierre,
    `timestamp_is_close=True` evita el desplazamiento y, por tanto, el lookahead.
    """
    n = len(data)
    decision_ns = _ns(data.index) if cfg.timestamp_is_close else _ns(data.index + pd.Timedelta(minutes=1))

    feats = {
        "decision_ns": decision_ns,
        "1h": _map_tf(data, "1h", cfg, decision_ns, adx=True),
        "30m": _map_tf(data, "30min", cfg, decision_ns),
        "15m": _map_tf(data, "15min", cfg, decision_ns, atr=True),
        "5m": _map_tf(data, "5min", cfg, decision_ns),
    }
    c1 = data["close"]
    feats["1m"] = {
        "ef": ema_closed(c1, cfg.ema_fast).to_numpy(),
        "es": ema_closed(c1, cfg.ema_slow).to_numpy(),
        "close": c1.to_numpy(float),
        "high": data["high"].to_numpy(float),
        "low": data["low"].to_numpy(float),
        "prev_high": np.r_[np.nan, data["high"].to_numpy(float)[:-1]],
        "prev_low": np.r_[np.nan, data["low"].to_numpy(float)[:-1]],
        "atr": atr_vec(data, cfg.atr_len),
    }
    feats["n"] = n
    return feats


def build_signals(f: dict, cfg: Config) -> dict:
    """Condiciones de Strategy1 + filtros improved como arrays booleanos."""
    n = f["n"]
    h1, m30, m15, m5, m1 = f["1h"], f["30m"], f["15m"], f["5m"], f["1m"]

    with np.errstate(invalid="ignore"):
        bull1h = (h1["ef"] > h1["es"]) & (h1["adx"] >= cfg.adx_threshold) \
            & (h1["plus_di"] > h1["minus_di"])
        bear1h = (h1["ef"] < h1["es"]) & (h1["adx"] >= cfg.adx_threshold) \
            & (h1["minus_di"] > h1["plus_di"])

        bull30 = m30["ef"] > m30["es"]
        bear30 = m30["ef"] < m30["es"]

        # Pullback: solo estructura 15m + cierre contra la EMA rápida.
        long_pb = (m15["ef"] > m15["es"]) & (m15["close"] <= m15["ef"])
        short_pb = (m15["ef"] < m15["es"]) & (m15["close"] >= m15["ef"])

        bull5 = m5["ef"] > m5["es"]
        bear5 = m5["ef"] < m5["es"]

        bull1m = (m1["ef"] > m1["es"]) & (m1["close"] > m1["ef"]) \
            & (m1["close"] > m1["prev_high"])
        bear1m = (m1["ef"] < m1["es"]) & (m1["close"] < m1["ef"]) \
            & (m1["close"] < m1["prev_low"])

        entry_long = bull1h & bull30 & long_pb & bull5 & bull1m
        entry_short = bear1h & bear30 & short_pb & bear5 & bear1m

        # ---- filtros improved (solo afectan a entradas) ----
        mask = np.ones(n, dtype=bool)
        if cfg.adx_max > 0:
            mask &= h1["adx"] <= cfg.adx_max
        if cfg.adx_rising:
            mask &= h1["adx"] > h1["adx_prev"]
        if cfg.min_ema_gap_1h_pct > 0:
            mask &= (np.abs(h1["ef"] - h1["es"]) / h1["es"]) >= cfg.min_ema_gap_1h_pct
        if cfg.min_trigger_range_atr > 0:
            mask &= (m1["high"] - m1["low"]) >= cfg.min_trigger_range_atr * m1["atr"]
        blocked = cfg.blocked_hours()
        if blocked:
            hours = (f["decision_ns"] // 3_600_000_000_000) % 24
            mask &= ~np.isin(hours, blocked)

    # Una señal solo es válida si todos los timeframes tienen estado completo.
    valid = (
        h1["ok"] & m30["ok"] & m15["ok"] & m5["ok"]
        & np.isfinite(h1["adx"]) & np.isfinite(h1["plus_di"]) & np.isfinite(h1["minus_di"])
        & (np.arange(n) >= 1)
    )

    return {
        "valid": valid,
        "bull1h": bull1h, "bear1h": bear1h,
        "bull30": bull30, "bear30": bear30,
        "entry_long": entry_long & mask,
        "entry_short": entry_short & mask,
        "atr15": m15["atr"],
    }


# ---------------------------------------------------------------------------
# Abstracciones reutilizables de estrategia
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Construye features MTF. Reutilizable por cualquier bot basado en OHLCV."""

    def __init__(self, config: Config):
        self.cfg = config

    def build(self, data: pd.DataFrame) -> dict:
        return build_features(data, self.cfg)


class SignalEngine:
    """Contrato mínimo para una estrategia.

    Un bot posterior solo necesita implementar ``build(features)`` y devolver
    el mismo contrato de señales que consume el motor de ejecución, o adaptar
    el motor mediante una implementación propia.
    """

    def build(self, features: dict) -> dict:
        raise NotImplementedError


class Strategy1SignalEngine(SignalEngine):
    def __init__(self, config: Config):
        self.cfg = config

    def build(self, features: dict) -> dict:
        return build_signals(features, self.cfg)


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class Strategy1Backtester:
    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        verbose: bool = False,
        feature_builder: Optional[FeatureBuilder] = None,
        signal_engine: Optional[SignalEngine] = None,
    ):
        self.data = data
        self.cfg = config
        self.verbose = verbose

        self.feature_builder = feature_builder or FeatureBuilder(config)
        self.signal_engine = signal_engine or Strategy1SignalEngine(config)
        feats = self.feature_builder.build(data)
        self.sig = self.signal_engine.build(feats)

        self.o = data["open"].to_numpy(float)
        self.h = data["high"].to_numpy(float)
        self.l = data["low"].to_numpy(float)
        self.c = data["close"].to_numpy(float)
        self.times = data.index

        self.cash = config.initial_capital
        self.position: Optional[Position] = None
        self.pending_signal: Optional[dict] = None

        self.trades: list[Trade] = []
        self.trade_counter = 0
        self.position_counter = 0

        self.eq_time: list = []
        self.eq_cash: list = []
        self.eq_equity: list = []
        self.eq_side: list = []
        self.eq_qty: list = []

    # -------------------------- precios y costos --------------------------

    def _execution_price(self, raw_price: float, side: int) -> float:
        bps = self.cfg.slippage_bps / 10_000.0
        return raw_price * (1.0 + bps) if side == 1 else raw_price * (1.0 - bps)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.cfg.commission_bps / 10_000.0

    def _stop_pct_for(self, i: int, ref_price: float) -> float:
        """Stop inicial efectivo: k*ATR(15m) acotado (o el fijo de respaldo)."""
        cfg = self.cfg
        if cfg.atr_stop_mult > 0:
            atr = self.sig["atr15"][i]
            if np.isfinite(atr) and ref_price > 0:
                pct = cfg.atr_stop_mult * atr / ref_price
                return min(max(pct, cfg.atr_stop_min_pct), cfg.atr_stop_max_pct)
        return cfg.stop_loss_pct

    # -------------------------- ejecución --------------------------

    def _open(self, side: int, price: float, time: pd.Timestamp, stop_pct: float):
        if self.position is not None:
            return
        cfg = self.cfg
        exec_price = self._execution_price(price, side)

        if stop_pct <= 0 or self.cash <= 0:
            return
        # Tamaño por riesgo, con tope de apalancamiento.
        qty = self.cash * cfg.risk_pct / (exec_price * stop_pct)
        qty = min(qty, self.cash * cfg.max_leverage / exec_price)
        if qty <= 0:
            return

        fee = self._fee(exec_price * qty)
        self.cash -= fee
        self.position_counter += 1
        self.position = Position(
            position_id=self.position_counter,
            side=side,
            quantity=qty,
            initial_quantity=qty,
            entry_price=exec_price,
            entry_time=time,
            entry_fee=fee,
            stop_pct=stop_pct,
            risk_amount=qty * exec_price * stop_pct,
        )

    def _close_fraction(
        self,
        fraction: float,
        price: float,
        time: pd.Timestamp,
        reason: str,
        move_stop_to_entry: bool = False,
    ):
        pos = self.position
        if pos is None:
            return

        fraction = min(max(float(fraction), 0.0), 1.0)
        qty = pos.quantity * fraction
        if qty <= 0:
            return

        side = pos.side
        exec_price = self._execution_price(price, -side)
        gross = (exec_price - pos.entry_price) * qty * side
        exit_fee = self._fee(exec_price * qty)
        entry_fee_alloc = pos.entry_fee * (qty / pos.quantity)
        total_fees = entry_fee_alloc + exit_fee
        pnl_net = gross - total_fees

        self.cash += gross - exit_fee

        self.trade_counter += 1
        notional_entry = pos.entry_price * qty
        return_pct = pnl_net / notional_entry * 100.0 if notional_entry != 0 else 0.0

        self.trades.append(Trade(
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
        ))

        remaining = pos.quantity - qty
        if remaining <= max(1e-12, pos.initial_quantity * 1e-12):
            self.position = None
        else:
            pos.quantity = remaining
            pos.entry_fee -= entry_fee_alloc
            if move_stop_to_entry:
                pos.stop_at_entry = True
                if self.cfg.trail_atr_mult > 0:
                    pos.trailing = True

    def _check_stop_loss(self, i: int) -> Optional[str]:
        """
        Stop intrabar contra el OHLC de la vela 1m. Niveles:
          inicial   : entrada -/+ stop_pct
          breakeven : entrada (tras scale-out)
          trailing  : el más ajustado entre breakeven y el trailing
        Gap: si la vela abre más allá del nivel, el fill es el open.
        Devuelve la etiqueta del cierre, o None si no se tocó.
        """
        pos = self.position
        if pos is None:
            return None

        at_entry = pos.stop_at_entry
        pct = pos.stop_pct
        if not at_entry and pct <= 0:
            return None

        bo, bh, bl = self.o[i], self.h[i], self.l[i]
        e = pos.entry_price
        side = pos.side
        trail_binding = False

        if side == 1:
            stop = e if at_entry else e * (1.0 - pct)
            if pos.trail_level is not None and pos.trail_level > stop:
                stop = pos.trail_level
                trail_binding = True
            if bl > stop:
                return None
            fill = bo if bo <= stop else stop
        else:
            stop = e if at_entry else e * (1.0 + pct)
            if pos.trail_level is not None and pos.trail_level < stop:
                stop = pos.trail_level
                trail_binding = True
            if bh < stop:
                return None
            fill = bo if bo >= stop else stop

        if trail_binding:
            reason = "TRAIL_STOP"
        elif at_entry:
            reason = "BREAKEVEN"
        else:
            reason = f"STOP_LOSS_{pct * 100:.2f}%"

        qty_before = pos.quantity
        self._close_fraction(1.0, fill, self.times[i], reason)
        if self.verbose:
            print(
                f"[STOP] {self.times[i]} | {'LONG' if side == 1 else 'SHORT'} | "
                f"entrada={e:.8f} | stop={stop:.8f} | fill={fill:.8f} | "
                f"qty={qty_before} | {reason}",
                flush=True,
            )
        return reason

    def _update_trailing(self, i: int):
        """Actualiza el trailing con la vela ya cerrada; rige desde la siguiente."""
        pos = self.position
        mult = self.cfg.trail_atr_mult
        atr = self.sig["atr15"][i]
        if pos is None or not pos.trailing or mult <= 0 or not np.isfinite(atr):
            return
        if pos.side == 1:
            pos.extreme = self.h[i] if pos.extreme is None else max(pos.extreme, self.h[i])
            cand = pos.extreme - mult * atr
            pos.trail_level = cand if pos.trail_level is None else max(pos.trail_level, cand)
        else:
            pos.extreme = self.l[i] if pos.extreme is None else min(pos.extreme, self.l[i])
            cand = pos.extreme + mult * atr
            pos.trail_level = cand if pos.trail_level is None else min(pos.trail_level, cand)

    def _execute_pending(self, i: int):
        sig = self.pending_signal
        if sig is None:
            return
        self.pending_signal = None
        bar_open, ts = self.o[i], self.times[i]

        if sig["action"] == "BUY":
            self._open(1, bar_open, ts, sig["stop_pct"])
        elif sig["action"] == "SELL":
            self._open(-1, bar_open, ts, sig["stop_pct"])
        elif sig["action"] == "EXIT":
            self._close_fraction(
                float(sig["fraction"]), bar_open, ts, sig["reason"],
                move_stop_to_entry=bool(sig.get("move_stop_to_entry", False)),
            )

    # -------------------------- equity --------------------------

    def _mark(self, time: pd.Timestamp, close: float):
        """Marca equity como valor de liquidación neto, no solo MTM bruto.

        Esto evita que la curva de equity sobreestime ligeramente el capital
        cuando comisión/slippage están activados. El coste hipotético de salida
        se incluye solo en la marca; no se descuenta del cash hasta cerrar.
        """
        pos = self.position
        equity = self.cash
        if pos is not None:
            liquidation_price = self._execution_price(close, -pos.side)
            gross = (liquidation_price - pos.entry_price) * pos.quantity * pos.side
            exit_fee = self._fee(liquidation_price * pos.quantity)
            equity += gross - exit_fee
        self.eq_time.append(time)
        self.eq_cash.append(self.cash)
        self.eq_equity.append(equity)
        self.eq_side.append("FLAT" if pos is None else ("LONG" if pos.side == 1 else "SHORT"))
        self.eq_qty.append(0.0 if pos is None else pos.quantity)

    # -------------------------- bucle principal --------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg, sig = self.cfg, self.sig
        n = len(self.data)
        valid, bull1h, bear1h = sig["valid"], sig["bull1h"], sig["bear1h"]
        bull30, bear30 = sig["bull30"], sig["bear30"]
        entry_long, entry_short = sig["entry_long"], sig["entry_short"]

        started = time.perf_counter()
        last_log = -10
        cooldown_until: Optional[pd.Timestamp] = None
        one_min = pd.Timedelta(minutes=1)
        print(f"[BACKTEST] {n:,} velas 1m", flush=True)

        for i in range(n):
            # 1) Ejecutar la señal generada al cierre anterior (open de esta vela).
            if self.pending_signal is not None:
                self._execute_pending(i)

            # 2) Stop intrabar: prioridad sobre cualquier señal nueva.
            stop_reason = self._check_stop_loss(i)
            stop_hit = stop_reason is not None
            if stop_hit and stop_reason.startswith("STOP_LOSS"):
                cooldown_until = self.times[i] + pd.Timedelta(minutes=cfg.cooldown_min)


            # 3) Trailing con la vela ya cerrada (rige desde la siguiente).
            if self.position is not None and self.position.trailing:
                self._update_trailing(i)

            # 4) Señal al cierre de esta vela, a ejecutar en la próxima apertura.
            new_sig = None
            if not stop_hit and valid[i]:
                pos = self.position
                if pos is not None:
                    side = pos.side
                    if (side == 1 and bear1h[i]) or (side == -1 and bull1h[i]):
                        new_sig = {"action": "EXIT", "fraction": 1.0, "reason": "1H_REVERSAL"}
                    elif pos.quantity >= pos.initial_quantity - 1e-12 and (
                        (side == 1 and bear30[i]) or (side == -1 and bull30[i])
                    ):
                        new_sig = {
                            "action": "EXIT",
                            "fraction": cfg.scale_out_fraction,
                            "reason": "30M_SCALE_OUT",
                            "move_stop_to_entry": True,
                        }
                elif cooldown_until is None or self.times[i] >= cooldown_until:
                    if entry_long[i]:
                        new_sig = {"action": "BUY", "reason": "MTF_LONG",
                                   "stop_pct": self._stop_pct_for(i, self.c[i])}
                    elif entry_short[i]:
                        new_sig = {"action": "SELL", "reason": "MTF_SHORT",
                                   "stop_pct": self._stop_pct_for(i, self.c[i])}
            if new_sig is not None:
                self.pending_signal = new_sig

            self._mark(self.times[i] + one_min, self.c[i])

            pct = int((i + 1) / n * 100)
            if pct >= last_log + 10 or i == n - 1:
                elapsed = time.perf_counter() - started
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    f"[BACKTEST] {pct:3d}% | {rate:,.0f} velas/s | "
                    f"{elapsed:.1f}s | tramos {len(self.trades)}",
                    flush=True,
                )
                last_log = pct

        if self.position is not None and cfg.force_close_at_end:
            last_ts = self.times[-1] + one_min
            self._close_fraction(1.0, self.c[-1], last_ts, "END_OF_DATA")
            self._mark(last_ts, self.c[-1])

        trades = (
            pd.DataFrame([asdict(t) for t in self.trades])
            if self.trades else empty_trades()
        )
        equity = pd.DataFrame({
            "timestamp": self.eq_time,
            "cash": self.eq_cash,
            "equity": self.eq_equity,
            "position_side": self.eq_side,
            "position_quantity": self.eq_qty,
        })
        print(
            f"[BACKTEST] Finalizado en {time.perf_counter() - started:.2f}s | "
            f"tramos={len(trades):,}",
            flush=True,
        )
        return trades, equity


# ---------------------------------------------------------------------------
# Métricas (por posición)
# ---------------------------------------------------------------------------

def positions_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los tramos (scale-out + resto) en una fila por posición."""
    if trades.empty:
        return pd.DataFrame(columns=[
            "position_id", "side", "entry_time", "exit_time", "pnl_net",
            "fees", "risk", "legs", "reason", "r_multiple",
        ])
    pos = trades.groupby("position_id", sort=True).agg(
        side=("side", "first"),
        entry_time=("entry_time", "first"),
        exit_time=("exit_time", "last"),
        pnl_net=("pnl_net", "sum"),
        fees=("fees", "sum"),
        risk=("position_risk", "first"),
        legs=("trade_id", "count"),
        reason=("reason", "last"),
    ).reset_index()
    pos["r_multiple"] = np.where(pos["risk"] > 0, pos["pnl_net"] / pos["risk"], np.nan)
    return pos


def position_stats(pos: pd.DataFrame) -> dict:
    if pos.empty:
        return {
            "positions": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
            "profit_factor": None, "net_pnl": 0.0, "avg_pnl": 0.0,
            "expectancy_r": None, "max_consecutive_losses": 0,
        }
    pnl = pos["pnl_net"].astype(float)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())

    streak = best_streak = 0
    for v in pnl.to_numpy():
        streak = streak + 1 if v < 0 else 0
        best_streak = max(best_streak, streak)

    r = pos["r_multiple"].dropna()
    return {
        "positions": int(len(pnl)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "net_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "expectancy_r": float(r.mean()) if len(r) else None,
        "max_consecutive_losses": int(best_streak),
    }


def calculate_metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict:
    pos = positions_from_trades(trades)
    stats = position_stats(pos)

    if equity.empty:
        final_equity = initial_capital
        max_dd = sharpe = sortino = 0.0
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
        min_vs_capital_pct = float((eq / initial_capital - 1.0).min() * 100.0) if initial_capital else 0.0

        trough_pos = int(dd.idxmin())
        dd_trough_equity = float(eq.loc[trough_pos])
        dd_peak_equity = float(peak.loc[trough_pos])
        dd_peak_vs_capital_pct = float((dd_peak_equity / initial_capital - 1.0) * 100.0) if initial_capital else 0.0
        dd_trough_vs_capital_pct = float((dd_trough_equity / initial_capital - 1.0) * 100.0) if initial_capital else 0.0

        daily = equity.set_index("timestamp")["equity"].resample("1D").last().dropna()
        rets = daily.pct_change().dropna()
        sharpe = sortino = 0.0
        if len(rets) > 2 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(365))
            downside = rets[rets < 0]
            if len(downside) > 1 and downside.std() > 0:
                sortino = float(rets.mean() / downside.std() * math.sqrt(365))

    net = final_equity - initial_capital
    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "net_pnl": net,
        "return_pct": net / initial_capital * 100.0 if initial_capital else 0.0,
        "max_drawdown_pct": max_dd,
        "max_dd_peak_equity": dd_peak_equity,
        "max_dd_trough_equity": dd_trough_equity,
        "max_dd_peak_vs_capital_pct": dd_peak_vs_capital_pct,
        "max_dd_trough_vs_capital_pct": dd_trough_vs_capital_pct,
        "min_equity": min_equity,
        "min_equity_vs_capital_pct": min_vs_capital_pct,
        "return_over_maxdd": (
            (net / initial_capital * 100.0) / abs(max_dd)
            if initial_capital and max_dd < 0 else None
        ),
        "sharpe_daily_365": sharpe,
        "sortino_daily_365": sortino,
        "legs": int(len(trades)),
        **stats,
    }


def equity_curve_from_trades(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Curva de equity reconstruida desde los trades (un punto por cierre)."""
    columns = ["timestamp", "trade_id", "pnl_net", "cum_pnl", "equity"]
    if trades.empty or "exit_time" not in trades.columns:
        return pd.DataFrame(columns=columns)

    df = trades.sort_values(["exit_time", "trade_id"], kind="stable").copy()
    df["cum_pnl"] = df["pnl_net"].astype(float).cumsum()
    df["equity"] = initial_capital + df["cum_pnl"]
    out = df[["trade_id", "exit_time", "pnl_net", "cum_pnl", "equity"]].rename(
        columns={"exit_time": "timestamp"}
    )
    return out.reset_index(drop=True)[columns]


def plot_equity(curve: pd.DataFrame, output_path: str | Path, initial_capital: float,
                title: str = "Equity curve", dpi: int = 150) -> Optional[Path]:
    """Guarda un gráfico de línea de la curva de equity. Devuelve None si no hay datos."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib ausente o sin backend
        print(f"[PLOT] matplotlib no disponible ({exc}); se omite el gráfico.", flush=True)
        return None

    if curve.empty:
        print("[PLOT] No hay trades: no se genera la curva de equity.", flush=True)
        return None

    out = Path(output_path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_datetime(curve["timestamp"], utc=True)
    equity = curve["equity"].to_numpy(float)
    pnl = curve["pnl_net"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, equity, color="#1f77b4", linewidth=1.4, label="Equity")
    ax.scatter(x[pnl >= 0], equity[pnl >= 0], s=14, color="#2ca02c",
               alpha=0.7, zorder=3, label="Cerrar ganadora")
    ax.scatter(x[pnl < 0], equity[pnl < 0], s=14, color="#d62728",
               alpha=0.7, zorder=3, label="Cerrar perdedora")
    ax.axhline(initial_capital, color="grey", linestyle="--", linewidth=0.9,
               label=f"Capital inicial ({initial_capital:,.2f})")

    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1.0) * 100.0
    dd_text = f" | max DD {dd.min():.2f}%" if len(dd) else ""

    if initial_capital:
        below_pct = (equity / initial_capital - 1.0) * 100.0
        min_below_pct = float(np.min(below_pct))
    else:
        min_below_pct = 0.0
    min_below_equity = initial_capital * (1.0 + min_below_pct / 100.0)
    ax.axhline(min_below_equity, color="#ff7f0e", linestyle=":", linewidth=1.1,
               label=f"Min equity bajo capital: {min_below_pct:.2f}% ({min_below_equity:,.2f})")

    if len(dd):
        trough_i = int(np.argmin(dd))
        trough_equity = float(equity[trough_i])
        peak_equity = float(peak[trough_i])
        peak_pct = (peak_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
        ax.axhline(trough_equity, color="#9467bd", linestyle="-.", linewidth=1.1,
                   label=f"Valle max DD: {dd.min():.2f}% desde pico {peak_equity:,.2f} "
                         f"[{peak_pct:+.2f}%] -> {trough_equity:,.2f}")
        ax.plot(x[trough_i], trough_equity, marker="v", color="#9467bd", markersize=7,
                zorder=4, linestyle="none")

    ax.set_title(f"{title} | {len(curve):,} cierres{dd_text} | min bajo capital {min_below_pct:.2f}%")
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[PLOT] Curva de equity guardada en: {out}", flush=True)
    return out


def _fmt(v, spec=".3f"):
    return "N/A" if v is None else format(v, spec)


def print_report(m: dict, cfg: Config, data: pd.DataFrame):
    print("\n" + "=" * 72)
    print("STRATEGY 1 — IMPROVED")
    print("=" * 72)
    print(f"Periodo         : {data.index[0]} -> {data.index[-1]}")
    print(f"Capital         : {m['initial_capital']:,.2f} -> {m['final_equity']:,.2f}")
    print(f"PnL neto        : {m['net_pnl']:,.2f}  ({m['return_pct']:.2f}%)")
    print(f"Max drawdown    : {m['max_drawdown_pct']:.2f}%  "
          f"(pico {m['max_dd_peak_equity']:,.2f} [{m['max_dd_peak_vs_capital_pct']:+.2f}%] -> "
          f"valle {m['max_dd_trough_equity']:,.2f} [{m['max_dd_trough_vs_capital_pct']:+.2f}%])")
    print(f"Equity mínimo   : {m['min_equity']:,.2f}  ({m['min_equity_vs_capital_pct']:.2f}% vs capital inicial)")
    print(f"Retorno / MaxDD : {_fmt(m['return_over_maxdd'], '.2f')}")
    print(f"Sharpe / Sortino: {m['sharpe_daily_365']:.2f} / {m['sortino_daily_365']:.2f}")
    print(f"Posiciones      : {m['positions']:,} (tramos: {m['legs']:,})")
    print(f"Win rate        : {m['win_rate_pct']:.2f}%  "
          f"({m['wins']} ganadoras / {m['losses']} perdedoras)")
    print(f"Profit factor   : {_fmt(m['profit_factor'])}")
    print(f"Expectancy (R)  : {_fmt(m['expectancy_r'])}")
    print(f"Avg PnL/posición: {m['avg_pnl']:,.4f}")
    print(f"Racha pérdidas  : {m['max_consecutive_losses']}")
    print(f"Costos          : {cfg.commission_bps} bps comisión | {cfg.slippage_bps} bps slippage")
    print(f"Sizing / stop   : riesgo {cfg.risk_pct:.2%} (lev máx {cfg.max_leverage}x) / "
          f"{cfg.atr_stop_mult}xATR15m [{cfg.atr_stop_min_pct:.2%}-{cfg.atr_stop_max_pct:.2%}]")
    print(f"Trailing        : {cfg.trail_atr_mult}xATR15m" if cfg.trail_atr_mult > 0
          else "Trailing        : desactivado")
    print(f"Filtros         : cooldown={cfg.cooldown_min}m adx_max={cfg.adx_max} "
          f"adx_rising={cfg.adx_rising} gap1h={cfg.min_ema_gap_1h_pct} "
          f"rango1m={cfg.min_trigger_range_atr}xATR horas_bloq={cfg.block_hours_utc or '-'}")
    print("=" * 72)


def oos_split(trades: pd.DataFrame, oos_start: str) -> Optional[dict]:
    """Métricas de posiciones antes y desde la fecha de corte de entrada."""
    pos = positions_from_trades(trades)
    if pos.empty:
        return None
    cut = pd.Timestamp(oos_start, tz="UTC")
    return {
        "cut": str(cut),
        "before_cut": position_stats(pos[pos["entry_time"] < cut]),
        "from_cut": position_stats(pos[pos["entry_time"] >= cut]),
    }


def print_oos_split(trades: pd.DataFrame, oos_start: str):
    split = oos_split(trades, oos_start)
    if split is None:
        return
    print(f"\nSplit por fecha de entrada (corte {oos_start}):")
    print(f"  {'':<14}{'posiciones':>11}{'win%':>8}{'PF':>8}{'exp.R':>8}{'PnL neto':>14}")
    for key, name in (("before_cut", "antes"), ("from_cut", "desde el corte")):
        s = split[key]
        print(f"  {name:<14}{s['positions']:>11}{s['win_rate_pct']:>8.1f}"
              f"{_fmt(s['profit_factor'], '.2f'):>8}{_fmt(s['expectancy_r'], '.2f'):>8}"
              f"{s['net_pnl']:>14,.2f}")


# ---------------------------------------------------------------------------
# Reporte en disco: trades.csv + report.json
# ---------------------------------------------------------------------------

def side_stats(trades: pd.DataFrame) -> dict:
    """Métricas separadas por lado."""
    out = {}
    for side in ("LONG", "SHORT"):
        pos = positions_from_trades(trades[trades["side"] == side] if len(trades) else trades)
        out[side] = position_stats(pos)
    return out


def exit_reason_breakdown(trades: pd.DataFrame) -> dict:
    """Tramos y PnL por motivo de cierre."""
    if trades.empty:
        return {}
    grouped = trades.groupby("reason", sort=False).agg(
        legs=("trade_id", "count"), pnl_net=("pnl_net", "sum")
    )
    return {
        str(reason): {"legs": int(row["legs"]), "pnl_net": float(row["pnl_net"])}
        for reason, row in grouped.iterrows()
    }


def build_report(
    cfg: Config,
    data: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict,
    source: Optional[str] = None,
    oos_start: Optional[str] = None,
) -> dict:
    """Reporte completo y serializable de la corrida."""
    report = {
        "strategy": "strategy1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": source,
            "bars_1m": int(len(data)),
            "start": data.index[0].isoformat(),
            "end": data.index[-1].isoformat(),
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
    """Escribe trades.csv, report.json y el resto de artefactos de la corrida."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades.to_csv(out / "trades.csv", index=False)
    positions_from_trades(trades).to_csv(out / "positions.csv", index=False)
    curve.to_csv(out / "equity_curve.csv", index=False)
    if write_equity_csv:
        equity.to_csv(out / "equity.csv", index=False)

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(report["metrics"]), f, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(report["config"], f, indent=2)

    written = ["trades.csv", "report.json", "metrics.json", "config.json",
               "positions.csv", "equity_curve.csv"]
    if write_equity_csv:
        written.append("equity.csv")
    print(f"[OUT] {out.resolve()} -> {', '.join(written)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    d = Config()
    p = argparse.ArgumentParser(
        description="Backtester Strategy1 IMPROVED sobre OHLCV 1m Parquet."
    )
    p.add_argument("parquet", help="Ruta al dataset .parquet")
    p.add_argument("--capital", type=float, default=d.initial_capital)
    p.add_argument("--scale-out", type=float, default=d.scale_out_fraction)
    p.add_argument("--adx-threshold", type=float, default=d.adx_threshold)
    p.add_argument("--commission-bps", type=float, default=d.commission_bps)
    p.add_argument("--slippage-bps", type=float, default=d.slippage_bps)
    p.add_argument("--stop-loss-pct", type=float, default=d.stop_loss_pct,
                   help="Stop fijo de respaldo si no hay ATR (0.006 = 0.6%%).")
    p.add_argument("--no-force-close", action="store_true")
    p.add_argument("--oos-start", default=None,
                   help="Fecha (YYYY-MM-DD) para separar el reporte antes/después.")
    p.add_argument("--verbose", action="store_true", help="Imprime cada stop.")
    p.add_argument("--timestamp-is-close", action=argparse.BooleanOptionalAction, default=None,
                   help=f"Indica que el timestamp de 1m es cierre, no apertura (default: {d.timestamp_is_close}).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Directorio de trades.csv y report.json (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--no-output", action="store_true",
                   help="No escribe trades.csv ni report.json.")
    p.add_argument("--equity-csv", action="store_true",
                   help="Escribe además equity.csv (marca por vela 1m; archivo pesado).")
    p.add_argument("--plot", default=None, metavar="PATH",
                   help="Ruta del gráfico de equity. Por defecto equity_curve.png "
                        "en --output-dir o en el directorio actual.")
    p.add_argument("--no-plot", action="store_true", help="No genera el gráfico de equity.")
    p.add_argument("--plot-dpi", type=int, default=150)

    # Parámetros improved: default None = usa el valor de Config.
    for key in CLI_KEYS:
        flag = "--" + key.replace("_", "-")
        default_value = getattr(d, key)
        if isinstance(default_value, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None,
                           help=f"(default: {default_value})")
        else:
            p.add_argument(flag, type=type(default_value), default=None,
                           help=f"(default: {default_value!r})")
    return p.parse_args()


def make_config(args) -> Config:
    overrides = {k: getattr(args, k) for k in CLI_KEYS if getattr(args, k) is not None}
    return Config(
        initial_capital=args.capital,
        scale_out_fraction=args.scale_out,
        adx_threshold=args.adx_threshold,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss_pct,
        force_close_at_end=not args.no_force_close,
        timestamp_is_close=(Config().timestamp_is_close if args.timestamp_is_close is None else args.timestamp_is_close),
        **overrides,
    )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main():
    args = parse_args()
    cfg = make_config(args)

    data = load_parquet(args.parquet)
    if len(data) < 100:
        raise ValueError("El dataset tiene muy pocas velas 1m para este backtest.")

    engine = Strategy1Backtester(data, cfg, verbose=args.verbose)
    trades, equity = engine.run()
    metrics = calculate_metrics(trades, equity, cfg.initial_capital)

    print_report(metrics, cfg, data)
    if args.oos_start:
        print_oos_split(trades, args.oos_start)

    curve = equity_curve_from_trades(trades, cfg.initial_capital)

    if not args.no_output:
        report = build_report(
            cfg, data, trades, metrics,
            source=str(args.parquet), oos_start=args.oos_start,
        )
        write_outputs(
            args.output_dir, trades, equity, curve, report,
            write_equity_csv=args.equity_csv,
        )

    if not args.no_plot:
        plot_path = args.plot or str(Path(args.output_dir or ".") / "equity_curve.png")
        plot_equity(
            curve,
            plot_path,
            cfg.initial_capital,
            title=f"Strategy1 improved | {metrics['initial_capital']:,.2f} -> "
                  f"{metrics['final_equity']:,.2f}",
            dpi=args.plot_dpi,
        )


if __name__ == "__main__":
    main()