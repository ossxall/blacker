#!/usr/bin/env python3
"""
Backtesting dedicado para Strategy1
====================================

Dataset de entrada:
    OHLCV de 1 minuto en .parquet.

La estrategia replica:
    1H  -> tendencia: EMA20/EMA50 + ADX14 >= 25 + DI
    30m -> confirmación: EMA20 vs EMA50
    15m -> pullback: estructura + cierre contra EMA20
    5m  -> setup: EMA20 vs EMA50
    1m  -> trigger: estructura + cierre sobre/bajo EMA20 +
           ruptura del máximo/mínimo de la vela anterior

Salidas:
    Stop-loss -> cierre total al 0.6% desde la entrada.
    1H contra la posición -> cierre total.
    30m contra la posición -> scale-out del 50% por defecto.

Convención anti-lookahead:
    La señal se calcula usando exclusivamente velas cerradas.
    Una señal conocida al cierre de una vela se ejecuta en la apertura
    de la siguiente vela 1m.

STOP LOSS:
    El backtest aplica un stop-loss fijo del 0.6% desde el precio de
    entrada. Se comprueba intrabar sobre cada vela 1m. Si hay un gap
    que cruza el stop, la ejecución se realiza al open de la vela.

FIDELIDAD CON my_strategy.py:
    Este archivo es una reimplementación, no un import. Las diferencias
    respecto al original están acotadas y son todas intencionadas:

      - Las series se calculan una sola vez sobre todo el histórico en
        lugar de avanzar tick a tick. `ema_closed()` y `adx_exact()`
        reproducen bit a bit la recursión de EMA.py y ADX.py, incluido
        el seed sin SMA, el warm-up del ADX (`dilen + adxlen - 2`) y la
        lectura exclusiva del valor confirmado.
      - El bracket (stop / take profit) lo gestiona el motor en el
        engine; aquí solo se modela el stop-loss fijo.
      - `scale_out_fraction` fuera de (0, 1) lanza ValueError en lugar de
        caer silenciosamente a 0.5 como en my_strategy.py:93.

    Cualquier otra diferencia en las condiciones de entrada o salida es
    un bug, no una adaptación.

Dependencias:
    pip install pandas pyarrow numpy

Ejemplo:
    python backtest_strategy1.py data.parquet --capital 10000 \
        --quantity 1 --commission-bps 2 --slippage-bps 1 \
        --output-dir ./backtest_out
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import time
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class Config:
    tf_trend: str = "1h"
    tf_confirm: str = "30min"
    tf_pullback: str = "15min"
    tf_setup: str = "5min"
    tf_trigger: str = "1min"

    ema_fast: int = 20
    ema_slow: int = 50

    adx_dilen: int = 14
    adx_len: int = 14
    adx_threshold: float = 25.0

    quantity: float = 1.0

    # my_strategy.py:54 -> DEFAULT_PARAMS["scale_out_fraction"] = 0.5
    scale_out_fraction: float = 0.5

    initial_capital: float = 10_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    # Stop loss fijo desde el precio de entrada, 0.6%.
    stop_loss_pct: float = 0.006

    # Si True, una posición que queda abierta al final se cierra al último close.
    force_close_at_end: bool = True

    def __post_init__(self):
        # my_strategy.py:93 deja caer cualquier fraccion fuera de (0, 1) a
        # 0.5, porque >= 1 convertiria el cross de 30m en un cierre completo
        # -- que es lo que ya hace el reverso de 1H -- y <= 0 no cerraria
        # nada. Aqui se falla de forma explicita en lugar de sustituirla en
        # silencio, para que un valor mal escrito no pase inadvertido.
        if not (0.0 < self.scale_out_fraction < 1.0):
            raise ValueError(
                "scale_out_fraction debe estar estrictamente entre 0 y 1 "
                f"(recibido: {self.scale_out_fraction})."
            )

        if self.quantity <= 0.0:
            raise ValueError(
                f"quantity debe ser > 0 (recibido: {self.quantity})."
            )


@dataclass
class Position:
    side: int                 # +1 LONG, -1 SHORT
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_fee: float = 0.0


@dataclass
class Trade:
    trade_id: int
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


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y localiza OHLCV."""
    rename = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=rename)

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

    # Timestamp en UTC para evitar problemas de DST.
    #
    # Binance suele exportar `open_time` como epoch en MILISEGUNDOS
    # (por ejemplo, 1609459200000). Si se pasa directamente a
    # pd.to_datetime sin `unit`, pandas lo interpreta como nanosegundos y
    # produce fechas incorrectas. Detectamos automáticamente la unidad.
    raw_ts = out["timestamp"]
    if pd.api.types.is_numeric_dtype(raw_ts):
        numeric_ts = pd.to_numeric(raw_ts, errors="coerce")
        finite = numeric_ts.dropna()
        if finite.empty:
            ts = pd.to_datetime(numeric_ts, utc=True, errors="coerce")
        else:
            magnitude = float(finite.abs().median())
            if magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            elif magnitude >= 1e9:
                unit = "s"
            else:
                unit = None

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
    print(f"[LOAD] {len(df):,} filas cargadas en {time.perf_counter() - t0:.2f}s", flush=True)
    out = normalize_columns(df)
    print(
        f"[LOAD] Datos normalizados: {len(out):,} velas | "
        f"{out.index.min()} -> {out.index.max()}",
        flush=True,
    )
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Agrega 1m a OHLCV del timeframe solicitado."""
    out = df.resample(
        rule,
        label="left",
        closed="left",
        origin="start_day",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def bar_end_index(bars: pd.DataFrame, rule: str) -> pd.DatetimeIndex:
    """Momento en el que una vela pasa a estar confirmada."""
    delta = pd.Timedelta(rule)
    return bars.index + delta


# ---------------------------------------------------------------------------
# Indicadores: EMA
# ---------------------------------------------------------------------------

def ema_closed(values: pd.Series, period: int) -> pd.Series:
    """
    Replica la EMA entregada:

        primer valor = close
        siguiente = alpha*close + (1-alpha)*EMA_anterior

    No utiliza SMA como inicialización.
    """
    if period <= 0:
        raise ValueError("EMA period must be > 0")

    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan, dtype=float)

    alpha = 2.0 / (period + 1.0)
    prev = None

    for i, close in enumerate(x):
        if not np.isfinite(close):
            continue

        if prev is None:
            value = close
        else:
            value = alpha * close + (1.0 - alpha) * prev

        out[i] = value
        prev = value

    return pd.Series(out, index=values.index, name=f"EMA{period}")


# ---------------------------------------------------------------------------
# Indicadores: ADX
# ---------------------------------------------------------------------------

def _visible_from(dilen: int, adxlen: int) -> int:
    """
    Indice de la primera vela cuyo ADX es visible en el motor.

    ADX.py:228 solo publica valores cuando

        len(history) >= dilen + adxlen - 1

    y la asignacion de `_closed` ocurre dentro de esa misma puerta, de modo
    que durante el warm-up `_closed` permanece en None. Como `history`
    recibe una vela por rollover, la puerta se abre al procesar la vela
    (dilen + adxlen - 1), confirmando en ese momento la vela anterior.

    Primer valor confirmado visible = dilen + adxlen - 2.
    """
    return dilen + adxlen - 2


def adx_exact(
    bars: pd.DataFrame,
    dilen: int = 14,
    adxlen: int = 14,
    key_level: float = 23.0,
) -> pd.DataFrame:
    """
    Replica la cadena de cálculo del ADX suministrada.

    Particularidades conservadas:
      - Primer TR = high-low.
      - Primer +DM/-DM = 0.
      - RMA recursivo desde el primer bar; no se inicializa con SMA.
      - DI = 100 * DM_RMA / TR_RMA.
      - DX = abs(+DI - -DI) / (+DI + -DI).
      - ADX = 100*DX en el primer estado y luego RMA de DX.
      - El valor queda 'visible/usable' en la vela
            dilen + adxlen - 2
        porque el motor solo asigna `_closed` dentro de la puerta de
        warm-up. Ver `_visible_from()` para la derivacion.
    """
    if dilen <= 0 or adxlen <= 0:
        raise ValueError("ADX lengths must be > 0")

    n = len(bars)
    result = pd.DataFrame(index=bars.index)
    result["adx"] = np.nan
    result["plus_di"] = np.nan
    result["minus_di"] = np.nan
    result["adx_color"] = pd.Series(index=bars.index, dtype="object")
    result["is_reversal"] = pd.Series(False, index=bars.index, dtype="bool")
    result["reversal_level"] = np.nan

    if n == 0:
        return result

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    adx_arr = np.full(n, np.nan)
    plus_arr = np.full(n, np.nan)
    minus_arr = np.full(n, np.nan)
    reversal = np.zeros(n, dtype=bool)
    reversal_level = np.full(n, np.nan)
    color = np.empty(n, dtype=object)

    di_alpha = 1.0 / dilen
    adx_alpha = 1.0 / adxlen

    prev_high = None
    prev_low = None
    prev_close = None

    tr_rma = None
    plus_dm_rma = None
    minus_dm_rma = None
    prev_adx = None
    prev_prev_adx = None

    for i in range(n):
        h, l, c = high[i], low[i], close[i]

        if prev_close is None:
            tr = h - l
            plus_dm = 0.0
            minus_dm = 0.0

            tr_rma = tr
            plus_dm_rma = plus_dm
            minus_dm_rma = minus_dm
        else:
            up = h - prev_high
            down = prev_low - l

            plus_dm = up if (up > down and up > 0) else 0.0
            minus_dm = down if (down > up and down > 0) else 0.0

            tr = max(
                h - l,
                abs(h - prev_close),
                abs(l - prev_close),
            )

            tr_rma = tr * di_alpha + tr_rma * (1.0 - di_alpha)
            plus_dm_rma = (
                plus_dm * di_alpha
                + plus_dm_rma * (1.0 - di_alpha)
            )
            minus_dm_rma = (
                minus_dm * di_alpha
                + minus_dm_rma * (1.0 - di_alpha)
            )

        plus_di = (
            100.0 * plus_dm_rma / tr_rma
            if tr_rma != 0 else 0.0
        )
        minus_di = (
            100.0 * minus_dm_rma / tr_rma
            if tr_rma != 0 else 0.0
        )

        summ = plus_di + minus_di
        divisor = summ if summ != 0 else 1.0
        dx = abs(plus_di - minus_di) / divisor

        if prev_adx is None:
            adx_value = 100.0 * dx
        else:
            adx_value = (
                (100.0 * dx) * adx_alpha
                + prev_adx * (1.0 - adx_alpha)
            )

        if prev_adx is not None and adx_value > prev_adx:
            color[i] = "lime"
        else:
            color[i] = "red"

        if prev_adx is not None and prev_prev_adx is not None:
            rule1 = adx_value < prev_adx
            rule2 = prev_adx > prev_prev_adx
            rule3 = prev_adx > key_level
            reversal[i] = rule1 and rule2 and rule3
            if reversal[i]:
                reversal_level[i] = prev_adx

        adx_arr[i] = adx_value
        plus_arr[i] = plus_di
        minus_arr[i] = minus_di

        prev_prev_adx = prev_adx
        prev_adx = adx_value
        prev_high = h
        prev_low = l
        prev_close = c

    # ADX.py:228 gatea con `len(history) >= dilen + adxlen - 1`, pero la
    # asignacion de `_closed` esta DENTRO de esa puerta, asi que durante el
    # warm-up `_closed` sigue en None.
    #
    # Traza del motor (history crece una vela por rollover):
    #   update() de la vela k  ->  history.append(vela k-1)  ->  len = k
    #   primer k que cumple     ->  k = dilen + adxlen - 1
    #   `_closed` recibe        ->  la vela k-1
    #
    # El primer valor confirmado visible es, por tanto, la vela
    # (dilen + adxlen - 1) - 1 = dilen + adxlen - 2.
    visible_from = _visible_from(dilen, adxlen)

    if n:
        if visible_from < n:
            result.iloc[visible_from:, result.columns.get_loc("adx")] = adx_arr[visible_from:]
            result.iloc[visible_from:, result.columns.get_loc("plus_di")] = plus_arr[visible_from:]
            result.iloc[visible_from:, result.columns.get_loc("minus_di")] = minus_arr[visible_from:]
            result.iloc[visible_from:, result.columns.get_loc("adx_color")] = color[visible_from:]
            result.iloc[visible_from:, result.columns.get_loc("is_reversal")] = reversal[visible_from:]
            result.iloc[visible_from:, result.columns.get_loc("reversal_level")] = reversal_level[visible_from:]

    return result


# ---------------------------------------------------------------------------
# Timeframe calculado
# ---------------------------------------------------------------------------

@dataclass
class TFData:
    bars: pd.DataFrame
    ema_fast: pd.Series
    ema_slow: pd.Series
    adx: Optional[pd.DataFrame]
    ends: pd.DatetimeIndex


def build_tf_data(
    one_minute: pd.DataFrame,
    rule: str,
    ema_fast: int,
    ema_slow: int,
    adx_needed: bool = False,
    adx_dilen: int = 14,
    adx_len: int = 14,
) -> TFData:
    bars = resample_ohlcv(one_minute, rule)
    fast = ema_closed(bars["close"], ema_fast)
    slow = ema_closed(bars["close"], ema_slow)
    adx = (
        adx_exact(
            bars,
            dilen=adx_dilen,
            adxlen=adx_len,
            key_level=23.0,
        )
        if adx_needed
        else None
    )
    return TFData(
        bars=bars,
        ema_fast=fast,
        ema_slow=slow,
        adx=adx,
        ends=bar_end_index(bars, rule),
    )


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class Strategy1Backtester:
    def __init__(self, data: pd.DataFrame, config: Config):
        self.data = data
        self.cfg = config

        self.cash = config.initial_capital
        self.position: Optional[Position] = None

        self.trades: list[Trade] = []
        self.equity_rows: list[dict] = []

        self.pending_signal = None

        self.trade_counter = 0

        self.tfs: dict[str, TFData] = {}
        self._build_timeframes()

        self.last_price = float(self.data["close"].iloc[-1])

    def _build_timeframes(self):
        c = self.cfg
        self.tfs["1h"] = build_tf_data(
            self.data, "1h",
            c.ema_fast, c.ema_slow,
            adx_needed=True,
            adx_dilen=c.adx_dilen,
            adx_len=c.adx_len,
        )
        self.tfs["30min"] = build_tf_data(
            self.data, "30min",
            c.ema_fast, c.ema_slow,
        )
        self.tfs["15min"] = build_tf_data(
            self.data, "15min",
            c.ema_fast, c.ema_slow,
        )
        self.tfs["5min"] = build_tf_data(
            self.data, "5min",
            c.ema_fast, c.ema_slow,
        )
        # 1m ya está disponible directamente, pero calculamos EMA sobre él.
        self.tfs["1min"] = build_tf_data(
            self.data, "1min",
            c.ema_fast, c.ema_slow,
        )

    # -------------------------- datos confirmados --------------------------

    @staticmethod
    def _latest_index(tf: TFData, now: pd.Timestamp) -> Optional[int]:
        """
        Devuelve la última vela cuyo end <= now.
        """
        pos = tf.ends.searchsorted(now, side="right") - 1
        return int(pos) if pos >= 0 else None

    def _tf_state(self, name: str, now: pd.Timestamp) -> Optional[dict]:
        tf = self.tfs[name]
        i = self._latest_index(tf, now)

        if i is None:
            return None

        if (
            not np.isfinite(tf.ema_fast.iloc[i])
            or not np.isfinite(tf.ema_slow.iloc[i])
        ):
            return None

        state = {
            "bar_time": tf.bars.index[i],
            "bar_end": tf.ends[i],

            "open": float(tf.bars["open"].iloc[i]),
            "high": float(tf.bars["high"].iloc[i]),
            "low": float(tf.bars["low"].iloc[i]),
            "close": float(tf.bars["close"].iloc[i]),

            "ema_fast": float(tf.ema_fast.iloc[i]),
            "ema_slow": float(tf.ema_slow.iloc[i]),
        }

        if tf.adx is not None:
            adx = tf.adx.iloc[i]

            if (
                pd.isna(adx["adx"])
                or pd.isna(adx["plus_di"])
                or pd.isna(adx["minus_di"])
            ):
                return None

            state.update({
                "adx": float(adx["adx"]),
                "plus_di": float(adx["plus_di"]),
                "minus_di": float(adx["minus_di"]),
            })

        return state

    def _last_two_1m(self, now: pd.Timestamp):
        tf = self.tfs["1min"]
        i = self._latest_index(tf, now)
        if i is None or i < 1:
            return None, None

        current = {
            "time": tf.bars.index[i],
            "end": tf.ends[i],
            "open": float(tf.bars["open"].iloc[i]),
            "high": float(tf.bars["high"].iloc[i]),
            "low": float(tf.bars["low"].iloc[i]),
            "close": float(tf.bars["close"].iloc[i]),
            "ema_fast": float(tf.ema_fast.iloc[i]),
            "ema_slow": float(tf.ema_slow.iloc[i]),
        }
        previous = {
            "time": tf.bars.index[i - 1],
            "end": tf.ends[i - 1],
            "open": float(tf.bars["open"].iloc[i - 1]),
            "high": float(tf.bars["high"].iloc[i - 1]),
            "low": float(tf.bars["low"].iloc[i - 1]),
            "close": float(tf.bars["close"].iloc[i - 1]),
        }

        if not np.isfinite(current["ema_fast"]) or not np.isfinite(current["ema_slow"]):
            return None, None

        return current, previous

    # -------------------------- señales --------------------------

    def evaluate_signal(self, now: pd.Timestamp):
        """
        Replica el orden de Strategy1.evaluate():

        posición abierta:
            1H contra -> EXIT total
            30m contra -> EXIT parcial
            otherwise -> HOLD

        sin posición:
            1H neutral -> no trade
            después exige 30m + 15m + 5m + 1m.
        """
        one_h = self._tf_state("1h", now)
        thirty = self._tf_state("30min", now)
        fifteen = self._tf_state("15min", now)
        five = self._tf_state("5min", now)
        one = self._tf_state("1min", now)

        if any(x is None for x in [one_h, thirty, fifteen, five, one]):
            return None

        bullish_1h = (
            one_h["ema_fast"] > one_h["ema_slow"]
            and one_h["adx"] >= self.cfg.adx_threshold
            and one_h["plus_di"] > one_h["minus_di"]
        )
        bearish_1h = (
            one_h["ema_fast"] < one_h["ema_slow"]
            and one_h["adx"] >= self.cfg.adx_threshold
            and one_h["minus_di"] > one_h["plus_di"]
        )

        # Posición abierta: salidas primero.
        if self.position is not None:
            if self.position.side == 1 and bearish_1h:
                return {"action": "EXIT", "fraction": 1.0, "reason": "1H_REVERSAL"}

            if self.position.side == -1 and bullish_1h:
                return {"action": "EXIT", "fraction": 1.0, "reason": "1H_REVERSAL"}

            # Idempotencia: solo si la posición sigue completa.
            if self.position.quantity >= self.cfg.quantity - 1e-12:
                bullish_30 = thirty["ema_fast"] > thirty["ema_slow"]
                bearish_30 = thirty["ema_fast"] < thirty["ema_slow"]

                if self.position.side == 1 and bearish_30:
                    return {
                        "action": "EXIT",
                        "fraction": self.cfg.scale_out_fraction,
                        "reason": "30M_SCALE_OUT",
                    }

                if self.position.side == -1 and bullish_30:
                    return {
                        "action": "EXIT",
                        "fraction": self.cfg.scale_out_fraction,
                        "reason": "30M_SCALE_OUT",
                    }

            return None

        # Sin posición: 1H neutral = no trade.
        if not bullish_1h and not bearish_1h:
            return None

        bullish_30 = thirty["ema_fast"] > thirty["ema_slow"]
        bearish_30 = thirty["ema_fast"] < thirty["ema_slow"]

        bullish_15 = fifteen["ema_fast"] > fifteen["ema_slow"]
        bearish_15 = fifteen["ema_fast"] < fifteen["ema_slow"]

        # my_strategy.py:296-304 -- el pullback es SOLO estructura + cierre
        # contra la EMA rapida. No se exige nada respecto a la EMA lenta:
        # anadir `close > ema_slow` (long) / `close < ema_slow` (short)
        # endurecia la entrada y descartaba ~23% de los setups que la
        # estrategia original si aceptaba.
        long_pullback = (
            bullish_15
            and fifteen["close"] <= fifteen["ema_fast"]
        )

        short_pullback = (
            bearish_15
            and fifteen["close"] >= fifteen["ema_fast"]
        )

        bullish_5 = five["ema_fast"] > five["ema_slow"]
        bearish_5 = five["ema_fast"] < five["ema_slow"]

        current_1m, previous_1m = self._last_two_1m(now)
        if current_1m is None or previous_1m is None:
            return None

        bullish_1m = (
            current_1m["ema_fast"] > current_1m["ema_slow"]
            and current_1m["close"] > current_1m["ema_fast"]
            and current_1m["close"] > previous_1m["high"]
        )

        bearish_1m = (
            current_1m["ema_fast"] < current_1m["ema_slow"]
            and current_1m["close"] < current_1m["ema_fast"]
            and current_1m["close"] < previous_1m["low"]
        )

        if (
            bullish_1h
            and bullish_30
            and long_pullback
            and bullish_5
            and bullish_1m
        ):
            return {
                "action": "BUY",
                "quantity": self.cfg.quantity,
                "reason": "MTF_LONG",
            }

        if (
            bearish_1h
            and bearish_30
            and short_pullback
            and bearish_5
            and bearish_1m
        ):
            return {
                "action": "SELL",
                "quantity": self.cfg.quantity,
                "reason": "MTF_SHORT",
            }

        return None

    # -------------------------- ejecución --------------------------

    def _execution_price(self, raw_price: float, side: int) -> float:
        """
        Slippage:
            BUY  -> precio mayor
            SELL -> precio menor
        """
        bps = self.cfg.slippage_bps / 10_000.0
        return raw_price * (1.0 + bps) if side == 1 else raw_price * (1.0 - bps)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.cfg.commission_bps / 10_000.0

    def _open(self, side: int, qty: float, price: float, time: pd.Timestamp, reason: str):
        if qty <= 0 or self.position is not None:
            return

        exec_price = self._execution_price(price, side)
        fee = self._fee(exec_price * qty)

        self.cash -= fee
        self.position = Position(
            side=side,
            quantity=qty,
            entry_price=exec_price,
            entry_time=time,
            entry_fee=fee,
        )

    def _close_fraction(
        self,
        fraction: float,
        price: float,
        time: pd.Timestamp,
        reason: str,
    ):
        if self.position is None:
            return

        fraction = min(max(float(fraction), 0.0), 1.0)
        qty = self.position.quantity * fraction
        if qty <= 0:
            return

        side = self.position.side
        exec_price = self._execution_price(price, -side)

        gross = (
            (exec_price - self.position.entry_price)
            * qty
            * side
        )
        exit_fee = self._fee(exec_price * qty)

        # La comisión de entrada ya fue pagada al abrir la posición.
        # Se asigna proporcionalmente a cada tramo cerrado.
        entry_fee_alloc = (
            self.position.entry_fee
            * (qty / self.position.quantity)
        )
        total_fees = entry_fee_alloc + exit_fee
        pnl_net = gross - total_fees

        self.cash += gross - exit_fee

        self.trade_counter += 1
        notional_entry = self.position.entry_price * qty
        return_pct = (
            pnl_net / notional_entry * 100.0
            if notional_entry != 0 else 0.0
        )

        self.trades.append(
            Trade(
                trade_id=self.trade_counter,
                side="LONG" if side == 1 else "SHORT",
                entry_time=self.position.entry_time,
                exit_time=time,
                entry_price=self.position.entry_price,
                exit_price=exec_price,
                quantity=qty,
                pnl_gross=gross,
                fees=total_fees,
                pnl_net=pnl_net,
                return_pct=return_pct,
                reason=reason,
            )
        )

        remaining = self.position.quantity - qty

        if remaining <= max(1e-12, self.cfg.quantity * 1e-12):
            self.position = None
        else:
            # La comisión de entrada restante se conserva solo para la
            # cantidad todavía abierta.
            self.position.quantity = remaining
            self.position.entry_fee -= entry_fee_alloc

    def _check_stop_loss(self, bar_open: float, bar_high: float, bar_low: float, bar_time: pd.Timestamp) -> bool:
        """
        Comprueba el stop-loss fijo del 0.6% contra el OHLC de la vela 1m.

        LONG: stop = entrada * (1 - stop_loss_pct)
        SHORT: stop = entrada * (1 + stop_loss_pct)

        Si existe gap a través del stop, la ejecución se hace al open.
        Si el precio cruza el nivel durante la vela, se ejecuta al nivel
        del stop. Devuelve True si cerró la posición.
        """
        if self.position is None or self.cfg.stop_loss_pct <= 0:
            return False

        pct = self.cfg.stop_loss_pct
        side = self.position.side

        if side == 1:
            stop_price = self.position.entry_price * (1.0 - pct)
            if bar_low > stop_price:
                return False
            # Gap bajista: el fill realista es el open, no el nivel del stop.
            fill_price = bar_open if bar_open <= stop_price else stop_price
        else:
            stop_price = self.position.entry_price * (1.0 + pct)
            if bar_high < stop_price:
                return False
            # Gap alcista: el fill realista es el open, no el nivel del stop.
            fill_price = bar_open if bar_open >= stop_price else stop_price

        qty_before = self.position.quantity
        entry_price = self.position.entry_price
        # La etiqueta deriva del valor configurado para que no pueda
        # contradecir al stop realmente aplicado.
        self._close_fraction(
            fraction=1.0,
            price=fill_price,
            time=bar_time,
            reason=f"STOP_LOSS_{pct * 100:.1f}%",
        )
        print(
            f"[STOP] {bar_time} | {'LONG' if side == 1 else 'SHORT'} | "
            f"entrada={entry_price:.8f} | "
            f"stop={stop_price:.8f} | fill={fill_price:.8f} | qty={qty_before}",
            flush=True,
        )
        return True

    def _execute_pending(self, bar_open: float, bar_time: pd.Timestamp):
        if self.pending_signal is None:
            return

        sig = self.pending_signal
        self.pending_signal = None

        if sig["action"] == "BUY":
            self._open(
                side=1,
                qty=float(sig["quantity"]),
                price=bar_open,
                time=bar_time,
                reason=sig["reason"],
            )

        elif sig["action"] == "SELL":
            self._open(
                side=-1,
                qty=float(sig["quantity"]),
                price=bar_open,
                time=bar_time,
                reason=sig["reason"],
            )

        elif sig["action"] == "EXIT":
            self._close_fraction(
                fraction=float(sig["fraction"]),
                price=bar_open,
                time=bar_time,
                reason=sig["reason"],
            )

    # -------------------------- equity --------------------------

    def mark_to_market(self, time: pd.Timestamp, close: float):
        equity = self.cash
        if self.position is not None:
            unrealized = (
                (close - self.position.entry_price)
                * self.position.quantity
                * self.position.side
            )
            equity += unrealized

        self.equity_rows.append({
            "timestamp": time,
            "cash": self.cash,
            "equity": equity,
            "close": close,
            "position_side": (
                "LONG" if self.position and self.position.side == 1
                else "SHORT" if self.position
                else "FLAT"
            ),
            "position_quantity": (
                self.position.quantity if self.position else 0.0
            ),
        })

    # -------------------------- backtest --------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        bars = self.data
        total = len(bars)
        started = time.perf_counter()
        last_log = -1
        print(f"[BACKTEST] Iniciando: {total:,} velas 1m", flush=True)
        print(
            f"[BACKTEST] Rango: {bars.index.min()} -> {bars.index.max()}",
            flush=True,
        )

        # Cada iteración representa el cierre de la vela 1m cuyo timestamp
        # es su inicio. La señal se conoce en timestamp + 1m y se ejecuta
        # en el siguiente bar disponible.
        for i in range(len(bars)):
            ts = bars.index[i]
            row = bars.iloc[i]

            bar_open = float(row["open"])
            bar_close = float(row["close"])

            # Si había una señal generada al cierre anterior, se ejecuta
            # aquí, en la apertura de la vela actual.
            self._execute_pending(bar_open, ts)

            # Stop-loss intrabar de 1m. Tiene prioridad sobre cualquier
            # nueva señal de la estrategia en esta misma vela.
            stop_hit = self._check_stop_loss(
                bar_open=bar_open,
                bar_high=float(row["high"]),
                bar_low=float(row["low"]),
                bar_time=ts,
            )

            close_time = ts + pd.Timedelta(minutes=1)

            # Evaluación exclusivamente con datos cuya vela ya terminó en
            # close_time. Si el stop acaba de cerrar la posición, no se
            # agenda otra señal usando la misma vela.
            signal = None if stop_hit else self.evaluate_signal(close_time)

            # No puede ejecutarse inmediatamente: se agenda para la próxima
            # apertura 1m. Si no existe próxima vela, se gestionará al final.
            if signal is not None:
                self.pending_signal = signal

            self.last_price = bar_close
            self.mark_to_market(close_time, bar_close)

            # Progreso aproximadamente cada 5%, sin inundar la consola.
            pct = int(((i + 1) / total) * 100) if total else 100
            if pct >= last_log + 5 or i == total - 1:
                elapsed = time.perf_counter() - started
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (total - i - 1) / rate if rate > 0 else 0.0
                print(
                    f"[BACKTEST] {pct:3d}% | {i + 1:,}/{total:,} | "
                    f"{rate:,.0f} velas/s | transcurrido {elapsed:.1f}s | "
                    f"ETA {eta:.1f}s | trades {len(self.trades)}",
                    flush=True,
                )
                last_log = pct

        # Si queda posición abierta, se cierra al último close.
        if self.position is not None and self.cfg.force_close_at_end:
            last_ts = bars.index[-1] + pd.Timedelta(minutes=1)
            self._close_fraction(
                fraction=1.0,
                price=float(bars["close"].iloc[-1]),
                time=last_ts,
                reason="END_OF_DATA",
            )
            self.mark_to_market(last_ts, float(bars["close"].iloc[-1]))

        trades = pd.DataFrame([asdict(t) for t in self.trades])
        equity = pd.DataFrame(self.equity_rows)
        elapsed = time.perf_counter() - started
        print(
            f"[BACKTEST] Finalizado en {elapsed:.2f}s | "
            f"trades={len(trades):,} | equity_rows={len(equity):,}",
            flush=True,
        )

        return trades, equity


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def calculate_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
) -> dict:
    if equity.empty:
        return {
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "avg_trade_net": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    final_equity = float(equity["equity"].iloc[-1])
    net_pnl = final_equity - initial_capital
    return_pct = (
        net_pnl / initial_capital * 100.0
        if initial_capital else 0.0
    )

    eq = equity["equity"].astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd_pct = float(dd.min() * 100.0)

    if trades.empty:
        wins = losses = 0
        win_rate = 0.0
        profit_factor = None
        avg_trade = best = worst = 0.0
    else:
        pnl = trades["pnl_net"].astype(float)
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        win_rate = wins / len(pnl) * 100.0

        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0 else None
        )

        avg_trade = float(pnl.mean())
        best = float(pnl.max())
        worst = float(pnl.min())

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "max_drawdown_pct": max_dd_pct,
        "trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_net": avg_trade,
        "best_trade": best,
        "worst_trade": worst,
    }


def print_report(metrics: dict, config: Config, data: pd.DataFrame):
    print("\n" + "=" * 72)
    print("STRATEGY 1 — BACKTEST")
    print("=" * 72)
    print(f"Periodo       : {data.index[0]} -> {data.index[-1]}")
    print(f"Barras 1m     : {len(data):,}")
    print(f"Capital inicial: {metrics['initial_capital']:,.2f}")
    print(f"Capital final  : {metrics['final_equity']:,.2f}")
    print(f"PnL neto       : {metrics['net_pnl']:,.2f}")
    print(f"Retorno        : {metrics['return_pct']:.2f}%")
    print(f"Max drawdown   : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Trades         : {metrics['trades']:,}")
    print(f"Ganadores      : {metrics['wins']:,}")
    print(f"Perdedores     : {metrics['losses']:,}")
    print(f"Win rate       : {metrics['win_rate_pct']:.2f}%")

    pf = metrics["profit_factor"]
    print(
        f"Profit factor  : "
        f"{pf:.3f}" if pf is not None else "Profit factor  : N/A"
    )
    print(f"Avg trade      : {metrics['avg_trade_net']:,.4f}")
    print(f"Mejor trade    : {metrics['best_trade']:,.4f}")
    print(f"Peor trade     : {metrics['worst_trade']:,.4f}")

    print("\nParámetros:")
    print(f"  EMA           : {config.ema_fast}/{config.ema_slow}")
    print(f"  ADX           : {config.adx_dilen}/{config.adx_len}")
    print(f"  ADX threshold : {config.adx_threshold}")
    print(f"  Quantity      : {config.quantity}")
    print(f"  Scale-out     : {config.scale_out_fraction:.2%}")
    print(f"  Commission    : {config.commission_bps} bps")
    print(f"  Slippage      : {config.slippage_bps} bps")
    print(f"  Stop loss     : {config.stop_loss_pct * 100:.3f}%")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Backtester dedicado para Strategy1 sobre OHLCV 1m Parquet."
    )
    p.add_argument("parquet", help="Ruta al dataset .parquet")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--quantity", type=float, default=1.0)
    p.add_argument("--scale-out", type=float, default=0.5)
    p.add_argument("--adx-threshold", type=float, default=25.0)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--stop-loss-pct", type=float, default=0.006,
                    help="Stop loss como fracción del precio de entrada (default: 0.006 = 0.6%%)")
    p.add_argument(
        "--no-force-close",
        action="store_true",
        help="No cerrar la posición al final del dataset.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directorio donde guardar trades.csv, equity.csv y metrics.json.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    config = Config(
        initial_capital=args.capital,
        quantity=args.quantity,
        scale_out_fraction=args.scale_out,
        adx_threshold=args.adx_threshold,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss_pct,
        force_close_at_end=not args.no_force_close,
    )

    data = load_parquet(args.parquet)

    if len(data) < 100:
        raise ValueError("El dataset tiene muy pocas velas 1m para este backtest.")

    engine = Strategy1Backtester(data, config)
    trades, equity = engine.run()

    metrics = calculate_metrics(
        trades=trades,
        equity=equity,
        initial_capital=config.initial_capital,
    )

    print_report(metrics, config, data)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        trades.to_csv(out / "trades.csv", index=False)
        equity.to_csv(out / "equity.csv", index=False)

        import json
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        with open(out / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)

        print(f"\nResultados guardados en: {out.resolve()}")


if __name__ == "__main__":
    main()
