#!/usr/bin/env python3
"""
Strategy 2 — EMA Cross + ADX / EMA Weakening
============================================

Gestión de salida por defecto: take-profit parcial a 1R (50%) y stop a
breakeven del resto (`partial_tp_enabled`). Sube el win rate y recorta la
racha de pérdidas en whipsaw; puede desactivarse con `partial_tp_enabled=False`.

Además, tras una entrada por cross EMA, un toque a la EMA slow cierra una parte
configurable de la posición (`close_on_ema_slow_touch_exit_pct`, 100 = cierre
total) una sola vez por posición. La tolerancia `close_on_ema_slow_touch_pct`
desplaza el nivel de disparo; con `close_on_ema_slow_touch_after=False` (default)
se aplica antes del toque y con `True` después.

Archivo único con API reutilizable para backtesting y bots.

Protecciones de riesgo (ver `Config`; las de régimen y tamaño adaptativo
arrancan desactivadas, la de racha y la mensual vienen activas):

* `streak_trend_only` (default True): tras una posición cerrada en pérdida,
  la siguiente entrada sólo se abre si la señal coincide con el sesgo de
  tendencia de `trend_timeframe` (vela superior ya cerrada, sin lookahead).
  Corta las rachas de -1R del whipsaw sin recortar los cruces en tendencia.
* `monthly_loss_limit_pct` (default 4.5): si la equity cae más de X% por
  debajo de la equity al inicio del mes calendario se cierra la posición (opcional)
  y no se abren nuevas hasta el siguiente mes (o `monthly_loss_limit_days` días).
* `consecutive_loss_pause` (default 0) + `consecutive_loss_pause_bars` (96):
  espera de 48h (en 30min) antes de volver a entrar tras cada posición cerrada
  en pérdida. Con `streak_trend_only` activo no baja la racha y cuesta
  rendimiento, así que arranca apagada.
* `trend_filter` ("off" | "cross" | "slow") y `chop_filter`: filtro de régimen
  permanente; sólo se opera en la dirección de la tendencia de `trend_timeframe`
  / con el ADX de `chop_timeframe` por encima del mínimo.
* `adaptive_risk_window`: escala `risk_pct` con el win rate de las últimas n
  posiciones (penaliza rachas frías y reutiliza en cálidas).

Sobre BTCUSDT 2020-2026 (30min), con los defaults, frente a no tener
protección alguna: racha máxima de pérdidas 7 -> 5 (lo mismo en train y en
test), peor mes -4,75% -> -2,65%, PF 1,76 -> 2,07, expectancy 0,29R -> 0,40R,
max drawdown -10,95% -> -10,68% y rendimiento total 327,6% -> 362,5%.

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
from dataclasses import asdict
from typing import Optional
from strategy.config import Config
from strategy.models import (
    Position,
    Trade,
    CROSS_REASON,
    MONTHLY_LIMIT_REASON,
    DEFAULT_OUTPUT_DIR,
    empty_trades,
)
from strategy.indicators import (
    ema,
    adx,
    confirmed_crosses,
    align_closed_values,
    build_adx_exit,
    build_weakening_exit,
)
from strategy.utils import (
    load_parquet,
    resample_ohlcv,
    to_timeframe
)
from strategy.report import generate_report
import argparse
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Estrategia: compartida por backtest y bots
# ---------------------------------------------------------------------------

class EMACrossStrategy:
    """Generador de señales sin estado de cartera ni ejecución de órdenes."""

    def __init__(self, config: Config):
        self.cfg = config

    def _upper_bars(self, bars: pd.DataFrame, timeframe: str) -> Optional[tuple[pd.DataFrame, str]]:
        """Velas del timeframe superior alineadas a ``bars`` (o ``None``)."""
        if pd.Timedelta(timeframe) <= pd.Timedelta(self.cfg.timeframe):
            upper, upper_tf = bars, self.cfg.timeframe
        else:
            try:
                upper, upper_tf = resample_ohlcv(bars, timeframe), timeframe
            except ValueError:
                return None
        if len(upper) < 2:
            return None
        return upper, upper_tf

    def trend_bias(self, bars: pd.DataFrame) -> np.ndarray:
        """Sesgo de tendencia (+1 alcista, -1 bajista, 0 sin información).

        Calcula el sesgo en ``cfg.trend_timeframe`` y lo alinea a cada barra del
        timeframe de la estrategia usando **sólo la vela superior ya cerrada**
        (ver :func:`align_closed_values`), de modo que no hay lookahead. Con
        ``cfg.trend_filter == "off"`` devuelve todo ceros (sin filtro).
        """
        cfg = self.cfg
        if (cfg.trend_filter == "off" and not cfg.streak_trend_only) or not len(bars):
            return np.zeros(len(bars), int)
        upper = self._upper_bars(bars, cfg.trend_timeframe)
        if upper is None:
            return np.zeros(len(bars), int)
        bars_upper, upper_tf = upper
        close = bars_upper.close.to_numpy(float)
        if cfg.trend_filter == "slow":
            level = ema(bars_upper.close, cfg.ema_slow).to_numpy(float)
            direction = np.where(close > level, 1.0, -1.0)
        else:
            fast = ema(bars_upper.close, cfg.ema_fast).to_numpy(float)
            slow = ema(bars_upper.close, cfg.ema_slow).to_numpy(float)
            direction = np.sign(fast - slow)
        aligned = align_closed_values(bars_upper.index, direction, bars.index, upper_tf, cfg.timeframe)
        return np.where(np.isfinite(aligned), np.sign(aligned), 0.0).astype(int)

    def chop_ok(self, bars: pd.DataFrame) -> np.ndarray:
        """``True`` donde el ADX de ``cfg.chop_timeframe`` (ya cerrado) supera el mínimo."""
        cfg = self.cfg
        if not cfg.chop_filter or not len(bars):
            return np.ones(len(bars), bool)
        upper = self._upper_bars(bars, cfg.chop_timeframe)
        if upper is None:
            return np.zeros(len(bars), bool)
        bars_upper, upper_tf = upper
        needed = cfg.adx_dilen + cfg.adx_adxlen + 2
        if len(bars_upper) < needed:
            return np.zeros(len(bars), bool)
        values = adx(bars_upper, cfg.adx_dilen, cfg.adx_adxlen)["adx"].to_numpy(float)
        aligned = align_closed_values(bars_upper.index, values, bars.index, upper_tf, cfg.timeframe)
        return np.nan_to_num(aligned, nan=-np.inf) >= cfg.chop_adx_min

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
        bias = self.trend_bias(bars) if (cfg.trend_filter != "off" or cfg.streak_trend_only) else None
        if bias is not None and cfg.trend_filter != "off":
            up, down = up & (bias > 0), down & (bias < 0)
        chop = self.chop_ok(bars) if cfg.chop_filter else None
        if chop is not None:
            up, down = up & chop, down & chop
        up &= valid
        down &= valid
        return pd.DataFrame({
            "ema_fast": fast, "ema_slow": slow, "adx_entry": entry_adx,
            "cross_up": up, "cross_down": down, "valid": valid,
            "trend_bias": bias if bias is not None else np.zeros(len(bars), int),
            "chop_ok": chop if chop is not None else np.ones(len(bars), bool),
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
        # Protección de riesgo: estado del límite mensual, de la pausa por
        # racha de pérdidas y del pnl acumulado de la posición abierta.
        self.halt_until: Optional[pd.Timestamp] = None
        self.pause_until = -1
        self._bar_index = -1
        self.consec_losses = 0
        self.skip_signals_left = 0
        self.streak_in_loss = False
        self._position_pnl = 0.0
        self._recent_wins: list[bool] = []

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
        by_risk = self.cash * self.cfg.risk_pct * self._risk_multiplier() / risk_unit if risk_unit > 0 else np.inf
        by_lev = self.cash * self.cfg.max_leverage / price
        qty = min(by_risk, by_lev)
        if not np.isfinite(qty) or qty <= 0:
            return
        fee = self._fee(price * qty)
        self.cash -= fee
        self.position_id += 1
        self._position_pnl = 0.0
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
        self._position_pnl += net
        self.trade_id += 1
        self.trades.append(Trade(self.trade_id, pos.position_id, "LONG" if side == 1 else "SHORT",
            pos.entry_time, ts, pos.entry_price, price, qty, gross, fees, net,
            net / (pos.entry_price * qty) * 100 if pos.entry_price * qty else 0.0,
            reason, pos.risk_amount))
        remaining = pos.quantity - qty
        if percent >= 100 or remaining <= max(1e-12, pos.initial_quantity * 1e-12):
            self.position = None
            self._close_streak()
        else:
            pos.quantity = remaining
            pos.entry_fee -= entry_fee

    def _risk_multiplier(self) -> float:
        """Multiplicador de `risk_pct` según el resultado de las últimas posiciones.

        Con `adaptive_risk_window` a 0 devuelve 1 (tamaño fijo). Si no,
        ``1 + k * (win_rate de las últimas n posiciones - 0.5)`` recortado a
        ``[adaptive_risk_min, adaptive_risk_max]``: penaliza las rachas frías
        (donde salen los meses rojos) y reutiliza el tamaño en las cálidas.
        """
        cfg = self.cfg
        if cfg.adaptive_risk_window <= 0 or not self._recent_wins:
            return 1.0
        wins = self._recent_wins[-cfg.adaptive_risk_window:]
        win_rate = sum(wins) / len(wins)
        factor = 1 + cfg.adaptive_risk_k * (win_rate - 0.5)
        return float(min(max(factor, cfg.adaptive_risk_min), cfg.adaptive_risk_max))

    def _close_streak(self) -> None:
        """Actualiza la racha de pérdidas y arma la pausa de entrada si procede.

        Una posición se considera en pérdida si el pnl neto acumulado de todos
        sus tramos es negativo. Al alcanzar ``consecutive_loss_pause`` posiciones
        seguidas en pérdida se bloquean las entradas durante
        ``consecutive_loss_pause_bars`` velas.
        """
        cfg = self.cfg
        won = self._position_pnl >= 0
        if cfg.adaptive_risk_window > 0:
            self._recent_wins.append(won)
            del self._recent_wins[:-cfg.adaptive_risk_window]
        if not won:
            self.consec_losses += 1
            self.skip_signals_left = max(self.skip_signals_left, cfg.consecutive_loss_skip_signals)
            self.streak_in_loss = True
        else:
            self.consec_losses = 0
            self.skip_signals_left = 0
            self.streak_in_loss = False
        self._position_pnl = 0.0
        if cfg.consecutive_loss_pause > 0 and self.consec_losses >= cfg.consecutive_loss_pause:
            self.pause_until = self._bar_index + cfg.consecutive_loss_pause_bars
            self.consec_losses = 0
            if self.verbose:
                print(f"[PAUSA] {cfg.consecutive_loss_pause} pérdidas seguidas: "
                      f"sin entradas {cfg.consecutive_loss_pause_bars} velas", flush=True)

    def entries_blocked(self) -> bool:
        """``True`` si está prohibido abrir posiciones nuevas en esta vela."""
        if self.pause_until is not None and self._bar_index <= self.pause_until:
            return True
        return self.halt_until is not None and self.times[self._bar_index] < self.halt_until

    def _risk_unit(self, pos: Position) -> float:
        return pos.risk_amount / pos.initial_quantity if pos.initial_quantity else 0.0

    def _ema_slow_touch_level(self, i: int) -> Optional[float]:
        """Nivel de cierre por toque a la EMA slow en la vela ``i``.

        Devuelve el precio de disparo con la tolerancia ya aplicada, o ``None``
        si el cierre por toque no está activo. Con ``close_on_ema_slow_touch_after``
        en False (default) la tolerancia se aplica ANTES del toque: para LONG el
        disparo es ``EMA_slow * (1 + tol)`` y para SHORT ``EMA_slow * (1 - tol)``.
        Con el flag en True se aplica DESPUÉS del toque: LONG ``EMA_slow * (1 - tol)``
        y SHORT ``EMA_slow * (1 + tol)``. Con tolerancia 0 es la propia EMA slow.
        """
        cfg, pos = self.cfg, self.position
        if pos is None or not cfg.close_on_ema_slow_touch or self.signals is None:
            return None
        if "ema_slow" not in self.signals.columns or i >= len(self.signals):
            return None
        ema_slow = float(self.signals["ema_slow"].iloc[i])
        if not np.isfinite(ema_slow) or ema_slow <= 0:
            return None
        tol = max(cfg.close_on_ema_slow_touch_pct, 0.0) / 100.0
        offset = -tol if cfg.close_on_ema_slow_touch_after else tol
        return ema_slow * (1 + offset) if pos.side == 1 else ema_slow * (1 - offset)

    def _ema_slow_touch(self, i: int) -> bool:
        """Salida (total o parcial) si el precio toca la EMA slow tras el cross.

        El parámetro `close_on_ema_slow_touch_pct` define la tolerancia en % y
        `close_on_ema_slow_touch_after` su sentido. Por defecto (False) la
        tolerancia se aplica ANTES del toque: para LONG se dispara cuando el
        mínimo llega a EMA_slow * (1 + tol), y para SHORT cuando el máximo llega
        a EMA_slow * (1 - tol). Con el flag en True se aplica DESPUÉS del toque:
        LONG cuando el mínimo llega a EMA_slow * (1 - tol) y SHORT cuando el
        máximo llega a EMA_slow * (1 + tol). Con tolerancia 0 el disparo ocurre
        justo al tocar la EMA slow.

        Se cierra `close_on_ema_slow_touch_exit_pct` de la posición: con 100 es
        un cierre total; con un valor menor es una **salida parcial** que sólo
        se aplica una vez por posición (`pos.touch_done`); el resto sigue
        gestionándose con las reglas habituales (stop, TP, cross, ADX, …).
        """
        cfg, pos = self.cfg, self.position
        trigger = self._ema_slow_touch_level(i)
        if pos is None or trigger is None or pos.touch_done:
            return False
        if pos.side == 1:
            if self.l[i] > trigger:
                return False
            fill = min(self.o[i], trigger)
        else:
            if self.h[i] < trigger:
                return False
            fill = max(self.o[i], trigger)
        prefix = "EMA_SLOW_TOUCH_AFTER" if cfg.close_on_ema_slow_touch_after else "EMA_SLOW_TOUCH"
        self.close_position(cfg.close_on_ema_slow_touch_exit_pct, fill, self.times[i],
                            f"{prefix}_{cfg.close_on_ema_slow_touch_pct:g}%")
        pos = self.position
        if pos is not None:
            pos.touch_done = True
        return True

    def _manage(self, i: int) -> bool:
        """Take-profit parcial y salida por toque a la EMA slow.

        Devuelve ``True`` si el toque a la EMA slow cerró (parte de) la posición,
        para que `run` pueda resolver el resto del stop en la misma vela.
        """
        cfg, pos = self.cfg, self.position
        if pos is None:
            return False
        if self._ema_slow_touch(i):
            return True
        if not cfg.partial_tp_enabled:
            return False
        risk_unit = self._risk_unit(pos)
        if risk_unit <= 0:
            return False
        pos_after = pos
        # TP2 primero si no hecho y trigger alcanzado (mayor distancia)
        if cfg.partial_tp2_r > 0 and not pos.partial2_done:
            trigger2 = pos.entry_price + pos.side * cfg.partial_tp2_r * risk_unit
            hit2 = self.h[i] >= trigger2 if pos.side == 1 else self.l[i] <= trigger2
            if hit2:
                fill2 = max(trigger2, self.o[i]) if pos.side == 1 else min(trigger2, self.o[i])
                self.close_position(cfg.partial_tp2_pct, fill2, self.times[i], f"PARTIAL_TP2_{cfg.partial_tp2_r:g}R")
                pos_after = self.position
                if pos_after is not None:
                    pos_after.partial2_done = True
                return False
        # TP1
        if not pos.partial_done:
            trigger = pos.entry_price + pos.side * cfg.partial_tp_r * risk_unit
            hit = self.h[i] >= trigger if pos.side == 1 else self.l[i] <= trigger
            if hit:
                fill = max(trigger, self.o[i]) if pos.side == 1 else min(trigger, self.o[i])
                self.close_position(cfg.partial_tp_pct, fill, self.times[i], f"PARTIAL_TP_{cfg.partial_tp_r:g}R")
                pos_after = self.position
                if pos_after is not None:
                    pos_after.partial_done = True
                    if cfg.move_stop_to_breakeven:
                        pos_after.breakeven = True
        return False

    def _stop(self, i: int, touch_level: Optional[float] = None) -> bool:
        pos = self.position
        if pos is None or pos.stop_pct <= 0: return False
        stop = pos.entry_price if pos.breakeven else self._stop_price(pos.side, pos.entry_price, pos.stop_pct)
        hit = self.l[i] <= stop if pos.side == 1 else self.h[i] >= stop
        if not hit: return False
        # Si en la misma vela también se toca la EMA slow y ésta se alcanza antes
        # que el stop (está más cerca de la entrada), gana el toque: el stop se
        # pospone y `_manage` cerrará por EMA_SLOW_TOUCH. En salida parcial el
        # resto se cierra al stop en la misma vela (ver `run`).
        if touch_level is not None and not pos.touch_done:
            touch_hit = self.l[i] <= touch_level if pos.side == 1 else self.h[i] >= touch_level
            touch_first = touch_level >= stop if pos.side == 1 else touch_level <= stop
            if touch_hit and touch_first:
                return False
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

    def _equity_value(self, close: float) -> float:
        """Equity mark-to-market al precio ``close`` (caja + posición a precio de liquidación)."""
        pos = self.position
        value = self.cash
        if pos is not None:
            liq = self._price(close, -pos.side)
            value += (liq - pos.entry_price) * pos.quantity * pos.side - self._fee(liq * pos.quantity)
        return value

    def _mark(self, ts: pd.Timestamp, close: float):
        pos = self.position
        value = self._equity_value(close)
        self.eq_time.append(ts); self.eq_cash.append(self.cash); self.eq_value.append(value)
        self.eq_side.append("FLAT" if pos is None else ("LONG" if pos.side == 1 else "SHORT"))
        self.eq_qty.append(0.0 if pos is None else pos.quantity)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        n, cfg = len(self.bars), self.cfg
        if not n:
            return empty_trades(), pd.DataFrame(columns=["timestamp", "cash", "equity", "position_side", "position_quantity"])
        adx_labels = "_".join(f"{int(pd.Timedelta(tf).total_seconds() // 60)}M"
                              for tf in cfg.adx_timeframes)
        adx_base = f"ADX_{adx_labels}_REVERSAL"
        adx_reason = adx_base if cfg.adx_exit_pct >= 100 else f"{adx_base}_{cfg.adx_exit_pct:g}%"
        weak_tfs = cfg.ema_weakening_timeframes or (cfg.timeframe,)
        weak_base = "EMA_WEAKENING_" + "_".join(tf.upper() for tf in weak_tfs)
        weak_reason = weak_base if cfg.ema_weakening_exit_pct >= 100 else f"{weak_base}_{cfg.ema_weakening_exit_pct:g}%"
        started = time.perf_counter()
        print(f"[BACKTEST] {n:,} velas {cfg.timeframe}", flush=True)
        month_key: Optional[tuple[int, int]] = None
        month_start_equity = cfg.initial_capital
        for i in range(n):
            self._bar_index = i
            # Reinicio del límite mensual en cada mes calendario; la referencia
            # es la equity mark-to-market del cierre de la vela anterior.
            key = (self.times[i].year, self.times[i].month)
            if key != month_key:
                month_key = key
                self.halt_until = None
                month_start_equity = self._equity_value(self.c[i - 1]) if i else cfg.initial_capital
            if self.pending_target is not None:
                self._execute_target(i, self.pending_target)
                self.pending_target = None
                self.pending_exit = None
            elif self.pending_exit is not None:
                reason, pct = self.pending_exit
                self.close_position(pct, self.o[i], self.times[i], reason)
                self.pending_exit = None

            stop_hit = self._stop(i, self._ema_slow_touch_level(i))
            if not stop_hit:
                # Tras una salida parcial por toque a la EMA slow, si la vela
                # también recorrió hasta el stop, se cierra el resto al stop.
                if self._manage(i) and self.position is not None:
                    stop_hit = self._stop(i)

            # Límite de pérdida mensual: si la equity cae por debajo del umbral
            # respecto al inicio del mes, se protege el mes completo (se cierra
            # la posición, si la hay, y no se abren nuevas hasta el próximo mes).
            if cfg.monthly_loss_limit_pct > 0 and self.halt_until is None:
                if self._equity_value(self.c[i]) <= month_start_equity * (1 - cfg.monthly_loss_limit_pct / 100):
                    if cfg.monthly_loss_limit_days > 0:
                        self.halt_until = self.times[i] + pd.Timedelta(days=cfg.monthly_loss_limit_days)
                        scope = f"{cfg.monthly_loss_limit_days} días"
                    else:
                        self.halt_until = pd.Timestamp.max.tz_localize(self.times[i].tz) if self.times[i].tz else pd.Timestamp.max
                        scope = "resto del mes"
                    if cfg.monthly_loss_limit_close and self.position is not None:
                        reason = f"{MONTHLY_LIMIT_REASON}_{cfg.monthly_loss_limit_pct:g}%"
                        self.close_position(100, self.c[i], self.times[i], reason)
                    print(f"[GUARD] {self.times[i]:%Y-%m-%d} límite "
                          f"-{cfg.monthly_loss_limit_pct:g}% alcanzado; sin entradas hasta "
                          f"{scope}", flush=True)

            if not stop_hit and bool(self.signals["valid"].iloc[i]):
                target = int(self.signals["target"].iloc[i])
                if target == 1 and not cfg.allow_long: target = 0
                if target == -1 and not cfg.allow_short: target = 0
                current = self.position.side if self.position else 0
                if target and target != current and not self.entries_blocked():
                    if self.skip_signals_left > 0:
                        # Tras una pérdida se ignoran las primeras señales: en
                        # whipsaw la siguiente señal suele ser otra trampa.
                        self.skip_signals_left -= 1
                        if self.verbose:
                            print(f"[SKIP] {self.times[i]} señal ignorada "
                                  f"({self.skip_signals_left} restantes)", flush=True)
                    elif (cfg.streak_trend_only and self.streak_in_loss
                          and int(self.signals["trend_bias"].iloc[i]) != target):
                        # Tras una pérdida sólo se entra con la tendencia mayor.
                        if self.verbose:
                            print(f"[RACHA] {self.times[i]} señal {target:+d} "
                                  f"contra tendencia {int(self.signals['trend_bias'].iloc[i]):+d}: ignorada",
                                  flush=True)
                    else:
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
    p.add_argument("--adx-timeframe", default=d.adx_timeframe,
                   help="Timeframe(s) ADX de salida separados por coma; se combinan con OR (p. ej. 1D,30min).")
    p.add_argument("--adx-dilen", type=int, default=d.adx_dilen)
    p.add_argument("--adx-adxlen", type=int, default=d.adx_adxlen)
    p.add_argument("--adx-key-level", type=float, default=d.adx_key_level)
    p.add_argument("--adx-exit-pct", type=float, default=d.adx_exit_pct)
    p.add_argument("--adx-exit-range", type=int, default=d.adx_exit_range,
                   help="Velas durante las que una reversión ADX mantiene activada la salida (default 4).")
    p.add_argument("--ema-weakening-exit", action=argparse.BooleanOptionalAction, default=d.ema_weakening_exit_enabled)
    p.add_argument("--ema-weakening-timeframe", default=d.ema_weakening_timeframe,
                   help="Timeframe(s) EMA weakening de salida separados por coma; se combinan con OR (p. ej. 1D,30min).")
    p.add_argument("--ema-weakening-gap-pct", type=float, default=d.ema_weakening_gap_pct)
    p.add_argument("--ema-weakening-exit-pct", type=float, default=d.ema_weakening_exit_pct)
    p.add_argument("--partial-tp", action=argparse.BooleanOptionalAction, default=d.partial_tp_enabled)
    p.add_argument("--partial-tp-r", type=float, default=d.partial_tp_r)
    p.add_argument("--partial-tp-pct", type=float, default=d.partial_tp_pct)
    p.add_argument("--partial-tp2-r", type=float, default=d.partial_tp2_r)
    p.add_argument("--partial-tp2-pct", type=float, default=d.partial_tp2_pct)
    p.add_argument("--breakeven-stop", action=argparse.BooleanOptionalAction, default=d.move_stop_to_breakeven)
    p.add_argument("--close-on-ema-slow-touch", action=argparse.BooleanOptionalAction, default=d.close_on_ema_slow_touch)
    p.add_argument("--close-on-ema-slow-touch-pct", type=float, default=d.close_on_ema_slow_touch_pct,
                   help="Tolerancia en %% alrededor de la EMA slow (0.10 = 0.10%%).")
    p.add_argument("--close-on-ema-slow-touch-after", action=argparse.BooleanOptionalAction,
                   default=d.close_on_ema_slow_touch_after,
                   help="Aplica la tolerancia después del toque a la EMA slow (por defecto, antes).")
    p.add_argument("--close-on-ema-slow-touch-exit-pct", type=float, default=d.close_on_ema_slow_touch_exit_pct,
                   help="%% de la posición a cerrar al tocar la EMA slow (100 = cierre total, < 100 = parcial).")
    p.add_argument("--no-long", action="store_true")
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--trend-filter", choices=("off", "cross", "slow"), default=d.trend_filter,
                   help="Filtro de régimen: entradas sólo en la dirección de la tendencia "
                        "del timeframe superior ('cross' = cruce EMA, 'slow' = EMA slow).")
    p.add_argument("--trend-timeframe", default=d.trend_timeframe,
                   help="Timeframe del filtro de régimen (default 1D).")
    p.add_argument("--chop-filter", action=argparse.BooleanOptionalAction, default=d.chop_filter,
                   help="Sólo entra si el ADX de --chop-timeframe supera --chop-adx-min.")
    p.add_argument("--chop-timeframe", default=d.chop_timeframe)
    p.add_argument("--chop-adx-min", type=float, default=d.chop_adx_min)
    p.add_argument("--monthly-loss-limit-pct", type=float, default=d.monthly_loss_limit_pct,
                   help="%% máxima de pérdida mensual antes de dejar de operar el resto del mes "
                        "(0 = desactivado).")
    p.add_argument("--monthly-loss-limit-close", action=argparse.BooleanOptionalAction,
                   default=d.monthly_loss_limit_close,
                   help="Cierra la posición abierta al alcanzar el límite mensual.")
    p.add_argument("--monthly-loss-limit-days", type=int, default=d.monthly_loss_limit_days,
                   help="Días sin entradas tras el límite mensual (0 = hasta el próximo mes).")
    p.add_argument("--consecutive-loss-pause", type=int, default=d.consecutive_loss_pause,
                   help="Pausa las entradas tras N posiciones consecutivas en pérdida (0 = desactivado).")
    p.add_argument("--consecutive-loss-pause-bars", type=int, default=d.consecutive_loss_pause_bars,
                   help="Velas sin entradas tras la racha de pérdidas (default 96 = 48h en 30min).")
    p.add_argument("--consecutive-loss-skip-signals", type=int, default=d.consecutive_loss_skip_signals,
                   help="Señales de entrada a ignorar tras cada pérdida (0 = desactivado).")
    p.add_argument("--streak-trend-only", action=argparse.BooleanOptionalAction, default=d.streak_trend_only,
                   help="Tras una pérdida, sólo entrar si la señal coincide con la tendencia mayor.")
    p.add_argument("--adaptive-risk-window", type=int, default=d.adaptive_risk_window,
                   help="Nº de últimas posiciones para escalar el riesgo (0 = tamaño fijo).")
    p.add_argument("--adaptive-risk-k", type=float, default=d.adaptive_risk_k)
    p.add_argument("--adaptive-risk-min", type=float, default=d.adaptive_risk_min)
    p.add_argument("--adaptive-risk-max", type=float, default=d.adaptive_risk_max)
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
        adx_exit_range=a.adx_exit_range,
        ema_weakening_exit_enabled=a.ema_weakening_exit, ema_weakening_timeframe=a.ema_weakening_timeframe,
        ema_weakening_gap_pct=a.ema_weakening_gap_pct, ema_weakening_exit_pct=a.ema_weakening_exit_pct,
        partial_tp_enabled=a.partial_tp, partial_tp_r=a.partial_tp_r, partial_tp_pct=a.partial_tp_pct,
        partial_tp2_r=a.partial_tp2_r, partial_tp2_pct=a.partial_tp2_pct,
        move_stop_to_breakeven=a.breakeven_stop,
        close_on_ema_slow_touch=a.close_on_ema_slow_touch,
        close_on_ema_slow_touch_pct=a.close_on_ema_slow_touch_pct,
        close_on_ema_slow_touch_after=a.close_on_ema_slow_touch_after,
        close_on_ema_slow_touch_exit_pct=a.close_on_ema_slow_touch_exit_pct,
        trend_filter=a.trend_filter, trend_timeframe=a.trend_timeframe,
        chop_filter=a.chop_filter, chop_timeframe=a.chop_timeframe, chop_adx_min=a.chop_adx_min,
        monthly_loss_limit_pct=a.monthly_loss_limit_pct,
        monthly_loss_limit_close=a.monthly_loss_limit_close,
        monthly_loss_limit_days=a.monthly_loss_limit_days,
        consecutive_loss_pause=a.consecutive_loss_pause,
        consecutive_loss_pause_bars=a.consecutive_loss_pause_bars,
        consecutive_loss_skip_signals=a.consecutive_loss_skip_signals,
        streak_trend_only=a.streak_trend_only,
        adaptive_risk_window=a.adaptive_risk_window,
        adaptive_risk_k=a.adaptive_risk_k,
        adaptive_risk_min=a.adaptive_risk_min,
        adaptive_risk_max=a.adaptive_risk_max,
        force_close_at_end=not a.no_force_close)


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
    generate_report(
        trades, equity, cfg, bars, args.parquet,
        output_dir=args.output_dir,
        write_outputs_enabled=not args.no_output,
        write_equity_csv=args.equity_csv,
        plot_enabled=not args.no_plot,
        plot_path=args.plot,
        plot_dpi=args.plot_dpi,
        raw=raw,
        signals=signals,
        adx_exit=adx_exit,
        weak_exit=weak_exit,
    )


if __name__ == "__main__":
    main()