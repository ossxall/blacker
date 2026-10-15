#!/usr/bin/env python3
"""
Strategy 2 — EMA Cross + ADX / EMA Weakening
============================================

Gestión de salida por defecto: take-profit parcial a 1R (50%) y stop a
breakeven del resto (`partial_tp_enabled`). Sube el win rate y recorta la
racha de pérdidas en whipsaw; puede desactivarse con `partial_tp_enabled=False`.

Además, tras una entrada por cross EMA, un toque a la EMA slow cierra una parte
configurable de la posición (`close_on_ema_slow_touch_exit_pct`, 100 = cierre
total) una sola vez por posición.

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
from dataclasses import asdict
from typing import Optional
from strategy.config import Config
from strategy.models import (
    Position,
    Trade,
    CROSS_REASON,
    DEFAULT_OUTPUT_DIR,
    empty_trades,
)
from strategy.indicators import (
    ema,
    adx,
    confirmed_crosses,
    build_adx_exit,
    build_weakening_exit,
)
from strategy.utils import (
    load_parquet,
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

    def _ema_slow_touch_level(self, i: int) -> Optional[float]:
        """Nivel de cierre por toque a la EMA slow en la vela ``i``.

        Devuelve el precio de disparo con la tolerancia ya aplicada, o ``None``
        si el cierre por toque no está activo. Para LONG el disparo es
        ``EMA_slow * (1 + tol)`` y para SHORT ``EMA_slow * (1 - tol)``; con
        tolerancia 0 es la propia EMA slow.
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
        return ema_slow * (1 + tol) if pos.side == 1 else ema_slow * (1 - tol)

    def _ema_slow_touch(self, i: int) -> bool:
        """Salida (total o parcial) si el precio toca la EMA slow tras el cross.

        El parámetro `close_on_ema_slow_touch_pct` define la tolerancia en %:
        para LONG se dispara cuando el mínimo llega a EMA_slow * (1 + tol), y
        para SHORT cuando el máximo llega a EMA_slow * (1 - tol). Con tolerancia
        0 el disparo ocurre justo al tocar la EMA slow.

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
        self.close_position(cfg.close_on_ema_slow_touch_exit_pct, fill, self.times[i],
                            f"EMA_SLOW_TOUCH_{cfg.close_on_ema_slow_touch_pct:g}%")
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
        if not cfg.partial_tp_enabled or pos.partial_done:
            return False
        risk_unit = self._risk_unit(pos)
        if risk_unit <= 0:
            return False
        trigger = pos.entry_price + pos.side * cfg.partial_tp_r * risk_unit
        hit = self.h[i] >= trigger if pos.side == 1 else self.l[i] <= trigger
        if not hit:
            return False
        fill = max(trigger, self.o[i]) if pos.side == 1 else min(trigger, self.o[i])
        self.close_position(cfg.partial_tp_pct, fill, self.times[i], f"PARTIAL_TP_{cfg.partial_tp_r:g}R")
        pos = self.position
        if pos is not None:
            pos.partial_done = True
            if cfg.move_stop_to_breakeven:
                pos.breakeven = True
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
        adx_labels = "_".join(f"{int(pd.Timedelta(tf).total_seconds() // 60)}M"
                              for tf in cfg.adx_timeframes)
        adx_base = f"ADX_{adx_labels}_REVERSAL"
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

            stop_hit = self._stop(i, self._ema_slow_touch_level(i))
            if not stop_hit:
                # Tras una salida parcial por toque a la EMA slow, si la vela
                # también recorrió hasta el stop, se cierra el resto al stop.
                if self._manage(i) and self.position is not None:
                    stop_hit = self._stop(i)
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
    p.add_argument("--ema-weakening-timeframe", default=d.ema_weakening_timeframe)
    p.add_argument("--ema-weakening-gap-pct", type=float, default=d.ema_weakening_gap_pct)
    p.add_argument("--ema-weakening-exit-pct", type=float, default=d.ema_weakening_exit_pct)
    p.add_argument("--partial-tp", action=argparse.BooleanOptionalAction, default=d.partial_tp_enabled)
    p.add_argument("--partial-tp-r", type=float, default=d.partial_tp_r)
    p.add_argument("--partial-tp-pct", type=float, default=d.partial_tp_pct)
    p.add_argument("--breakeven-stop", action=argparse.BooleanOptionalAction, default=d.move_stop_to_breakeven)
    p.add_argument("--close-on-ema-slow-touch", action=argparse.BooleanOptionalAction, default=d.close_on_ema_slow_touch)
    p.add_argument("--close-on-ema-slow-touch-pct", type=float, default=d.close_on_ema_slow_touch_pct,
                   help="Tolerancia en %% alrededor de la EMA slow (0.10 = 0.10%%).")
    p.add_argument("--close-on-ema-slow-touch-exit-pct", type=float, default=d.close_on_ema_slow_touch_exit_pct,
                   help="%% de la posición a cerrar al tocar la EMA slow (100 = cierre total, < 100 = parcial).")
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
        adx_exit_range=a.adx_exit_range,
        ema_weakening_exit_enabled=a.ema_weakening_exit, ema_weakening_timeframe=a.ema_weakening_timeframe,
        ema_weakening_gap_pct=a.ema_weakening_gap_pct, ema_weakening_exit_pct=a.ema_weakening_exit_pct,
        partial_tp_enabled=a.partial_tp, partial_tp_r=a.partial_tp_r, partial_tp_pct=a.partial_tp_pct,
        move_stop_to_breakeven=a.breakeven_stop,
        close_on_ema_slow_touch=a.close_on_ema_slow_touch,
        close_on_ema_slow_touch_pct=a.close_on_ema_slow_touch_pct,
        close_on_ema_slow_touch_exit_pct=a.close_on_ema_slow_touch_exit_pct,
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