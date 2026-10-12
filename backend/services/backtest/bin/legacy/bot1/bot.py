#!/usr/bin/env python3
"""
Bot 1 — Estrategia EMA Cross en vivo (eventos en signal.db)
===========================================================

Lee la data OHLCV almacenada por `feed.py` en `data/feed.db` (1m por defecto) y
**replica exactamente el backtest** definido en `strategy/strategy.py`:

- Remuestrea a `STRATEGY_TIMEFRAME`, calcula señales y salidas con
  `EMACrossStrategy` y ejecuta el `Backtester` sobre las velas cerradas.
- Por cada acción que tomaría el backtest (apertura, reversal, cierre por
  EMA_CROSS, ADX_REVERSAL, EMA_WEAKENING, STOP_LOSS, BREAKEVEN, PARTIAL_TP)
  guarda un evento en `signal.db`.

Reglas de diseño
----------------
- El bot **no crea órdenes**. Solo publica los eventos con `executed = 0`
  (false); un consumidor externo decide si los ejecuta y actualiza la marca.
- Cada evento es idempotente: se deduplica por
  (symbol, interval, timeframe, event, signal, reason, bar_time).
- Las decisiones se conocen al cierre de la vela y se ejecutan en la apertura
  siguiente (igual que el backtest); por eso se guardan `bar_time` (vela de la
  señal) y `signal_time` (momento de ejecución). Las salidas por stop/breakeven/
  partial se disparan dentro de la vela, por lo que `bar_time == signal_time`.
- `force_close_at_end` se desactiva: en vivo una posición abierta no debe
  cerrarse artificialmente al final de los datos disponibles.

Orden de los eventos (para el consumidor)
-----------------------------------------
Cada posición genera un `ENTRY` y uno o más `EXIT` (partial TP, weakening,
breakeven, stop, reversal o cierre completo). Para reproducir el backtest,
ordena por `(signal_time, bar_time, event)` con `EXIT` antes que `ENTRY`:

- `ENTRY`: `signal_time = bar_time + timeframe` (se ejecuta en la apertura
  siguiente a la vela de la señal).
- `EXIT` por EMA_CROSS / ADX / EMA_WEAKENING: también `signal_time = bar_time
  + timeframe` (se decide al cierre y se ejecuta en la apertura siguiente).
- `EXIT` por STOP_LOSS / BREAKEVEN / PARTIAL_TP: ocurre dentro de la vela, por
  eso `signal_time == bar_time`. Si coincide con una apertura, la apertura va
  primero (bar_time menor).

El `payload` de cada `EXIT` incluye `action` (`CLOSE`/`PARTIAL_CLOSE`),
`close_pct` (porcentaje de la posición *restante* a cerrar, igual que el
backtest), `exit_price`, `pnl_net` y `r_multiple`.

Uso
---
    python bot.py                 # bucle continuo (polling de feed.db)
    python bot.py --once          # procesa lo disponible y termina
    python bot.py --help

Modo live-only
--------------
Con `--live-only` (o `LIVE_ONLY=true`) el bot ignora el histórico y publica
solo las señales cuya vela (`bar_time`) es igual o posterior al arranque; así
no vuelca de golpe todas las señales pasadas al ponerse en marcha. La marca de
arranque se guarda en `signal.db` (`bot_state.live_since_ms`) para no perder
señales entre reinicios. Para reanudar desde una fecha concreta:
`--live-since 2026-09-01` (o epoch en ms).

La configuración se toma de `.env` (mismas variables que `feed.py` más las
de la estrategia con prefijo `STRATEGY_`).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Estrategia compartida (strategy/strategy.py)
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent          # .../bin/legacy/bot1
_LEGACY_DIR = _BASE_DIR.parent                       # .../bin/legacy
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

from strategy.strategy import (
    Backtester,
    Config,
    EMACrossStrategy,
    resample_ohlcv,
    source_minutes,
)

try:
    from dotenv import load_dotenv

    # Prioriza el .env del propio bot y admite el del directorio actual.
    load_dotenv(_BASE_DIR / ".env", override=False)
    load_dotenv(override=False)
except ImportError:
    pass


# ===========================================================================
# Configuración
# ===========================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
)
logger = logging.getLogger("bot1")

STOP_EVENT = threading.Event()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def parse_live_since(value: str) -> int:
    """Convierte epoch en ms o una fecha ISO 8601 a epoch en ms."""
    text = value.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_path(value: str) -> Path:
    """Resuelve rutas relativas contra el directorio del bot (bot1/)."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_BASE_DIR / path)


def touch_heartbeat() -> None:
    """Registra actividad para el healthcheck del contenedor (best-effort).

    Solo actúa si `BOT_HEARTBEAT` está definido; si no, es un no-op, de modo
    que ejecuciones locales/tests no escriben nada.
    """
    path = os.getenv("BOT_HEARTBEAT")
    if not path:
        return
    try:
        Path(path).write_text(utc_iso(), encoding="utf-8")
    except OSError:
        logger.debug("No se pudo escribir el heartbeat en %s", path)


def build_strategy_config() -> Config:
    """Config de la estrategia. Los defaults coinciden con `strategy.Config`.

    Se exponen todos los parámetros para que el bot ejecute exactamente la
    misma estrategia que el backtest (señales, salidas y gestión de posición).
    """
    defaults = Config()
    return Config(
        ema_fast=env_int("STRATEGY_EMA_FAST", defaults.ema_fast),
        ema_slow=env_int("STRATEGY_EMA_SLOW", defaults.ema_slow),
        timeframe=os.getenv("STRATEGY_TIMEFRAME", defaults.timeframe),
        allow_long=env_bool("STRATEGY_ALLOW_LONG", defaults.allow_long),
        allow_short=env_bool("STRATEGY_ALLOW_SHORT", defaults.allow_short),
        confirm_bars=env_int("STRATEGY_CONFIRM_BARS", defaults.confirm_bars),
        entry_adx_enabled=env_bool("STRATEGY_ENTRY_ADX", defaults.entry_adx_enabled),
        entry_adx_len=env_int("STRATEGY_ENTRY_ADX_LEN", defaults.entry_adx_len),
        entry_adx_min=env_float("STRATEGY_ENTRY_ADX_MIN", defaults.entry_adx_min),
        # Gestión de posición: afecta el sizing y las salidas (stop/partial).
        initial_capital=env_float(
            "STRATEGY_INITIAL_CAPITAL", defaults.initial_capital
        ),
        commission_bps=env_float("STRATEGY_COMMISSION_BPS", defaults.commission_bps),
        slippage_bps=env_float("STRATEGY_SLIPPAGE_BPS", defaults.slippage_bps),
        risk_pct=env_float("STRATEGY_RISK_PCT", defaults.risk_pct),
        max_leverage=env_float("STRATEGY_MAX_LEVERAGE", defaults.max_leverage),
        stop_loss_pct=env_float("STRATEGY_STOP_LOSS_PCT", defaults.stop_loss_pct),
        partial_tp_enabled=env_bool(
            "STRATEGY_PARTIAL_TP", defaults.partial_tp_enabled
        ),
        partial_tp_r=env_float("STRATEGY_PARTIAL_TP_R", defaults.partial_tp_r),
        partial_tp_pct=env_float("STRATEGY_PARTIAL_TP_PCT", defaults.partial_tp_pct),
        move_stop_to_breakeven=env_bool(
            "STRATEGY_MOVE_STOP_TO_BREAKEVEN", defaults.move_stop_to_breakeven
        ),
        # Salidas: `prepare_exits` usa estos parámetros para generar los
        # eventos de salida, así que deben ser configurables igual que
        # en el backtest para que ambos usen la misma estrategia.
        adx_exit_enabled=env_bool("STRATEGY_ADX_EXIT", defaults.adx_exit_enabled),
        adx_timeframe=os.getenv("STRATEGY_ADX_TIMEFRAME", defaults.adx_timeframe),
        adx_dilen=env_int("STRATEGY_ADX_DILEN", defaults.adx_dilen),
        adx_adxlen=env_int("STRATEGY_ADX_ADXLEN", defaults.adx_adxlen),
        adx_key_level=env_float("STRATEGY_ADX_KEY_LEVEL", defaults.adx_key_level),
        adx_exit_pct=env_float("STRATEGY_ADX_EXIT_PCT", defaults.adx_exit_pct),
        ema_weakening_exit_enabled=env_bool(
            "STRATEGY_EMA_WEAKENING_EXIT", defaults.ema_weakening_exit_enabled
        ),
        ema_weakening_timeframe=os.getenv(
            "STRATEGY_EMA_WEAKENING_TIMEFRAME", defaults.ema_weakening_timeframe
        ),
        ema_weakening_gap_pct=env_float(
            "STRATEGY_EMA_WEAKENING_GAP_PCT", defaults.ema_weakening_gap_pct
        ),
        ema_weakening_exit_pct=env_float(
            "STRATEGY_EMA_WEAKENING_EXIT_PCT", defaults.ema_weakening_exit_pct
        ),
        timestamp_is_close=env_bool(
            "STRATEGY_TIMESTAMP_IS_CLOSE", defaults.timestamp_is_close
        ),
    )


# ===========================================================================
# signal.db — almacén de eventos
# ===========================================================================

SIGNALS_DDL = """
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    bar_time    INTEGER NOT NULL,
    signal_time INTEGER NOT NULL,
    event       TEXT    NOT NULL,
    signal      TEXT    NOT NULL,
    side        TEXT,
    reason      TEXT    NOT NULL,
    price       REAL,
    ema_fast    REAL,
    ema_slow    REAL,
    adx_entry   REAL,
    executed    INTEGER NOT NULL DEFAULT 0 CHECK (executed IN (0, 1)),
    created_at  TEXT    NOT NULL,
    payload     TEXT    NOT NULL DEFAULT '{}'
)
"""

SIGNALS_INDEXES = (
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_signals_event
    ON signals(symbol, interval, timeframe, event, signal, reason, bar_time)
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_bar_time ON signals(bar_time)",
    "CREATE INDEX IF NOT EXISTS idx_signals_signal_time ON signals(signal_time)",
    "CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed)",
)

BOT_STATE_DDL = """
CREATE TABLE IF NOT EXISTS bot_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

INSERT_SQL = """
INSERT OR IGNORE INTO signals (
    symbol, interval, timeframe, bar_time, signal_time, event, signal,
    side, reason, price, ema_fast, ema_slow, adx_entry, executed,
    created_at, payload
) VALUES (
    :symbol, :interval, :timeframe, :bar_time, :signal_time, :event, :signal,
    :side, :reason, :price, :ema_fast, :ema_slow, :adx_entry, :executed,
    :created_at, :payload
)
"""


class SignalStore:
    """Gestiona `signal.db` y la inserción idempotente de eventos."""

    def __init__(self, path: Path):
        self.path = path
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(SIGNALS_DDL)
        self.conn.execute(BOT_STATE_DDL)
        for ddl in SIGNALS_INDEXES:
            self.conn.execute(ddl)
        self.conn.commit()
        logger.info("signal.db listo en %s", self.path)

    def insert(self, events: list[dict]) -> int:
        if not events or self.conn is None:
            return 0
        cursor = self.conn.executemany(INSERT_SQL, events)
        self.conn.commit()
        return cursor.rowcount if cursor.rowcount is not None else 0

    def get_state(self, key: str) -> str | None:
        if self.conn is None:
            return None
        row = self.conn.execute(
            "SELECT value FROM bot_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        if self.conn is None:
            return
        self.conn.execute(
            "INSERT INTO bot_state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def opt_float(value) -> float | None:
    """Convierte a float JSON-safe; None si no es finito."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


# ===========================================================================
# feed.db — lectura de OHLCV
# ===========================================================================

class FeedReader:
    """Lee velas ya cerradas desde `feed.db` (solo lecturas)."""

    def __init__(self, path: Path, symbol: str, interval: str):
        self.path = path
        self.symbol = symbol
        self.interval = interval
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"No existe feed.db en {self.path}. Inicia feed.py primero."
            )
        self.conn = sqlite3.connect(self.path, timeout=30)
        # Garantiza que el bot nunca escriba en feed.db.
        self.conn.execute("PRAGMA query_only=ON")

    def load_closed(self) -> pd.DataFrame:
        """Devuelve todas las velas cerradas indexadas por timestamp UTC."""
        if self.conn is None:
            raise RuntimeError("FeedReader sin conexión.")
        query = """
            SELECT open_time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ? AND is_closed = 1
            ORDER BY open_time
        """
        frame = pd.read_sql_query(
            query, self.conn, params=(self.symbol, self.interval)
        )
        if frame.empty:
            return frame
        frame = frame.rename(columns={"open_time": "timestamp"})
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], unit="ms", utc=True
        )
        frame = (
            frame.dropna(subset=["open", "high", "low", "close"])
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .set_index("timestamp")
        )
        return frame[["open", "high", "low", "close", "volume"]]

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# ===========================================================================
# Generación de eventos
# ===========================================================================

def to_strategy_timeframe(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Remuestrea a `cfg.timeframe` sin los prints de `strategy.to_timeframe`."""
    if raw.empty:
        return raw.copy()
    if pd.Timedelta(minutes=source_minutes(raw)) >= pd.Timedelta(cfg.timeframe):
        return raw.copy()
    return resample_ohlcv(raw, cfg.timeframe)


def _reason_is_intrabar(reason: str) -> bool:
    """Stop, breakeven y take-profit parcial se disparan dentro de la vela."""
    return reason.startswith(("STOP_LOSS", "PARTIAL_TP")) or reason == "BREAKEVEN"


def build_events(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    adx_exit: np.ndarray | None = None,
    weak_exit: np.ndarray | None = None,
    cfg: Config = None,
    symbol: str = "",
    interval: str = "",
) -> list[dict]:
    """Ejecuta el backtest y traduce cada acción real a un evento.

    Reutiliza el motor `Backtester` de `strategy/strategy.py` sobre las velas
    cerradas, de modo que el bot publica exactamente las acciones que tomaría
    el backtest: aperturas (incluidas reversals), cierres por EMA_CROSS,
    ADX_REVERSAL, EMA_WEAKENING, y las salidas por STOP_LOSS, BREAKEVEN y
    PARTIAL_TP. La posición abierta al final de los datos no se cierra
    (`force_close_at_end=False`): en vivo sigue vigente.
    """
    if bars.empty or len(bars) < 2:
        return []
    if cfg is None:
        raise ValueError("cfg requerido")

    live_cfg = replace(cfg, force_close_at_end=False)
    engine = Backtester(bars, live_cfg, signals, adx_exit, weak_exit)
    # `Backtester.run` imprime progreso; lo silenciamos para no ensuciar el log.
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()

    tf_ms = int(pd.Timedelta(cfg.timeframe).total_seconds() * 1000)
    created_at = utc_iso()

    # Indicadores por barra para enriquecer los eventos.
    ema_fast = signals["ema_fast"].to_numpy(float)
    ema_slow = signals["ema_slow"].to_numpy(float)
    adx_entry = signals["adx_entry"].to_numpy(float)
    meta = {
        int(ms): (ema_fast[i], ema_slow[i], adx_entry[i])
        for i, ms in enumerate(bars.index.asi8 // 10**6)
    }

    def indicators(bar_ms: int) -> tuple[float, float, float]:
        values = meta.get(bar_ms)
        return values if values is not None else (np.nan, np.nan, np.nan)

    events: list[dict] = []

    # --- Aperturas: una por posición (cerrada o todavía abierta) ---
    entries: dict[int, tuple[str, pd.Timestamp, float]] = {}
    for trade in engine.trades:
        if trade.position_id not in entries:
            entries[trade.position_id] = (trade.side, trade.entry_time, trade.entry_price)
    open_pos = engine.position
    if open_pos is not None and open_pos.position_id not in entries:
        entries[open_pos.position_id] = (
            "LONG" if open_pos.side == 1 else "SHORT",
            open_pos.entry_time,
            open_pos.entry_price,
        )

    for position_id in sorted(entries):
        side, entry_time, entry_price = entries[position_id]
        signal_time = int(entry_time.timestamp() * 1000)
        bar_time = signal_time - tf_ms
        fast_v, slow_v, adx_v = indicators(bar_time)
        events.append(
            {
                "symbol": symbol,
                "interval": interval,
                "timeframe": cfg.timeframe,
                "bar_time": bar_time,
                "signal_time": signal_time,
                "event": "ENTRY",
                "signal": side,
                "side": side,
                "reason": "EMA_CROSS",
                "price": opt_float(entry_price),
                "ema_fast": opt_float(fast_v),
                "ema_slow": opt_float(slow_v),
                "adx_entry": opt_float(adx_v),
                "executed": 0,
                "created_at": created_at,
                "payload": json.dumps({"action": "OPEN", "position_id": position_id}),
            }
        )

    # --- Cierres: un evento por cada tramo ejecutado por el backtest ---
    legs_by_position: dict[int, list] = {}
    for trade in engine.trades:
        legs_by_position.setdefault(trade.position_id, []).append(trade)

    # Cantidad inicial de una posición que quedó abierta con cierres parciales.
    open_quantity = (
        engine.position.initial_quantity if engine.position is not None else None
    )
    open_position_id = engine.position.position_id if engine.position is not None else None

    for position_id in sorted(legs_by_position):
        legs = legs_by_position[position_id]
        # Para posiciones cerradas la suma de tramos es la cantidad inicial; en
        # una posición todavía abierta se toma del propio motor.
        if position_id == open_position_id and open_quantity:
            initial_qty = open_quantity
        else:
            initial_qty = sum(leg.quantity for leg in legs)
        remaining = initial_qty
        for leg in legs:
            signal_time = int(leg.exit_time.timestamp() * 1000)
            intrabar = _reason_is_intrabar(leg.reason)
            bar_time = signal_time if intrabar else signal_time - tf_ms
            fast_v, slow_v, adx_v = indicators(bar_time)
            # El backtest cierra `percent` de la posición *restante*; replicamos
            # ese porcentaje (el que debe aplicar el consumidor sobre su posición).
            close_pct = 100.0 * leg.quantity / remaining if remaining else 100.0
            r_multiple = (
                leg.pnl_net / leg.position_risk if leg.position_risk else None
            )
            events.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "timeframe": cfg.timeframe,
                    "bar_time": bar_time,
                    "signal_time": signal_time,
                    "event": "EXIT",
                    "signal": leg.side,
                    "side": leg.side,
                    "reason": leg.reason,
                    "price": opt_float(leg.exit_price),
                    "ema_fast": opt_float(fast_v),
                    "ema_slow": opt_float(slow_v),
                    "adx_entry": opt_float(adx_v),
                    "executed": 0,
                    "created_at": created_at,
                    "payload": json.dumps(
                        {
                            "action": "CLOSE" if close_pct >= 99.999 else "PARTIAL_CLOSE",
                            "position_id": position_id,
                            "close_pct": close_pct,
                            "exit_price": opt_float(leg.exit_price),
                            "pnl_net": opt_float(leg.pnl_net),
                            "r_multiple": opt_float(r_multiple),
                        }
                    ),
                }
            )
            remaining -= leg.quantity

    return events


# ===========================================================================
# Bot
# ===========================================================================

class SignalBot:
    def __init__(
        self,
        cfg: Config,
        feed: FeedReader,
        store: SignalStore,
        symbol: str,
        interval: str,
        min_bars: int | None = None,
        live_only: bool = False,
        live_since_ms: int | None = None,
    ):
        self.cfg = cfg
        self.feed = feed
        self.store = store
        self.symbol = symbol
        self.interval = interval
        self.strategy = EMACrossStrategy(cfg)
        self.min_bars = min_bars or (cfg.ema_slow + 2)
        self.live_only = live_only
        self.live_since_ms = live_since_ms
        self._live_resolved = False

    def _resolve_live_since(self) -> None:
        """Fija la marca desde la que se publican señales (modo live-only).

        En modo live-only se ignora el histórico: solo se publican señales
        cuya vela (`bar_time`) es igual o posterior al arranque del bot. La
        marca se guarda en `signal.db` (`bot_state`) para no perder señales
        entre reinicios.
        """
        if self._live_resolved:
            return
        self._live_resolved = True
        if not self.live_only:
            return

        if self.live_since_ms is None:
            stored = self.store.get_state("live_since_ms")
            if stored is not None:
                self.live_since_ms = int(stored)
                logger.info(
                    "Modo live-only: reanudando desde %s.",
                    ms_to_iso(self.live_since_ms),
                )
                return
            self.live_since_ms = int(utc_now().timestamp() * 1000)

        self.store.set_state("live_since_ms", str(self.live_since_ms))
        logger.info(
            "Modo live-only: solo señales desde %s (se ignora el histórico).",
            ms_to_iso(self.live_since_ms),
        )

    def process_once(self) -> int:
        """Ejecuta un ciclo: lee, calcula y publica eventos nuevos."""
        self._resolve_live_since()
        if self.feed.conn is None:
            self.feed.connect()
        raw = self.feed.load_closed()
        if raw.empty:
            logger.debug("feed.db sin velas cerradas todavía.")
            return 0

        bars = to_strategy_timeframe(raw, self.cfg)
        if len(bars) < self.min_bars:
            logger.info(
                "Insuficientes velas %s (%d < %d); esperando más datos.",
                self.cfg.timeframe,
                len(bars),
                self.min_bars,
            )
            return 0

        signals = self.strategy.generate_signals(bars)
        adx_exit, weak_exit = self.strategy.prepare_exits(raw, bars)
        events = build_events(bars, signals, adx_exit, weak_exit, self.cfg, self.symbol, self.interval)
        if self.live_since_ms is not None:
            # live-only: descarta las señales anteriores al arranque.
            events = [e for e in events if e["bar_time"] >= self.live_since_ms]
        inserted = self.store.insert(events)
        if inserted:
            logger.info(
                "Publicados %d eventos nuevos (de %d acciones del backtest).",
                inserted,
                len(events),
            )
        return inserted

    def run(self, once: bool = False, poll_seconds: float = 30.0) -> None:
        while not STOP_EVENT.is_set():
            try:
                self.process_once()
            except FileNotFoundError as exc:
                logger.warning("%s", exc)
            except Exception:
                logger.exception("Error durante el ciclo del bot")

            # Señal de vida para el healthcheck del contenedor.
            touch_heartbeat()

            if once or STOP_EVENT.wait(poll_seconds):
                break


# ===========================================================================
# CLI / arranque
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bot EMA Cross en vivo: replica el backtest y publica sus "
        "acciones como eventos en signal.db."
    )
    parser.add_argument(
        "--feed-db",
        default=os.getenv("DB_PATH", "data/feed.db"),
        help="Ruta a feed.db (por defecto DB_PATH o data/feed.db).",
    )
    parser.add_argument(
        "--signal-db",
        default=os.getenv("SIGNAL_DB", "data/signal.db"),
        help="Ruta a signal.db (por defecto SIGNAL_DB o data/signal.db).",
    )
    parser.add_argument("--symbol", default=os.getenv("SYMBOL", "BTCUSDT"))
    parser.add_argument("--interval", default=os.getenv("INTERVAL", "1m"))
    parser.add_argument("--timeframe", default=None, help="Override de STRATEGY_TIMEFRAME.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=env_float("POLL_SECONDS", 30.0),
        help="Intervalo de sondeo de feed.db en modo continuo.",
    )
    parser.add_argument("--once", action="store_true", help="Un solo ciclo y salir.")
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Publica solo señales desde el arranque (ignora el histórico).",
    )
    parser.add_argument(
        "--live-since",
        default=None,
        help="Marca temporal (epoch ms o ISO 8601) desde la que publicar; "
        "implica --live-only.",
    )
    return parser.parse_args()


def request_stop(signum=None, frame=None) -> None:
    logger.info("Señal de parada recibida; cerrando bot…")
    STOP_EVENT.set()


def main() -> int:
    args = parse_args()
    cfg = build_strategy_config()
    if args.timeframe:
        cfg.timeframe = args.timeframe

    symbol = args.symbol.upper()
    interval = args.interval
    feed_path = resolve_path(args.feed_db)
    signal_path = resolve_path(args.signal_db)

    logger.info(
        "Bot iniciado: symbol=%s interval=%s timeframe=%s feed=%s signal=%s",
        symbol,
        interval,
        cfg.timeframe,
        feed_path,
        signal_path,
    )

    feed = FeedReader(feed_path, symbol, interval)
    store = SignalStore(signal_path)
    live_only = (
        args.live_only
        or args.live_since is not None
        or env_bool("LIVE_ONLY", False)
    )
    live_since_ms = parse_live_since(args.live_since) if args.live_since else None
    try:
        store.connect()
        bot = SignalBot(
            cfg,
            feed,
            store,
            symbol,
            interval,
            live_only=live_only,
            live_since_ms=live_since_ms,
        )
        bot.run(once=args.once, poll_seconds=args.poll_seconds)
    finally:
        feed.close()
        store.close()

    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        STOP_EVENT.set()
        raise SystemExit(130)
