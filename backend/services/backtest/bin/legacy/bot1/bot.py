#!/usr/bin/env python3
"""
Bot 1 — Señales EMA Cross (eventos en signal.db)
================================================

Lee la data OHLCV almacenada por `feed.py` en `feed.db` (1m por defecto),
aplica la estrategia definida en `strategy/strategy.py` y guarda cada señal
como un evento en `signal.db`.

Reglas de diseño
----------------
- El bot **no crea órdenes**. Solo publica eventos con `executed = 0`
  (false); un consumidor externo decide si los ejecuta y actualiza la marca.
- Cada evento es idempotente: se deduplica por
  (symbol, interval, timeframe, event, signal, reason, bar_time).
- La señal se conoce al cierre de la vela de la estrategia y es accionable
  en la apertura siguiente; por eso se guardan `bar_time` y `signal_time`.

Uso
---
    python bot.py                 # bucle continuo (polling de feed.db)
    python bot.py --once          # procesa lo disponible y termina
    python bot.py --help

La configuración se toma de `.env` (mismas variables que `feed.py` más las
de la estrategia con prefijo `STRATEGY_`).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import threading
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


def build_strategy_config() -> Config:
    """Config de la estrategia. Los defaults coinciden con `strategy.Config`."""
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
        timestamp_is_close=defaults.timestamp_is_close,
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


def build_events(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    adx_exit: np.ndarray | None = None,
    weak_exit: np.ndarray | None = None,
    cfg: Config = None,
    symbol: str = "",
    interval: str = "",
) -> list[dict]:
    """Construye eventos de entrada y condiciones de salida."""
    if bars.empty:
        return []
    if cfg is None:
        raise ValueError("cfg requerido")

    timeframe_ms = int(pd.Timedelta(cfg.timeframe).total_seconds() * 1000)
    target = signals["target"].to_numpy() if signals is not None else np.zeros(len(bars))
    fast = signals["ema_fast"].to_numpy(float) if signals is not None else np.full(len(bars), np.nan)
    slow = signals["ema_slow"].to_numpy(float) if signals is not None else np.full(len(bars), np.nan)
    entry_adx = signals["adx_entry"].to_numpy(float) if signals is not None else np.full(len(bars), np.nan)
    closes = bars["close"].to_numpy(float)
    index = bars.index
    created_at = utc_iso()

    adx_arr = np.asarray(adx_exit, dtype=bool) if adx_exit is not None else np.zeros(len(bars), dtype=bool)
    weak_arr = np.asarray(weak_exit, dtype=bool) if weak_exit is not None else np.zeros(len(bars), dtype=bool)
    if len(adx_arr) != len(bars):
        adx_arr = np.zeros(len(bars), dtype=bool)
    if len(weak_arr) != len(bars):
        weak_arr = np.zeros(len(bars), dtype=bool)

    events: list[dict] = []
    for i in np.flatnonzero(target != 0):
        direction = int(target[i])
        if direction == 1 and not cfg.allow_long:
            continue
        if direction == -1 and not cfg.allow_short:
            continue
        signal_side = "LONG" if direction == 1 else "SHORT"
        bar_time = int(index[i].timestamp() * 1000)
        signal_time = bar_time if cfg.timestamp_is_close else bar_time + timeframe_ms
        events.append(
            {
                "symbol": symbol,
                "interval": interval,
                "timeframe": cfg.timeframe,
                "bar_time": bar_time,
                "signal_time": signal_time,
                "event": "ENTRY",
                "signal": signal_side,
                "side": signal_side,
                "reason": "EMA_CROSS",
                "price": opt_float(closes[i]),
                "ema_fast": opt_float(fast[i]),
                "ema_slow": opt_float(slow[i]),
                "adx_entry": opt_float(entry_adx[i]),
                "executed": 0,
                "created_at": created_at,
                "payload": json.dumps({"target": direction}),
            }
        )

    for i in np.flatnonzero(adx_arr):
        bar_time = int(index[i].timestamp() * 1000)
        signal_time = bar_time if cfg.timestamp_is_close else bar_time + timeframe_ms
        adx_tf_min = int(pd.Timedelta(cfg.adx_timeframe).total_seconds() / 60)
        reason = f"ADX_{adx_tf_min}M_REVERSAL" if cfg.adx_exit_pct >= 100 else f"ADX_{adx_tf_min}M_REVERSAL_{cfg.adx_exit_pct:g}%"
        events.append(
            {
                "symbol": symbol,
                "interval": interval,
                "timeframe": cfg.timeframe,
                "bar_time": bar_time,
                "signal_time": signal_time,
                "event": "EXIT_CONDITION",
                "signal": "ADX_REVERSAL",
                "side": None,
                "reason": reason,
                "price": opt_float(closes[i]),
                "ema_fast": opt_float(fast[i]),
                "ema_slow": opt_float(slow[i]),
                "adx_entry": opt_float(entry_adx[i]),
                "executed": 0,
                "created_at": created_at,
                "payload": json.dumps({"condition": "adx_reversal", "exit_pct": cfg.adx_exit_pct}),
            }
        )

    for i in np.flatnonzero(weak_arr):
        bar_time = int(index[i].timestamp() * 1000)
        signal_time = bar_time if cfg.timestamp_is_close else bar_time + timeframe_ms
        weak_tf = cfg.ema_weakening_timeframe or cfg.timeframe
        reason = f"EMA_WEAKENING_{weak_tf.upper()}" if cfg.ema_weakening_exit_pct >= 100 else f"EMA_WEAKENING_{weak_tf.upper()}_{cfg.ema_weakening_exit_pct:g}%"
        events.append(
            {
                "symbol": symbol,
                "interval": interval,
                "timeframe": cfg.timeframe,
                "bar_time": bar_time,
                "signal_time": signal_time,
                "event": "EXIT_CONDITION",
                "signal": "EMA_WEAKENING",
                "side": None,
                "reason": reason,
                "price": opt_float(closes[i]),
                "ema_fast": opt_float(fast[i]),
                "ema_slow": opt_float(slow[i]),
                "adx_entry": opt_float(entry_adx[i]),
                "executed": 0,
                "created_at": created_at,
                "payload": json.dumps({"condition": "ema_weakening", "exit_pct": cfg.ema_weakening_exit_pct}),
            }
        )

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
    ):
        self.cfg = cfg
        self.feed = feed
        self.store = store
        self.symbol = symbol
        self.interval = interval
        self.strategy = EMACrossStrategy(cfg)
        self.min_bars = min_bars or (cfg.ema_slow + 2)

    def process_once(self) -> int:
        """Ejecuta un ciclo: lee, calcula y publica eventos nuevos."""
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
        inserted = self.store.insert(events)
        if inserted:
            logger.info(
                "Publicados %d eventos nuevos (de %d señales).", inserted, len(events)
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

            if once or STOP_EVENT.wait(poll_seconds):
                break


# ===========================================================================
# CLI / arranque
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bot de señales EMA Cross -> signal.db (solo eventos)."
    )
    parser.add_argument(
        "--feed-db",
        default=os.getenv("DB_PATH", "feed.db"),
        help="Ruta a feed.db (por defecto DB_PATH o feed.db).",
    )
    parser.add_argument(
        "--signal-db",
        default=os.getenv("SIGNAL_DB", "signal.db"),
        help="Ruta a signal.db (por defecto SIGNAL_DB o signal.db).",
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
    try:
        store.connect()
        bot = SignalBot(cfg, feed, store, symbol, interval)
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
