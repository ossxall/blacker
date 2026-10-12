#!/usr/bin/env python3
"""
Resilient Binance USDⓈ-M Futures OHLCV collector.
Symbol: BTCUSDT | Interval: 1m | Storage: SQLite

- REST backfill with pagination (default: from 2019-09-08 UTC).
- WebSocket live updates, including the currently forming candle.
- Idempotent SQLite UPSERT keyed by candle open time.
- Reconnects with exponential backoff and periodic REST gap repair.
- No API key required: public market data only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import logging
import os
import signal
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
import websocket

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
REST_URL = "https://fapi.binance.com/fapi/v1/klines"
WS_URL = "wss://fstream.binance.com/market/ws/btcusdt@kline_1m"
PAGE_LIMIT = 1500
MINUTE_MS = 60_000

STOP = threading.Event()
DB_LOCK = threading.Lock()

LOG = logging.getLogger("binance_ohlcv")


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def parse_utc_date(value: str) -> int:
    """Parse YYYY-MM-DD as UTC epoch milliseconds."""
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def db_session(db_path: str):
    conn = connect_db(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with db_session(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trades INTEGER NOT NULL,
                taker_buy_volume REAL NOT NULL,
                taker_buy_quote_volume REAL NOT NULL,
                is_closed INTEGER NOT NULL CHECK (is_closed IN (0, 1)),
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_closed_time
            ON candles(symbol, interval, is_closed, open_time)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS collector_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)
        conn.commit()


def row_from_rest(k: list) -> tuple:
    now = utc_now_ms()
    return (
        SYMBOL, INTERVAL, int(k[0]), int(k[6]),
        float(k[1]), float(k[2]), float(k[3]), float(k[4]),
        float(k[5]), float(k[7]), int(k[8]), float(k[9]),
        float(k[10]), int(int(k[6]) < now), now
    )


def row_from_ws(k: dict) -> tuple:
    return (
        SYMBOL, INTERVAL, int(k["t"]), int(k["T"]),
        float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]),
        float(k["v"]), float(k["q"]), int(k["n"]), float(k["V"]),
        float(k["Q"]), int(bool(k["x"])), utc_now_ms()
    )


UPSERT_SQL = """
INSERT INTO candles (
    symbol, interval, open_time, close_time, open, high, low, close,
    volume, quote_volume, trades, taker_buy_volume,
    taker_buy_quote_volume, is_closed, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
    close_time=excluded.close_time,
    open=excluded.open, high=excluded.high, low=excluded.low,
    close=excluded.close, volume=excluded.volume,
    quote_volume=excluded.quote_volume, trades=excluded.trades,
    taker_buy_volume=excluded.taker_buy_volume,
    taker_buy_quote_volume=excluded.taker_buy_quote_volume,
    is_closed=MAX(candles.is_closed, excluded.is_closed),
    updated_at=excluded.updated_at
"""


def save_rows(db_path: str, rows: Iterable[tuple]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with DB_LOCK:
        with db_session(db_path) as conn:
            conn.executemany(UPSERT_SQL, rows)
            conn.commit()
    return len(rows)


def get_latest_open_time(db_path: str) -> int | None:
    with db_session(db_path) as conn:
        result = conn.execute(
            "SELECT MAX(open_time) FROM candles WHERE symbol=? AND interval=?",
            (SYMBOL, INTERVAL)
        ).fetchone()[0]
    return int(result) if result is not None else None


def fetch_klines(session: requests.Session, start_ms: int, end_ms: int | None = None) -> list:
    params = {
        "symbol": SYMBOL, "interval": INTERVAL,
        "startTime": start_ms, "limit": PAGE_LIMIT
    }
    if end_ms is not None:
        params["endTime"] = end_ms
    response = session.get(REST_URL, params=params, timeout=(5, 30))
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Respuesta REST inesperada: {data!r}")
    return data


def backfill(db_path: str, session: requests.Session, start_ms: int,
             end_ms: int | None = None, label: str = "backfill") -> int:
    """Fetch candles forward in pages; safe to repeat because writes are UPSERTs."""
    cursor = start_ms
    total = 0
    end_bound = end_ms if end_ms is not None else utc_now_ms()
    while not STOP.is_set() and cursor <= end_bound:
        try:
            candles = fetch_klines(session, cursor, end_bound)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            LOG.warning("%s REST error at %s: %s", label, cursor, exc)
            if STOP.wait(3):
                break
            continue

        if not candles:
            break
        rows = [row_from_rest(k) for k in candles]
        total += save_rows(db_path, rows)
        next_cursor = int(candles[-1][0]) + MINUTE_MS
        if next_cursor <= cursor:
            LOG.error("REST pagination did not advance; stopping this backfill.")
            break
        cursor = next_cursor
        if len(candles) < PAGE_LIMIT:
            break
        # Be polite to the public REST API.
        STOP.wait(0.08)

    if total:
        LOG.info("%s: processed %s candles (cursor=%s)", label, total, cursor)
    return total


def repair_recent(db_path: str, session: requests.Session, lookback_minutes: int = 10) -> None:
    """Re-fetch a short overlap to repair disconnects and refresh the current candle."""
    latest = get_latest_open_time(db_path)
    start = max(0, (latest - lookback_minutes * MINUTE_MS) if latest is not None
                else utc_now_ms() - lookback_minutes * MINUTE_MS)
    backfill(db_path, session, start, label="repair")


def set_state(db_path: str, key: str, value: str) -> None:
    with db_session(db_path) as conn:
        conn.execute("""
            INSERT INTO collector_state(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              value=excluded.value, updated_at=excluded.updated_at
        """, (key, value, utc_now_ms()))
        conn.commit()


def repair_worker(db_path: str, stop: threading.Event, every_seconds: int) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-um-ohlcv-collector/1.0"})
    while not stop.wait(every_seconds):
        try:
            repair_recent(db_path, session, lookback_minutes=10)
            set_state(db_path, "last_rest_repair_utc",
                      datetime.now(timezone.utc).isoformat())
        except Exception:
            LOG.exception("Error inesperado en reparación REST periódica.")
    session.close()


def run_websocket(db_path: str) -> None:
    def on_message(ws, message: str) -> None:
        try:
            payload = __import__("json").loads(message)
            k = payload.get("k")
            if not k:
                return
            save_rows(db_path, [row_from_ws(k)])
            if k.get("x"):
                LOG.info("Candle closed %s close=%s volume=%s",
                         datetime.fromtimestamp(int(k["t"]) / 1000,
                                                timezone.utc).isoformat(),
                         k["c"], k["v"])
                set_state(db_path, "last_closed_candle_open_time", str(k["t"]))
        except Exception:
            LOG.exception("No se pudo procesar mensaje WebSocket.")

    def on_open(ws) -> None:
        LOG.info("WebSocket conectado: %s", WS_URL)
        set_state(db_path, "websocket_status", "connected")

    def on_error(ws, error) -> None:
        LOG.warning("WebSocket error: %s", error)
        set_state(db_path, "websocket_status", "error")

    def on_close(ws, status_code, message) -> None:
        LOG.warning("WebSocket cerrado: code=%s message=%s", status_code, message)
        set_state(db_path, "websocket_status", "disconnected")

    backoff = 1
    while not STOP.is_set():
        app = websocket.WebSocketApp(
            WS_URL, on_open=on_open, on_message=on_message,
            on_error=on_error, on_close=on_close
        )
        try:
            app.run_forever(ping_interval=20, ping_timeout=10,
                            reconnect=0, skip_utf8_validation=True)
        except Exception:
            LOG.exception("Fallo del bucle WebSocket.")
        if STOP.is_set():
            break
        LOG.info("Reconectando en %s s...", backoff)
        if STOP.wait(backoff):
            break
        backoff = min(backoff * 2, 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Binance UM BTCUSDT 1m OHLCV -> SQLite")
    parser.add_argument("--db", default=os.getenv("OHLCV_DB", "feed.db"),
                        help="Ruta SQLite (default: feed.db)")
    parser.add_argument("--start-date", default=os.getenv("START_DATE", "2026-08-01"),
                        help="Fecha UTC inicial YYYY-MM-DD para backfill")
    parser.add_argument("--repair-seconds", type=int,
                        default=int(os.getenv("REPAIR_SECONDS", "60")),
                        help="Intervalo de reparación REST (default: 60)")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    # Log timestamps are UTC.
    logging.Formatter.converter = time.gmtime

    try:
        initial_start = parse_utc_date(args.start_date)
    except ValueError:
        parser.error("--start-date debe tener formato YYYY-MM-DD")
    if args.repair_seconds < 10:
        parser.error("--repair-seconds debe ser >= 10")

    init_db(args.db)
    LOG.info("DB: %s", Path(args.db).resolve())
    LOG.info("Iniciando backfill desde %s UTC", args.start_date)

    session = requests.Session()
    session.headers.update({"User-Agent": "btc-um-ohlcv-collector/1.0"})
    try:
        latest = get_latest_open_time(args.db)
        start = initial_start if latest is None else max(initial_start, latest - 10 * MINUTE_MS)
        backfill(args.db, session, start, label="initial backfill")
        repair_recent(args.db, session, lookback_minutes=10)
    except Exception:
        LOG.exception("Backfill inicial falló; se seguirá intentando con el servicio activo.")

    worker = threading.Thread(
        target=repair_worker, args=(args.db, STOP, args.repair_seconds),
        name="rest-gap-repair", daemon=True
    )
    worker.start()

    def request_stop(signum, frame) -> None:
        LOG.info("Señal %s recibida; cerrando...", signum)
        STOP.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        run_websocket(args.db)
    finally:
        STOP.set()
        session.close()
        worker.join(timeout=5)
        set_state(args.db, "collector_status", "stopped")
        LOG.info("Collector detenido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())