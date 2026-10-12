#!/usr/bin/env python3
"""
Binance USDⓈ-M Futures OHLCV collector.
BTCUSDT | 1m | SQLite

- Backfill REST paginado.
- WebSocket en tiempo real.
- UPSERT idempotente.
- Protección de velas cerradas.
- Reparación periódica de huecos.
- Backoff exponencial con jitter.
- Detección de desconexiones.
- Apagado ordenado.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests
import websocket

SYMBOL = "BTCUSDT"
INTERVAL = "1m"

REST_URL = "https://fapi.binance.com/fapi/v1/klines"
WS_URL = "wss://fstream.binance.com/market/ws/btcusdt@kline_1m"

MINUTE_MS = 60_000
PAGE_LIMIT = 1500
REST_TIMEOUT = (5, 20)
MAX_BACKOFF = 60
WS_STALE_SECONDS = 90
LOOKBACK_MINUTES = 10

STOP = threading.Event()
DB_LOCK = threading.RLock()

LOG = logging.getLogger("collector")


# ---------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------

def utc_now_ms() -> int:
    return int(time.time() * 1000)


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc_date(value: str) -> int:
    dt = datetime.strptime(value, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def backoff_delay(attempt: int) -> float:
    base = min(MAX_BACKOFF, 2 ** min(attempt, 10))
    return random.uniform(base * 0.5, base)


# ---------------------------------------------------------
# SQLITE
# ---------------------------------------------------------

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
    Path(db_path).expanduser().resolve().parent.mkdir(
        parents=True, exist_ok=True
    )

    with DB_LOCK, db_session(db_path) as conn:
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
                is_closed INTEGER NOT NULL
                    CHECK(is_closed IN (0, 1)),
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(symbol, interval, open_time)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_time
            ON candles(symbol, interval, open_time)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS collector_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)


def set_state(db_path: str, key: str, value: str) -> None:
    with DB_LOCK, db_session(db_path) as conn:
        conn.execute("""
            INSERT INTO collector_state(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """, (key, value, utc_now_ms()))


def get_state(db_path: str, key: str) -> str | None:
    with DB_LOCK, db_session(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM collector_state WHERE key = ?",
            (key,)
        ).fetchone()

    return row[0] if row else None


def get_latest_open_time(db_path: str) -> int | None:
    with DB_LOCK, db_session(db_path) as conn:
        row = conn.execute("""
            SELECT MAX(open_time)
            FROM candles
            WHERE symbol = ? AND interval = ?
        """, (SYMBOL, INTERVAL)).fetchone()

    return int(row[0]) if row and row[0] is not None else None


# ---------------------------------------------------------
# VALIDACIÓN DE VELAS
# ---------------------------------------------------------

def validate_candle(
    open_time: int,
    close_time: int,
    o: float,
    h: float,
    low: float,
    close: float,
    volume: float,
    quote_volume: float,
    trades: int,
    taker_buy_volume: float,
    taker_buy_quote_volume: float,
) -> None:

    import math

    values = (
        o, h, low, close, volume, quote_volume,
        taker_buy_volume, taker_buy_quote_volume
    )

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Valores numéricos no finitos")

    if open_time < 0 or close_time < open_time:
        raise ValueError("Tiempos de vela inválidos")

    if open_time % MINUTE_MS != 0:
        raise ValueError("Vela desalineada con el intervalo de 1 minuto")

    if min(o, h, low, close) <= 0:
        raise ValueError("Precio no positivo")

    if h < max(o, low, close):
        raise ValueError("Máximo OHLC inválido")

    if low > min(o, h, close):
        raise ValueError("Mínimo OHLC inválido")

    if min(volume, quote_volume, taker_buy_volume,
           taker_buy_quote_volume) < 0:
        raise ValueError("Volumen negativo")

    if trades < 0:
        raise ValueError("Número de operaciones negativo")

    if taker_buy_volume > volume + 1e-8:
        raise ValueError("Volumen comprador superior al volumen total")


def row_from_rest(k: list) -> tuple:
    if not isinstance(k, list) or len(k) < 11:
        raise ValueError("Respuesta REST con estructura inválida")

    open_time = int(k[0])
    close_time = int(k[6])

    o, h, low, close = map(float, (k[1], k[2], k[3], k[4]))
    volume = float(k[5])
    quote_volume = float(k[7])
    trades = int(k[8])
    taker_buy_volume = float(k[9])
    taker_buy_quote_volume = float(k[10])

    validate_candle(
        open_time, close_time, o, h, low, close,
        volume, quote_volume, trades,
        taker_buy_volume, taker_buy_quote_volume
    )

    now = utc_now_ms()

    return (
        SYMBOL, INTERVAL, open_time, close_time,
        o, h, low, close, volume, quote_volume,
        trades, taker_buy_volume, taker_buy_quote_volume,
        int(close_time < now), now
    )


def row_from_ws(k: dict) -> tuple:
    open_time = int(k["t"])
    close_time = int(k["T"])

    o, h, low, close = map(
        float, (k["o"], k["h"], k["l"], k["c"])
    )

    volume = float(k["v"])
    quote_volume = float(k["q"])
    trades = int(k["n"])
    taker_buy_volume = float(k["V"])
    taker_buy_quote_volume = float(k["Q"])

    validate_candle(
        open_time, close_time, o, h, low, close,
        volume, quote_volume, trades,
        taker_buy_volume, taker_buy_quote_volume
    )

    return (
        SYMBOL, INTERVAL, open_time, close_time,
        o, h, low, close, volume, quote_volume,
        trades, taker_buy_volume, taker_buy_quote_volume,
        int(bool(k["x"])), utc_now_ms()
    )


# ---------------------------------------------------------
# PERSISTENCIA IDEMPOTENTE
# ---------------------------------------------------------

UPSERT_SQL = """
INSERT INTO candles (
    symbol, interval, open_time, close_time,
    open, high, low, close, volume, quote_volume,
    trades, taker_buy_volume, taker_buy_quote_volume,
    is_closed, updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
    close_time = excluded.close_time,
    open = excluded.open,
    high = excluded.high,
    low = excluded.low,
    close = excluded.close,
    volume = excluded.volume,
    quote_volume = excluded.quote_volume,
    trades = excluded.trades,
    taker_buy_volume = excluded.taker_buy_volume,
    taker_buy_quote_volume = excluded.taker_buy_quote_volume,
    is_closed = MAX(candles.is_closed, excluded.is_closed),
    updated_at = excluded.updated_at

WHERE candles.is_closed = 0 OR excluded.is_closed = 1
"""


def save_rows(db_path: str, rows: list[tuple]) -> int:
    if not rows:
        return 0

    with DB_LOCK, db_session(db_path) as conn:
        conn.executemany(UPSERT_SQL, rows)

    return len(rows)


# ---------------------------------------------------------
# REST API
# ---------------------------------------------------------

class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


def fetch_klines(
    session: requests.Session,
    start_ms: int,
    end_ms: int | None = None
) -> list:

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ms,
        "limit": PAGE_LIMIT,
    }

    if end_ms is not None:
        params["endTime"] = end_ms

    response = session.get(
        REST_URL, params=params, timeout=REST_TIMEOUT
    )

    if response.status_code in (418, 429):
        retry = response.headers.get("Retry-After", "5")

        try:
            retry_seconds = float(retry)
        except ValueError:
            retry_seconds = 5.0

        raise RateLimitError(
            f"HTTP {response.status_code}",
            max(1.0, retry_seconds)
        )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError(f"Respuesta REST inesperada: {data!r}")

    return data


def backfill(
    db_path: str,
    session: requests.Session,
    start_ms: int,
    end_ms: int | None = None,
    label: str = "backfill",
    max_retries: int = 5
) -> int:

    cursor = max(0, start_ms)
    end_bound = end_ms if end_ms is not None else utc_now_ms()

    total = 0
    attempt = 0

    while not STOP.is_set() and cursor <= end_bound:

        try:
            candles = fetch_klines(session, cursor, end_bound)
            attempt = 0

        except RateLimitError as exc:
            attempt += 1
            delay = max(exc.retry_after, backoff_delay(attempt))

            LOG.warning(
                "%s: límite REST; reintento en %.1f s",
                label, delay
            )

            if attempt >= max_retries:
                return total

            if STOP.wait(delay):
                break

            continue

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            attempt += 1
            delay = backoff_delay(attempt)

            LOG.warning(
                "%s: error REST; reintento en %.1f s: %s",
                label, delay, exc
            )

            if attempt >= max_retries:
                LOG.error(
                    "%s: se pospone la recuperación desde %s",
                    label, cursor
                )
                return total

            if STOP.wait(delay):
                break

            continue

        if not candles:
            break

        try:
            opens = [int(k[0]) for k in candles]

            if opens != sorted(set(opens)):
                raise ValueError(
                    "Velas duplicadas o desordenadas en respuesta REST"
                )

            rows = [row_from_rest(k) for k in candles]

        except (ValueError, TypeError, IndexError, KeyError) as exc:
            LOG.error("%s: lote inválido: %s", label, exc)
            return total

        total += save_rows(db_path, rows)

        next_cursor = opens[-1] + MINUTE_MS

        if next_cursor <= cursor:
            LOG.error("%s: cursor REST no avanza", label)
            break

        cursor = next_cursor

        if len(candles) < PAGE_LIMIT:
            break

        if STOP.wait(0.1):
            break

    if total:
        LOG.info("%s: procesadas %s velas", label, total)

    return total


# ---------------------------------------------------------
# DETECCIÓN Y REPARACIÓN DE HUECOS
# ---------------------------------------------------------

def find_recent_gaps(
    db_path: str,
    scan_minutes: int = 1440
) -> list[tuple[int, int]]:

    latest = get_latest_open_time(db_path)

    if latest is None:
        return []

    lower = max(0, latest - scan_minutes * MINUTE_MS)

    with DB_LOCK, db_session(db_path) as conn:
        times = [
            int(row[0])
            for row in conn.execute("""
                SELECT open_time
                FROM candles
                WHERE symbol = ?
                  AND interval = ?
                  AND open_time BETWEEN ? AND ?
                ORDER BY open_time
            """, (SYMBOL, INTERVAL, lower, latest))
        ]

    gaps = []

    for previous, current in zip(times, times[1:]):
        if current - previous > MINUTE_MS:
            gaps.append((
                previous + MINUTE_MS,
                current - MINUTE_MS
            ))

    return gaps


def repair_recent(
    db_path: str,
    session: requests.Session,
    lookback_minutes: int = LOOKBACK_MINUTES,
    gap_scan_minutes: int = 1440
) -> None:

    latest = get_latest_open_time(db_path)

    if latest is None:
        start = utc_now_ms() - lookback_minutes * MINUTE_MS
    else:
        start = max(
            0, latest - lookback_minutes * MINUTE_MS
        )

    backfill(
        db_path, session, start,
        label="reparación reciente"
    )

    for gap_start, gap_end in find_recent_gaps(
        db_path, gap_scan_minutes
    ):
        if STOP.is_set():
            break

        LOG.warning(
            "Hueco detectado: %s - %s",
            gap_start, gap_end
        )

        backfill(
            db_path, session,
            gap_start, end_ms=gap_end,
            label="reparación de hueco"
        )


def rest_repair_worker(
    db_path: str,
    every_seconds: int,
    gap_scan_minutes: int
) -> None:

    session = requests.Session()
    session.headers.update({
        "User-Agent": "btc-um-ohlcv-collector/2.0"
    })

    try:
        while not STOP.wait(every_seconds):
            try:
                repair_recent(
                    db_path, session,
                    gap_scan_minutes=gap_scan_minutes
                )

                set_state(
                    db_path,
                    "last_rest_repair_utc",
                    utc_iso_now()
                )

            except Exception:
                LOG.exception("Fallo en reparación REST periódica")

    finally:
        session.close()


# ---------------------------------------------------------
# WEBSOCKET
# ---------------------------------------------------------

def run_websocket(db_path: str) -> None:

    backoff_attempt = 0

    while not STOP.is_set():

        opened_at = None

        def on_open(ws):
            nonlocal opened_at

            opened_at = time.monotonic()

            LOG.info("WebSocket conectado: %s", WS_URL)

            try:
                set_state(db_path, "websocket_status", "connected")
            except Exception:
                LOG.exception("No se pudo guardar estado WebSocket")

        def on_message(ws, message):
            try:
                payload = json.loads(message)
                k = payload.get("k")

                if not isinstance(k, dict):
                    return

                row = row_from_ws(k)
                save_rows(db_path, [row])

                set_state(
                    db_path,
                    "last_ws_message_utc",
                    utc_iso_now()
                )

                if bool(k["x"]):
                    candle_time = int(k["t"])

                    LOG.info(
                        "Vela cerrada %s close=%s volume=%s",
                        datetime.fromtimestamp(
                            candle_time / 1000,
                            timezone.utc
                        ).isoformat(),
                        k["c"], k["v"]
                    )

                    set_state(
                        db_path,
                        "last_closed_candle_open_time",
                        str(candle_time)
                    )

            except Exception:
                LOG.exception("Error procesando mensaje WebSocket")

        def on_error(ws, error):
            LOG.warning("Error WebSocket: %s", error)

            try:
                set_state(db_path, "websocket_status", "error")
            except Exception:
                LOG.exception("No se pudo guardar error WebSocket")

        def on_close(ws, status_code, message):
            LOG.warning(
                "WebSocket cerrado: código=%s mensaje=%s",
                status_code, message
            )

            try:
                set_state(
                    db_path, "websocket_status", "disconnected"
                )
            except Exception:
                LOG.exception("No se pudo guardar cierre WebSocket")

        app = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        try:
            app.run_forever(
                ping_interval=20,
                ping_timeout=10,
                reconnect=0,
                skip_utf8_validation=True
            )

        except Exception:
            LOG.exception("Fallo en bucle WebSocket")

        if STOP.is_set():
            break

        # Reiniciar el backoff tras una conexión estable.
        if opened_at is not None:
            duration = time.monotonic() - opened_at
            if duration >= 120:
                backoff_attempt = 0

        backoff_attempt += 1
        delay = backoff_delay(backoff_attempt)

        LOG.info("Reconexión WebSocket en %.1f s", delay)

        if STOP.wait(delay):
            break

    try:
        set_state(db_path, "websocket_status", "stopped")
    except Exception:
        LOG.exception("No se pudo guardar estado final WebSocket")


# ---------------------------------------------------------
# SUPERVISOR DE ACTIVIDAD
# ---------------------------------------------------------

def websocket_watchdog(
    db_path: str,
    stale_seconds: int = WS_STALE_SECONDS
) -> None:

    while not STOP.wait(10):
        try:
            last_message = get_state(
                db_path, "last_ws_message_utc"
            )

            if not last_message:
                continue

            timestamp = datetime.fromisoformat(last_message)

            age = (
                datetime.now(timezone.utc) - timestamp
            ).total_seconds()

            if age > stale_seconds:
                LOG.error(
                    "Feed WebSocket sin actividad durante %.1f s",
                    age
                )

                set_state(
                    db_path, "websocket_status", "stale"
                )

        except Exception:
            LOG.exception("Error en watchdog WebSocket")


# ---------------------------------------------------------
# APAGADO Y MAIN
# ---------------------------------------------------------

def handle_signal(signum, frame):
    LOG.info("Señal %s recibida; iniciando apagado", signum)
    STOP.set()


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Binance BTCUSDT 1m OHLCV -> SQLite"
    )

    parser.add_argument(
        "--db",
        default=os.getenv("OHLCV_DB", "feed.db")
    )

    parser.add_argument(
        "--start-date",
        default=os.getenv("START_DATE", "2026-08-01")
    )

    parser.add_argument(
        "--repair-seconds",
        type=int,
        default=int(os.getenv("REPAIR_SECONDS", "60"))
    )

    parser.add_argument(
        "--gap-scan-minutes",
        type=int,
        default=int(os.getenv("GAP_SCAN_MINUTES", "1440"))
    )

    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s"
    )

    logging.Formatter.converter = time.gmtime

    try:
        initial_start = parse_utc_date(args.start_date)
    except ValueError:
        parser.error("--start-date debe ser YYYY-MM-DD")

    if args.repair_seconds < 10:
        parser.error("--repair-seconds debe ser >= 10")

    if args.gap_scan_minutes < 1:
        parser.error("--gap-scan-minutes debe ser >= 1")

    # Registrar señales antes de iniciar las tareas.
    signal.signal(signal.SIGINT, handle_signal)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "btc-um-ohlcv-collector/2.0"
    })

    workers = []

    try:
        init_db(args.db)

        set_state(args.db, "collector_status", "starting")

        LOG.info(
            "Base de datos: %s",
            Path(args.db).expanduser().resolve()
        )

        latest = get_latest_open_time(args.db)

        if latest is None:
            start = initial_start
        else:
            start = max(
                initial_start,
                latest - LOOKBACK_MINUTES * MINUTE_MS
            )

        # REST no bloquea el arranque del WebSocket.
        def initial_backfill_worker():
            try:
                backfill(
                    args.db, session, start,
                    label="backfill inicial"
                )

                repair_recent(
                    args.db, session,
                    gap_scan_minutes=args.gap_scan_minutes
                )

                set_state(
                    args.db, "initial_backfill_status", "completed"
                )

            except Exception:
                LOG.exception("Falló backfill inicial")

                try:
                    set_state(
                        args.db, "initial_backfill_status", "error"
                    )
                except Exception:
                    LOG.exception("No se pudo guardar estado del backfill")

        initial_worker = threading.Thread(
            target=initial_backfill_worker,
            name="initial-backfill",
            daemon=True
        )

        repair_worker = threading.Thread(
            target=rest_repair_worker,
            args=(
                args.db,
                args.repair_seconds,
                args.gap_scan_minutes
            ),
            name="rest-repair",
            daemon=True
        )

        watchdog_worker = threading.Thread(
            target=websocket_watchdog,
            args=(args.db, WS_STALE_SECONDS),
            name="websocket-watchdog",
            daemon=True
        )

        for worker in (
            initial_worker,
            repair_worker,
            watchdog_worker
        ):
            worker.start()
            workers.append(worker)

        set_state(args.db, "collector_status", "running")

        run_websocket(args.db)

    except Exception:
        LOG.exception("Error fatal del collector")
        return 1

    finally:
        STOP.set()
        session.close()

        for worker in workers:
            worker.join(timeout=25)

            if worker.is_alive():
                LOG.error(
                    "El worker %s no terminó dentro del timeout",
                    worker.name
                )

        try:
            set_state(args.db, "collector_status", "stopped")
        except Exception:
            LOG.exception("No se pudo guardar el estado final")

        LOG.info("Collector detenido")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())