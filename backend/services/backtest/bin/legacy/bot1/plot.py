#!/usr/bin/env python3
"""
Grafica el JSON del endpoint `/series` de bot.py (ver server.py).

Genera una página HTML interactiva (Plotly) con un gráfico por timeframe:

- timeframe de la estrategia: velas + EMA rápida/lenta, marcadores de cruce
  (`cross_up` / `cross_down`) y de salida (`adx_exit` / `weak_exit`), más
  paneles de `adx_entry`, volumen y `target`.
- resto de timeframes: velas + volumen.

Uso
---
    python plot.py                               # lee http://127.0.0.1:8080/series
    python plot.py series.json                   # lee un archivo guardado
    python plot.py http://bot:8080/series --last 300 --tz America/Bogota
    python plot.py --timeframe 30min --timeframe 1m
    python plot.py --watch 30 --open             # refresca cada 30 s

Requiere `pandas` y `plotly` (`pip install plotly`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_SOURCE = "http://127.0.0.1:8080/series"

UP, DOWN = "#26a69a", "#ef5350"
FAST, SLOW, NEUTRAL = "#ff9800", "#2962ff", "#7e57c2"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_payload(source: str, limit: int | None = None) -> dict:
    """Lee el JSON desde una URL o un archivo."""
    if source.startswith(("http://", "https://")):
        if limit is not None and "limit=" not in source:
            source += ("&" if "?" in source else "?") + f"limit={limit}"
        with urllib.request.urlopen(source, timeout=15) as response:
            return json.load(response)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def to_index(t: list[int], tz: str) -> pd.DatetimeIndex:
    """`t` (epoch ms) -> DatetimeIndex en la zona horaria pedida."""
    values = pd.Series(t, dtype="int64")
    # Payloads antiguos (antes del fix en server.py) traían t = ms // 10**6.
    if len(values) and values.max() < 10**8:
        values = values * 10**6
    return pd.DatetimeIndex(pd.to_datetime(values, unit="ms", utc=True)).tz_convert(tz)


def to_frame(entry: dict, tz: str, last: int | None) -> pd.DataFrame:
    frame = pd.DataFrame(entry["series"], index=to_index(entry["t"], tz))
    return frame.tail(last) if last else frame


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    """Serie booleana (False si la columna no existe o tiene nulos)."""
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _candles(frame: pd.DataFrame) -> go.Candlestick:
    return go.Candlestick(
        x=frame.index, open=frame["open"], high=frame["high"],
        low=frame["low"], close=frame["close"], name="precio",
        increasing_line_color=UP, decreasing_line_color=DOWN,
    )


def _volume(frame: pd.DataFrame) -> go.Bar:
    colors = [UP if c >= o else DOWN for o, c in zip(frame["open"], frame["close"])]
    return go.Bar(x=frame.index, y=frame["volume"], marker_color=colors,
                  name="volumen", showlegend=False)


def _markers(fig, frame, column, y, name, symbol, color, row) -> None:
    mask = _flag(frame, column)
    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=frame.index[mask], y=y[mask], mode="markers", name=name,
                marker=dict(symbol=symbol, size=10, color=color,
                            line=dict(width=1, color="white")),
            ),
            row=row, col=1,
        )


def build_figure(entry: dict, payload: dict, tz: str, last: int | None) -> go.Figure | None:
    frame = to_frame(entry, tz, last)
    if frame.empty:
        return None

    is_strategy = "strategy" in entry["roles"]
    panels = ["price"]
    if is_strategy and "adx_entry" in frame:
        panels.append("adx")
    if "volume" in frame:
        panels.append("volume")
    if is_strategy and "target" in frame:
        panels.append("target")
    row = {name: i + 1 for i, name in enumerate(panels)}
    heights = {"price": 0.55, "adx": 0.15, "volume": 0.15, "target": 0.15}

    fig = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[heights[p] for p in panels],
    )
    fig.add_trace(_candles(frame), row=row["price"], col=1)

    if is_strategy:
        for column, color in (("ema_fast", FAST), ("ema_slow", SLOW)):
            if column in frame:
                fig.add_trace(
                    go.Scatter(x=frame.index, y=frame[column], name=column,
                               line=dict(color=color, width=1.5)),
                    row=row["price"], col=1,
                )
        # Marcadores a una distancia proporcional al rango visible.
        pad = (frame["high"].max() - frame["low"].min()) * 0.02
        low, high = frame["low"] - pad, frame["high"] + pad
        p = row["price"]
        _markers(fig, frame, "cross_up", low, "cross_up", "triangle-up", UP, p)
        _markers(fig, frame, "cross_down", high, "cross_down", "triangle-down", DOWN, p)
        _markers(fig, frame, "adx_exit", high + pad, "adx_exit", "x", "#ffeb3b", p)
        _markers(fig, frame, "weak_exit", high + 2 * pad, "weak_exit", "diamond", "#e040fb", p)

    if "adx" in row:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["adx_entry"], name="adx_entry",
                       line=dict(color=NEUTRAL, width=1.5)),
            row=row["adx"], col=1,
        )
        fig.update_yaxes(title_text="ADX", row=row["adx"], col=1)
    if "volume" in row:
        fig.add_trace(_volume(frame), row=row["volume"], col=1)
        fig.update_yaxes(title_text="vol", row=row["volume"], col=1)
    if "target" in row:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["target"], name="target",
                       line=dict(color=FAST, width=1.5, shape="hv")),
            row=row["target"], col=1,
        )
        fig.update_yaxes(title_text="target", tickvals=[-1, 0, 1], row=row["target"], col=1)

    roles = ", ".join(entry["roles"])
    fig.update_layout(
        title=f"{payload.get('symbol', '')} · {entry['timeframe']} ({roles}) · "
              f"{len(frame)} velas",
        template="plotly_dark", height=250 + 170 * len(panels),
        margin=dict(l=60, r=30, t=60, b=30), hovermode="x unified",
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ---------------------------------------------------------------------------
# Página HTML
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">{refresh}
<title>{title}</title>
<style>
  body {{ background:#111; color:#ddd; font:14px system-ui,sans-serif; margin:0 auto; max-width:1500px; padding:16px; }}
  h1 {{ font-size:18px; margin:0 0 4px; }} p {{ margin:0 0 16px; color:#999; }}
  .chart {{ margin-bottom:24px; }}
</style></head><body>
<h1>{title}</h1><p>{subtitle}</p>
{charts}
</body></html>
"""


def build_page(payload: dict, args: argparse.Namespace) -> tuple[str, int]:
    entries = payload.get("timeframes", [])
    if args.timeframe:
        entries = [e for e in entries if e["timeframe"] in args.timeframe]

    figures = [fig for e in entries if (fig := build_figure(e, payload, args.tz, args.last))]
    plotly_js = True if args.offline else "cdn"  # solo se incluye en el primer gráfico
    charts = "\n".join(
        '<div class="chart">'
        + fig.to_html(full_html=False, include_plotlyjs=plotly_js if i == 0 else False)
        + "</div>"
        for i, fig in enumerate(figures)
    )
    subtitle = (
        f"intervalo {payload.get('interval', '?')} · estrategia "
        f"{payload.get('strategy_timeframe', '?')} · calculado {payload.get('updated_at', '?')} "
        f"· zona horaria {args.tz}"
    )
    refresh = f'<meta http-equiv="refresh" content="{args.watch:g}">' if args.watch else ""
    html = PAGE.format(
        refresh=refresh, title=f"{payload.get('symbol', 'series')} — series",
        subtitle=subtitle, charts=charts,
    )
    return html, len(figures)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grafica el JSON de /series del bot.")
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE,
                        help=f"URL o archivo JSON (por defecto {DEFAULT_SOURCE}).")
    parser.add_argument("-o", "--out", default="series.html", help="HTML de salida.")
    parser.add_argument("-t", "--timeframe", action="append",
                        help="Solo este timeframe (repetible), p. ej. -t 30min -t 1m.")
    parser.add_argument("--last", type=int, default=None, help="Graficar solo las últimas N velas.")
    parser.add_argument("--tz", default="UTC", help="Zona horaria (p. ej. America/Bogota).")
    parser.add_argument("--offline", action="store_true",
                        help="Incrusta plotly.js en el HTML (por defecto se carga desde CDN).")
    parser.add_argument("--watch", type=float, default=0, metavar="SEGUNDOS",
                        help="Vuelve a leer y regenerar cada N segundos (la página se autorrecarga).")
    parser.add_argument("--open", action="store_true", help="Abre el HTML en el navegador.")
    return parser.parse_args()


def render(args: argparse.Namespace) -> int:
    payload = load_payload(args.source, args.last)
    if not payload.get("ready", True) or not payload.get("timeframes"):
        print("El bot todavía no tiene series calculadas.", file=sys.stderr)
        return 0
    html, count = build_page(payload, args)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{count} gráfico(s) -> {args.out}  (calculado {payload.get('updated_at')})")
    return count


def main() -> int:
    args = parse_args()
    opened = False
    try:
        while True:
            try:
                count = render(args)
            except Exception as exc:  # en --watch seguimos intentando
                if not args.watch:
                    raise
                print(f"Error: {exc}", file=sys.stderr)
                count = 0
            if args.open and count and not opened:
                webbrowser.open(Path(args.out).resolve().as_uri())
                opened = True
            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())