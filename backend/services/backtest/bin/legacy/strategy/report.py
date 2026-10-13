from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.config import Config
from strategy.models import DEFAULT_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Métricas, reportes y archivos
# ---------------------------------------------------------------------------

def positions_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    cols = ["position_id", "side", "entry_time", "exit_time", "pnl_net", "fees", "risk", "legs", "reason", "r_multiple"]
    if trades.empty: return pd.DataFrame(columns=cols)
    pos = trades.groupby("position_id", sort=True).agg(
        side=("side", "first"), entry_time=("entry_time", "first"), exit_time=("exit_time", "last"),
        pnl_net=("pnl_net", "sum"), fees=("fees", "sum"), risk=("position_risk", "first"),
        legs=("trade_id", "count"), reason=("reason", "last")).reset_index()
    pos["r_multiple"] = np.where(pos.risk > 0, pos.pnl_net / pos.risk, np.nan)
    return pos[cols]


def position_stats(pos: pd.DataFrame) -> dict:
    if pos.empty:
        return {"positions": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "profit_factor": None,
                "net_pnl": 0.0, "avg_pnl": 0.0, "expectancy_r": None, "max_consecutive_losses": 0}
    pnl = pos.pnl_net.astype(float)
    gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    streak = best = 0
    for value in pnl:
        streak = streak + 1 if value < 0 else 0
        best = max(best, streak)
    r = pos.r_multiple.dropna()
    return {"positions": int(len(pnl)), "wins": int((pnl > 0).sum()), "losses": int((pnl < 0).sum()),
            "win_rate_pct": float((pnl > 0).mean() * 100), "profit_factor": gp / gl if gl else None,
            "net_pnl": float(pnl.sum()), "avg_pnl": float(pnl.mean()),
            "expectancy_r": float(r.mean()) if len(r) else None, "max_consecutive_losses": int(best)}


def equity_curve_from_trades(trades: pd.DataFrame, capital: float) -> pd.DataFrame:
    cols = ["timestamp", "trade_id", "pnl_net", "cum_pnl", "equity"]
    if trades.empty: return pd.DataFrame(columns=cols)
    out = trades.sort_values(["exit_time", "trade_id"], kind="stable").copy()
    out["cum_pnl"] = out.pnl_net.astype(float).cumsum()
    out["equity"] = capital + out.cum_pnl
    return out[["exit_time", "trade_id", "pnl_net", "cum_pnl", "equity"]].rename(columns={"exit_time": "timestamp"})[cols].reset_index(drop=True)


def calculate_metrics(trades: pd.DataFrame, equity: pd.DataFrame, capital: float) -> dict:
    stats = position_stats(positions_from_trades(trades))
    if equity.empty:
        final, max_dd, min_eq, min_pct = capital, 0.0, capital, 0.0
        peak_eq = trough_eq = capital
        peak_pct = trough_pct = sharpe = sortino = 0.0
    else:
        eq = equity.equity.astype(float)
        final = float(eq.iloc[-1])
        peak = eq.cummax()
        dd = (eq / peak - 1) * 100
        max_dd = float(dd.min())
        min_eq = float(eq.min())
        min_pct = float((eq / capital - 1).min() * 100)
        idx = int(np.argmin(dd.to_numpy()))
        peak_eq, trough_eq = float(peak.iloc[idx]), float(eq.iloc[idx])
        peak_pct, trough_pct = (peak_eq / capital - 1) * 100, (trough_eq / capital - 1) * 100
        daily = equity.set_index("timestamp").equity.resample("1D").last().dropna()
        rets = daily.pct_change().dropna()
        vol = rets.std(ddof=1) if len(rets) > 1 else 0
        sharpe = float(rets.mean() / vol * math.sqrt(365)) if len(rets) > 1 and np.isfinite(vol) and vol > 0 else 0.0
        downside = math.sqrt(float(np.mean(np.minimum(rets.to_numpy(float), 0) ** 2))) if len(rets) else 0
        sortino = float(rets.mean() / downside * math.sqrt(365)) if downside > 0 else 0.0
    net = final - capital
    return {"initial_capital": capital, "final_equity": final, "net_pnl": net,
            "return_pct": net / capital * 100, "max_drawdown_pct": max_dd,
            "max_dd_peak_equity": peak_eq, "max_dd_trough_equity": trough_eq,
            "max_dd_peak_vs_capital_pct": peak_pct, "max_dd_trough_vs_capital_pct": trough_pct,
            "min_equity": min_eq, "min_equity_vs_capital_pct": min_pct,
            "return_over_maxdd": (net / capital * 100) / abs(max_dd) if max_dd < 0 else None,
            "sharpe_daily_365": sharpe, "sortino_daily_365": sortino, "legs": int(len(trades)), **stats}


def jsonable(obj):
    if isinstance(obj, dict): return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)): return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    if isinstance(obj, Path): return str(obj)
    return obj


def dataset_quality(data: pd.DataFrame) -> dict:
    if len(data) < 2: return {"bars": len(data), "gaps_gt_1bar": 0, "max_gap_minutes": 0.0}
    delta = data.index.to_series().diff().dropna().dt.total_seconds() / 60
    median = float(delta.median())
    gaps = delta[delta > median]
    return {"bars": int(len(data)), "gaps_gt_1bar": int(len(gaps)),
            "max_gap_minutes": float(gaps.max()) if len(gaps) else median}


def build_report(cfg: Config, data: pd.DataFrame, trades: pd.DataFrame, metrics: dict, source: str) -> dict:
    sides = {}
    for side in ("LONG", "SHORT"):
        subset = trades[trades.side == side] if not trades.empty else trades
        sides[side] = position_stats(positions_from_trades(subset))
    reasons = {}
    if not trades.empty:
        grouped = trades.groupby("reason", sort=False).agg(legs=("trade_id", "count"), positions=("position_id", "nunique"), pnl_net=("pnl_net", "sum"))
        reasons = {str(k): {"legs": int(v.legs), "positions": int(v.positions), "pnl_net": float(v.pnl_net)} for k, v in grouped.iterrows()}
    return jsonable({"strategy": "strategy2_ema_cross", "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"source": source, "bars": len(data), "start": data.index[0].isoformat(), "end": data.index[-1].isoformat(), "quality": dataset_quality(data)},
        "config": asdict(cfg), "metrics": metrics, "by_side": sides, "exit_reasons": reasons})


def write_outputs(out_dir: str | Path, trades: pd.DataFrame, equity: pd.DataFrame, curve: pd.DataFrame,
                  report: dict, write_equity_csv: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    positions_from_trades(trades).to_csv(out / "positions.csv", index=False)
    curve.to_csv(out / "equity_curve.csv", index=False)
    if write_equity_csv: equity.to_csv(out / "equity.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metrics_path = out / "metrics.json"
    current = jsonable(report["metrics"])
    previous = None
    if metrics_path.exists():
        try: previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): pass
    delta = {}
    if previous is not None:
        for key in sorted(set(previous) | set(current)):
            a, b = previous.get(key), current.get(key)
            numeric = isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool)
            diff = b - a if numeric else None
            delta[key] = {"previous": a, "current": b, "delta": diff,
                          "delta_pct": diff / abs(a) * 100 if numeric and a else None,
                          "changed": diff != 0 if numeric else a != b}
    (out / "metrics_delta.json").write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
        "previous_available": previous is not None, "delta": delta}, indent=2, allow_nan=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(current, indent=2, allow_nan=False), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(report["config"], indent=2, allow_nan=False), encoding="utf-8")
    print(f"[OUT] Archivos guardados en {out.resolve()}", flush=True)


def print_report(metrics: dict, cfg: Config, data: pd.DataFrame):
    print("\n" + "=" * 68 + "\nSTRATEGY 2 — EMA CROSS\n" + "=" * 68)
    print(f"Periodo       : {data.index[0]} -> {data.index[-1]}")
    print(f"Timeframe     : {cfg.timeframe} | EMA{cfg.ema_fast}/{cfg.ema_slow}")
    print(f"Capital       : {metrics['initial_capital']:,.2f} -> {metrics['final_equity']:,.2f}")
    print(f"PnL neto      : {metrics['net_pnl']:,.2f} ({metrics['return_pct']:.2f}%)")
    print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe/Sortino: {metrics['sharpe_daily_365']:.2f} / {metrics['sortino_daily_365']:.2f}")
    print(f"Posiciones    : {metrics['positions']} | tramos: {metrics['legs']}")
    print(f"Win rate      : {metrics['win_rate_pct']:.2f}% | PF: {metrics['profit_factor']}")
    print(f"Expectancy R  : {metrics['expectancy_r']} | racha pérdidas: {metrics['max_consecutive_losses']}")
    print(f"Costes        : comisión {cfg.commission_bps} bps | slippage {cfg.slippage_bps} bps")
    print("=" * 68)


def plot_equity(curve: pd.DataFrame, path: str | Path, capital: float, dpi: int = 150):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib no disponible; se omite el gráfico.")
        return
    if curve.empty:
        print("[PLOT] Sin cierres; no se genera gráfico.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = pd.to_datetime(curve.timestamp, utc=True)
    y = curve.equity.to_numpy(float)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, y, linewidth=1.4, label="Equity")
    ax.axhline(capital, linestyle="--", linewidth=0.9, label=f"Capital inicial ({capital:,.2f})")
    ax.set(title="Strategy 2 — Equity curve", xlabel="Fecha (UTC)", ylabel="Equity")
    ax.grid(True, alpha=.3); ax.legend()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def generate_report(trades: pd.DataFrame, equity: pd.DataFrame, cfg: Config, data: pd.DataFrame,
                    source: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR,
                    write_outputs_enabled: bool = True, write_equity_csv: bool = False,
                    plot_enabled: bool = True, plot_path: str | Path | None = None,
                    plot_dpi: int = 150) -> dict:
    """Calcula métricas, imprime el informe, guarda outputs y genera el gráfico.

    Punto de entrada único que unifica `calculate_metrics`, `print_report`,
    `equity_curve_from_trades`, `build_report`, `write_outputs` y `plot_equity`.
    Devuelve el diccionario de métricas calculado.
    """
    metrics = calculate_metrics(trades, equity, cfg.initial_capital)
    print_report(metrics, cfg, data)
    curve = equity_curve_from_trades(trades, cfg.initial_capital)
    if write_outputs_enabled:
        report = build_report(cfg, data, trades, metrics, source)
        write_outputs(output_dir, trades, equity, curve, report, write_equity_csv)
    if plot_enabled:
        plot_equity(curve, plot_path or str(Path(output_dir) / "equity_curve.png"),
                    cfg.initial_capital, plot_dpi)
    return metrics
