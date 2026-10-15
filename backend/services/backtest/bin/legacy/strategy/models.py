from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class Position:
    position_id: int
    side: int
    quantity: float
    initial_quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_fee: float
    stop_pct: float
    risk_amount: float
    breakeven: bool = False
    partial_done: bool = False
    touch_done: bool = False


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
CROSS_REASON = "EMA_CROSS_EXIT"
DEFAULT_OUTPUT_DIR = "backtest_out"


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TRADE_COLUMNS})

