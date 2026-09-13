# BLACKER
# Copyright (C) 2026 Juan José Caballero Rey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    side: Side
    quantity: float
    avg_price: float
    entry_time: int
    entry_tick_index: int

    def unrealized_pnl(self, price: float) -> float:
        if self.side == Side.BUY:
            return (price - self.avg_price) * self.quantity
        return (self.avg_price - price) * self.quantity

    def to_dict(self) -> dict:
        return {
            "side": self.side.value,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "entry_time": self.entry_time,
            "entry_tick_index": self.entry_tick_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            side=Side(data["side"]),
            quantity=data["quantity"],
            avg_price=data["avg_price"],
            entry_time=data["entry_time"],
            entry_tick_index=data["entry_tick_index"],
        )


@dataclass
class Trade:
    """A completed round trip used for backtest analytics."""

    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: int
    exit_time: int
    pnl: float

    def to_dict(self) -> dict:
        return {
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "pnl": self.pnl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        return cls(
            side=Side(data["side"]),
            quantity=data["quantity"],
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            entry_time=data["entry_time"],
            exit_time=data["exit_time"],
            pnl=data["pnl"],
        )


class Portfolio:
    """
    Single source of truth for the account: cash, positions and PnL.

    The OrderManager delegates every fill to this object, so the
    strategy can derive its state from ``state.portfolio.position``
    instead of keeping its own duplicated flags.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position: Optional[Position] = None
        self.realized_pnl: float = 0.0
        self.trades: list[Trade] = []

    def equity(self, price: Optional[float] = None) -> float:
        if self.position is None:
            return self.cash

        if price is None:
            price = self.position.avg_price

        if self.position.side == Side.BUY:
            market_value = self.position.quantity * price
        else:
            market_value = -self.position.quantity * price

        return self.cash + market_value

    def apply_fill(self, fill) -> None:
        role = fill.role.value

        if role == "ENTRY":
            self._entry(fill)
        elif role in ("STOP", "TARGET", "EXIT"):
            self._reduce(fill)

    def _entry(self, fill) -> None:
        quantity = fill.quantity
        position = self.position

        if fill.side == Side.BUY:
            self.cash -= quantity * fill.price
        else:
            self.cash += quantity * fill.price

        if position is not None and position.side == fill.side:
            total_quantity = position.quantity + quantity
            position.avg_price = (
                (position.avg_price * position.quantity) + (fill.price * quantity)
            ) / total_quantity
            position.quantity = total_quantity
            return

        self.position = Position(
            side=fill.side,
            quantity=quantity,
            avg_price=fill.price,
            entry_time=fill.time,
            entry_tick_index=fill.tick_index,
        )

    def _reduce(self, fill) -> None:
        position = self.position

        if position is None:
            return

        quantity = min(fill.quantity, position.quantity)

        if position.side == Side.BUY:
            self.cash += quantity * fill.price
            pnl = (fill.price - position.avg_price) * quantity
        else:
            self.cash -= quantity * fill.price
            pnl = (position.avg_price - fill.price) * quantity

        self.realized_pnl += pnl

        self.trades.append(
            Trade(
                side=position.side,
                quantity=quantity,
                entry_price=position.avg_price,
                exit_price=fill.price,
                entry_time=position.entry_time,
                exit_time=fill.time,
                pnl=pnl,
            )
        )

        position.quantity -= quantity

        if position.quantity <= 0:
            self.position = None

    def to_dict(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "position": (
                self.position.to_dict() if self.position is not None else None
            ),
            "realized_pnl": self.realized_pnl,
            "trades": [trade.to_dict() for trade in self.trades],
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Portfolio":
        if not data:
            return cls()

        portfolio = cls(
            initial_cash=float(data.get("initial_cash", 100_000.0)),
        )
        portfolio.cash = float(data.get("cash", portfolio.initial_cash))
        portfolio.realized_pnl = float(data.get("realized_pnl", 0.0))
        portfolio.trades = [
            Trade.from_dict(trade) for trade in data.get("trades", [])
        ]

        position = data.get("position")
        if position is not None:
            portfolio.position = Position.from_dict(position)

        return portfolio