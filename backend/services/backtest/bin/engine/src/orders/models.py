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

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.portfolio import Side


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    WORKING = "WORKING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class OrderRole(str, Enum):
    ENTRY = "ENTRY"
    TARGET = "TARGET"
    STOP = "STOP"
    EXIT = "EXIT"


@dataclass
class Signal:
    """
    Alpha intent emitted by a Strategy.

    The signal only carries *what* the strategy wants: an entry
    (BUY / SELL) or a close (EXIT), with the desired quantity.

    Protective orders (stop-loss, take-profit targets, trailing)
    are attached later by the RiskManager, not by the strategy.
    """

    action: str
    quantity: Optional[float] = None


@dataclass
class Order:
    id: int
    side: Side
    type: OrderType
    role: OrderRole
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.WORKING
    group_id: int = 0
    trailing: bool = False
    trailing_distance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "side": self.side.value,
            "type": self.type.value,
            "role": self.role.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "group_id": self.group_id,
            "trailing": self.trailing,
            "trailing_distance": self.trailing_distance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        return cls(
            id=data["id"],
            side=Side(data["side"]),
            type=OrderType(data["type"]),
            role=OrderRole(data["role"]),
            quantity=data["quantity"],
            price=data.get("price"),
            status=OrderStatus(data["status"]),
            group_id=data.get("group_id", 0),
            trailing=data.get("trailing", False),
            trailing_distance=data.get("trailing_distance", 0.0),
        )


@dataclass
class Fill:
    id: int
    order_id: int
    group_id: int
    role: OrderRole
    side: Side
    price: float
    quantity: float
    time: int
    tick_index: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "order_id": self.order_id,
            "group_id": self.group_id,
            "role": self.role.value,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "time": self.time,
            "tick_index": self.tick_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fill":
        return cls(
            id=data["id"],
            order_id=data["order_id"],
            group_id=data["group_id"],
            role=OrderRole(data["role"]),
            side=Side(data["side"]),
            price=data["price"],
            quantity=data["quantity"],
            time=data["time"],
            tick_index=data["tick_index"],
        )


@dataclass
class OrderGroup:
    """
    A bracket of orders tied to one entry.

    When the ENTRY order fills, the group receives a STOP order
    and one or more TARGET orders. All siblings are OCO: the first
    exit fill cancels the remaining ones.
    """

    id: int
    side: Side
    entry_quantity: float
    entry_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    target_order_ids: list[int] = field(default_factory=list)
    remaining_quantity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "side": self.side.value,
            "entry_quantity": self.entry_quantity,
            "entry_order_id": self.entry_order_id,
            "stop_order_id": self.stop_order_id,
            "target_order_ids": list(self.target_order_ids),
            "remaining_quantity": self.remaining_quantity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OrderGroup":
        return cls(
            id=data["id"],
            side=Side(data["side"]),
            entry_quantity=data["entry_quantity"],
            entry_order_id=data.get("entry_order_id"),
            stop_order_id=data.get("stop_order_id"),
            target_order_ids=list(data.get("target_order_ids", [])),
            remaining_quantity=data.get("remaining_quantity", 0.0),
        )