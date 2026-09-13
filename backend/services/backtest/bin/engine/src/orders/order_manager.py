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

from typing import Optional

from core.portfolio import Portfolio
from orders.models import (
    Fill,
    Order,
    OrderGroup,
    OrderRole,
    OrderStatus,
    OrderType,
    Side,
    Signal,
)
from orders.risk_manager import RiskManager


class OrderManager:
    """
    Owns the order lifecycle and the bracket (OCO) groups.

    Pipeline:

        1. ``handle(signal)``        -> translate strategy intent into orders.
        2. ``make_fill`` / ``on_fill`` -> execute fills, open positions and
           place the stop-loss + take-profit bracket on entry.
        3. ``update_trailing_stops`` -> ratchet trailing stops with price.
    """

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        self._portfolio = portfolio or Portfolio()
        self._risk_manager = risk_manager or RiskManager()

        self._next_order_id = 0
        self._next_group_id = 0
        self._next_fill_id = 0

        self._orders: dict[int, Order] = {}
        self._working: dict[int, Order] = {}
        self._groups: dict[int, OrderGroup] = {}

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk_manager

    # ======================================================
    # SERIALIZATION
    # ======================================================

    def to_dict(self) -> dict:
        return {
            "next_order_id": self._next_order_id,
            "next_group_id": self._next_group_id,
            "next_fill_id": self._next_fill_id,
            "orders": [order.to_dict() for order in self._orders.values()],
            "groups": [group.to_dict() for group in self._groups.values()],
        }

    @classmethod
    def from_dict(
        cls,
        data: Optional[dict],
        portfolio: Optional[Portfolio] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> "OrderManager":
        data = data or {}

        manager = cls(portfolio=portfolio, risk_manager=risk_manager)
        manager._next_order_id = data.get("next_order_id", 0)
        manager._next_group_id = data.get("next_group_id", 0)
        manager._next_fill_id = data.get("next_fill_id", 0)

        for order_data in data.get("orders", []):
            order = Order.from_dict(order_data)
            manager._orders[order.id] = order
            if order.status == OrderStatus.WORKING:
                manager._working[order.id] = order

        for group_data in data.get("groups", []):
            group = OrderGroup.from_dict(group_data)
            manager._groups[group.id] = group

        return manager

    # ======================================================
    # ORDER / GROUP CREATION
    # ======================================================

    def _new_order(
        self,
        side: Side,
        order_type: OrderType,
        role: OrderRole,
        quantity: float,
        price: Optional[float],
        group_id: int = 0,
        *,
        trailing: bool = False,
        trailing_distance: float = 0.0,
    ) -> Order:
        self._next_order_id += 1

        order = Order(
            id=self._next_order_id,
            side=side,
            type=order_type,
            role=role,
            quantity=quantity,
            price=price,
            group_id=group_id,
            trailing=trailing,
            trailing_distance=trailing_distance,
        )

        self._orders[order.id] = order
        self._working[order.id] = order

        return order

    def _new_group(self, side: Side, quantity: float) -> OrderGroup:
        self._next_group_id += 1

        group = OrderGroup(
            id=self._next_group_id,
            side=side,
            entry_quantity=quantity,
            remaining_quantity=quantity,
        )

        self._groups[group.id] = group

        return group

    # ======================================================
    # SIGNAL HANDLING
    # ======================================================

    def handle(self, signal: Optional[Signal]) -> list[Order]:
        if signal is None:
            return []

        if signal.action == "BUY":
            return self._open(signal, Side.BUY)

        if signal.action == "SELL":
            return self._open(signal, Side.SELL)

        if signal.action == "EXIT":
            return self._exit_position()

        raise ValueError(f"Unknown signal action: {signal.action}")

    def _open(self, signal: Signal, side: Side) -> list[Order]:
        # Single-position engine: ignore entries while a position is open.
        if self._portfolio.position is not None:
            return []

        if signal.quantity is None:
            raise ValueError(f"{signal.action} signal requires quantity")

        group = self._new_group(side, signal.quantity)

        order = self._new_order(
            side=side,
            order_type=OrderType.MARKET,
            role=OrderRole.ENTRY,
            quantity=signal.quantity,
            price=None,
            group_id=group.id,
        )

        group.entry_order_id = order.id

        return [order]

    def _exit_position(self) -> list[Order]:
        position = self._portfolio.position

        if position is None:
            return []

        # An explicit exit invalidates all protective brackets.
        for group in self._groups.values():
            if group.remaining_quantity > 0 and group.entry_order_id is not None:
                self._cancel_children(group)

        group_id = next(
            (
                group.id
                for group in self._groups.values()
                if group.remaining_quantity > 0
            ),
            0,
        )

        side = Side.SELL if position.side == Side.BUY else Side.BUY

        return [
            self._new_order(
                side=side,
                order_type=OrderType.MARKET,
                role=OrderRole.EXIT,
                quantity=position.quantity,
                price=None,
                group_id=group_id,
            )
        ]

    # ======================================================
    # FILLS
    # ======================================================

    def make_fill(
        self,
        order: Order,
        price: float,
        quantity: float,
        tick,
    ) -> Fill:
        group = self._groups.get(order.group_id)

        if group is not None and order.role != OrderRole.ENTRY:
            quantity = min(quantity, max(group.remaining_quantity, 0.0))

        self._next_fill_id += 1

        return Fill(
            id=self._next_fill_id,
            order_id=order.id,
            group_id=order.group_id,
            role=order.role,
            side=order.side,
            price=price,
            quantity=quantity,
            time=tick.time,
            tick_index=tick.tick_index,
        )

    def on_fill(self, fill: Fill) -> list[Order]:
        order = self._orders.get(fill.order_id)

        if order is None or order.status != OrderStatus.WORKING:
            return []

        order.status = OrderStatus.FILLED
        self._working.pop(order.id, None)

        group = self._groups.get(fill.group_id)

        if group is not None and order.role != OrderRole.ENTRY:
            group.remaining_quantity = max(0.0, group.remaining_quantity - fill.quantity)

        self._portfolio.apply_fill(fill)

        if order.role == OrderRole.ENTRY:
            return self._create_bracket(group, fill)

        return self._after_exit_fill(order, group, fill.quantity)

    def _create_bracket(self, group: Optional[OrderGroup], fill: Fill) -> list[Order]:
        if group is None:
            return []

        levels = self._risk_manager.apply(group.side, fill.price)

        exit_side = Side.SELL if group.side == Side.BUY else Side.BUY

        orders = []

        entry_quantity = group.entry_quantity

        if levels.target_prices:
            used = 0.0
            for index, price in enumerate(levels.target_prices):
                if index == len(levels.target_prices) - 1:
                    target_quantity = entry_quantity - used
                else:
                    target_quantity = entry_quantity / len(levels.target_prices)

                used += target_quantity

                order = self._new_order(
                    side=exit_side,
                    order_type=OrderType.LIMIT,
                    role=OrderRole.TARGET,
                    quantity=target_quantity,
                    price=price,
                    group_id=group.id,
                )

                group.target_order_ids.append(order.id)
                orders.append(order)

        if levels.stop_price is not None:
            stop = self._new_order(
                side=exit_side,
                order_type=OrderType.STOP,
                role=OrderRole.STOP,
                quantity=entry_quantity,
                price=levels.stop_price,
                group_id=group.id,
                trailing=levels.trailing_distance > 0.0,
                trailing_distance=levels.trailing_distance,
            )

            group.stop_order_id = stop.id
            orders.append(stop)

        return orders

    def _after_exit_fill(
        self,
        order: Order,
        group: Optional[OrderGroup],
        fill_quantity: float,
    ) -> list[Order]:
        if order.role == OrderRole.STOP:
            # The stop consumed the remaining quantity. Cancel all targets.
            self._cancel_children(group, exclude=order.id)
            return []

        if order.role == OrderRole.TARGET:
            if group is not None and group.remaining_quantity <= 0:
                self._cancel_children(group, exclude=order.id)
            else:
                self._reduce_stop_quantity(group, fill_quantity)
            return []

        if order.role == OrderRole.EXIT:
            self._cancel_children(group, exclude=order.id)
            return []

        return []

    # ======================================================
    # TRAILING STOPS
    # ======================================================

    def update_trailing_stops(self, price: float) -> None:
        for group in self._groups.values():
            stop_id = group.stop_order_id

            if stop_id is None:
                continue

            stop = self._orders.get(stop_id)

            if stop is None or stop.status != OrderStatus.WORKING:
                continue

            if not stop.trailing or stop.trailing_distance <= 0.0:
                continue

            if group.side == Side.BUY:
                candidate = price - stop.trailing_distance
                if candidate > stop.price:
                    stop.price = candidate
            else:
                candidate = price + stop.trailing_distance
                if candidate < stop.price:
                    stop.price = candidate

    # ======================================================
    # UTILITIES
    # ======================================================

    def submit(self, orders: list[Order]) -> None:
        for order in orders:
            if order.status == OrderStatus.WORKING:
                self._working[order.id] = order

    def working_orders(self) -> list[Order]:
        return list(self._working.values())

    def active_orders(self) -> list[Order]:
        return self.working_orders()

    def groups(self) -> list[OrderGroup]:
        return list(self._groups.values())

    def _reduce_stop_quantity(self, group: Optional[OrderGroup], amount: float) -> None:
        if group is None or group.stop_order_id is None:
            return

        stop = self._orders.get(group.stop_order_id)

        if stop is None or stop.status != OrderStatus.WORKING:
            return

        stop.quantity = max(0.0, stop.quantity - amount)

        if stop.quantity <= 0.0:
            self._cancel_order(stop.id)

    def _cancel_children(self, group: Optional[OrderGroup], exclude: Optional[int] = None) -> None:
        if group is None:
            return

        order_ids = list(group.target_order_ids)
        if group.stop_order_id is not None:
            order_ids.append(group.stop_order_id)

        for order_id in order_ids:
            if order_id == exclude:
                continue
            self._cancel_order(order_id)

    def _cancel_order(self, order_id: int) -> None:
        order = self._orders.get(order_id)

        if order is None or order.status != OrderStatus.WORKING:
            return

        order.status = OrderStatus.CANCELLED
        self._working.pop(order_id, None)