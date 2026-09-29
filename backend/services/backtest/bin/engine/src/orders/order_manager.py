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
import math

from core.portfolio import QUANTITY_EPSILON, Portfolio
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

    Invariants the rest of the engine relies on:

        * At most one entry order is working at a time, and at most one
          position is open.
        * A working order always has a positive fill quantity, or it is
          cancelled. Orders are never left working with nothing to fill.
        * The bracket of a group is created on the entry fill. It is
          OCO: the first exit fill that flattens the position cancels
          the rest, while a partial one leaves the residual protected by
          the same stop (shrunk to the quantity that is left) and the
          same targets (capped by the group when they fill).
        * The bracket is static. A stop is priced once against the entry
          fill and is never moved again, so no order in the book depends
          on where price travelled afterwards.
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
            return self._exit_position(signal)

        raise ValueError(f"Unknown signal action: {signal.action}")

    @staticmethod
    def _positive_quantity(value, action: str) -> float:
        """
        Quantity a signal asks for, or a loud failure.

        A non-positive or non-finite quantity can never fill, and the
        order would sit in the book as WORKING forever while the
        strategy keeps seeing the account it asked for.
        """
        quantity = float(value)

        if not math.isfinite(quantity) or quantity <= 0.0:
            raise ValueError(
                f"{action} signal requires a positive quantity, got {value!r}"
            )

        return quantity

    def _open(self, signal: Signal, side: Side) -> list[Order]:
        # Single-position engine: ignore entries while a position is open.
        if self._portfolio.position is not None:
            return []

        # A market entry is filled by the next tick, so until then there is
        # no position to block a second entry. Without this guard two
        # entries could be accepted on consecutive ticks and both would
        # fill, doubling the exposure the strategy asked for once.
        if self._pending_entry() is not None:
            return []

        if signal.quantity is None:
            raise ValueError(f"{signal.action} signal requires quantity")

        quantity = self._positive_quantity(signal.quantity, signal.action)

        group = self._new_group(side, quantity)

        order = self._new_order(
            side=side,
            order_type=OrderType.MARKET,
            role=OrderRole.ENTRY,
            quantity=quantity,
            price=None,
            group_id=group.id,
        )

        group.entry_order_id = order.id

        return [order]

    def _pending_entry(self) -> Optional[Order]:
        for order in self._working.values():
            if order.role == OrderRole.ENTRY:
                return order

        return None

    def _open_group(self) -> Optional[OrderGroup]:
        """
        The single group whose remaining quantity still backs a position.

        Cancellation and order creation must agree on which group is
        live, otherwise an exit order can be attached to a group that was
        already closed and its fills would be accounted against the wrong
        remaining quantity.
        """
        for group in self._groups.values():
            if (
                group.remaining_quantity > QUANTITY_EPSILON
                and group.entry_order_id is not None
            ):
                return group

        return None

    def _exit_position(self, signal: Optional[Signal] = None) -> list[Order]:
        position = self._portfolio.position

        if position is None:
            return []

        group = self._open_group()

        quantity = self._exit_quantity(signal, position)

        # An exit that leaves the position flat invalidates the whole
        # bracket. A partial one does not: the residual stays protected by
        # the same stop and targets, shrunk to whatever is left of the
        # group when they fill (see ``_after_exit_fill``).
        if group is not None and quantity >= position.quantity - QUANTITY_EPSILON:
            self._cancel_children(group)

        side = Side.SELL if position.side == Side.BUY else Side.BUY

        exit_order = self._new_order(
            side=side,
            order_type=OrderType.MARKET,
            role=OrderRole.EXIT,
            quantity=quantity,
            price=None,
            group_id=group.id if group is not None else 0,
        )

        # The exit belongs to the bracket: if the position closes by any
        # other route first -- a stop filling, a target filling -- this
        # order is cancelled with the rest. Without it a partial exit that
        # never filled would stay WORKING with nothing left to fill, and
        # the book would keep an order for a position that no longer
        # exists.
        if group is not None:
            group.exit_order_ids.append(exit_order.id)

        return [exit_order]

    def _exit_quantity(self, signal: Optional[Signal], position) -> float:
        """
        Quantity an EXIT asks to close.

        No quantity means "close everything", which is what a strategy
        that only ever trades one position wants. A quantity closes that
        many units and leaves the rest of the position open and protected.

        A quantity larger than the position is capped rather than
        rejected: the venue would fill what exists, and a strategy sizing
        an exit off a position it already partly closed should see that
        exit fill, not kill the run.
        """
        if signal is None or signal.quantity is None:
            return position.quantity

        requested = self._positive_quantity(signal.quantity, "EXIT")

        return min(requested, position.quantity)

    # ======================================================
    # FILLS
    # ======================================================

    def fill_quantity(self, order: Order) -> float:
        """
        Quantity ``order`` would actually fill right now.

        Protective orders are capped by what is left of the group, so an
        order can be working while nothing is left to fill it. Callers
        must retire those orders instead of leaving them in the book.

        A leftover of floating-point dust counts as nothing left: a group
        holding 1.1e-16 is closed, and reporting it as fillable would book
        a fill that moves nothing and leave the order working forever.
        """
        group = self._groups.get(order.group_id)

        if group is not None and order.role != OrderRole.ENTRY:
            if group.remaining_quantity <= QUANTITY_EPSILON:
                return 0.0

            return min(order.quantity, group.remaining_quantity)

        return order.quantity

    def make_fill(
        self,
        order: Order,
        price: float,
        quantity: float,
        tick,
    ) -> Fill:
        quantity = min(quantity, self.fill_quantity(order))

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
            group.remaining_quantity = max(
                0.0, group.remaining_quantity - fill.quantity
            )

        self._portfolio.apply_fill(fill)

        if order.role == OrderRole.ENTRY:
            return self._create_bracket(group, fill)

        if group is not None and order.id in group.exit_order_ids:
            group.exit_order_ids.remove(order.id)

        return self._after_exit_fill(order, group, fill.quantity)

    def _create_bracket(self, group: Optional[OrderGroup], fill: Fill) -> list[Order]:
        if group is None:
            return []

        levels = self._risk_manager.apply(group.side, fill.price)

        exit_side = Side.SELL if group.side == Side.BUY else Side.BUY

        orders = []

        entry_quantity = group.entry_quantity

        # The stop is created first so that it takes matching priority over
        # the targets within a tick: on a tie the conservative order wins.
        if levels.stop_price is not None:
            stop = self._new_order(
                side=exit_side,
                order_type=OrderType.STOP,
                role=OrderRole.STOP,
                quantity=entry_quantity,
                price=levels.stop_price,
                group_id=group.id,
            )

            group.stop_order_id = stop.id
            orders.append(stop)

        total = len(levels.target_prices)

        if total:
            used = 0.0

            for index, price in enumerate(levels.target_prices):
                if index == total - 1:
                    target_quantity = entry_quantity - used
                else:
                    target_quantity = entry_quantity / total

                used += target_quantity

                # A target that rounds down to nothing would sit in the
                # book working forever without ever being fillable.
                if target_quantity <= QUANTITY_EPSILON:
                    continue

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

        if order.role in (OrderRole.TARGET, OrderRole.EXIT):
            if group is not None and group.remaining_quantity <= QUANTITY_EPSILON:
                self._cancel_children(group, exclude=order.id)
            else:
                # Quantity is left on the position, so the stop shrinks to
                # match. The sibling targets keep their own size: they are
                # capped by what is left of the group when they fill, which
                # is what lets a ladder survive a partial exit untouched.
                self._reduce_stop_quantity(group, fill_quantity)
            return []

        return []

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

    # ======================================================
    # BOOK MAINTENANCE
    # ======================================================

    def prune(self, keep: int = 200) -> tuple[int, int]:
        """
        Drops settled history so the book stays proportional to live exposure.

        The engine rebuilds and republishes the whole book on every tick, so
        a book that only ever grows makes a linear engine quadratic: each
        terminal order is reserialized for the rest of the run and the
        payload grows with the length of the replay instead of with the
        position being managed.

        Two things are never dropped:

            * a WORKING order, because that is the engine's live truth;
            * a group that still backs quantity, even if nothing of it is
              working -- an exposed position with no protective order is an
              anomaly, and hiding its record would hide the anomaly.

        The last ``keep`` settled orders and groups are retained so the
        client can still render the trades that just happened. Returns the
        number of orders and groups removed.
        """
        if keep < 0:
            return 0, 0

        removed_orders = 0

        settled = [
            order_id
            for order_id, order in self._orders.items()
            if order.status is not OrderStatus.WORKING
        ]

        for order_id in settled[: max(0, len(settled) - keep)]:
            del self._orders[order_id]
            self._working.pop(order_id, None)
            removed_orders += 1

        closed = [
            group_id
            for group_id, group in self._groups.items()
            if group.remaining_quantity <= QUANTITY_EPSILON
            and not self._group_is_working(group)
        ]

        removed_groups = 0

        for group_id in closed[: max(0, len(closed) - keep)]:
            del self._groups[group_id]
            removed_groups += 1

        return removed_orders, removed_groups

    def _group_is_working(self, group: OrderGroup) -> bool:
        order_ids = list(group.target_order_ids)

        if group.entry_order_id is not None:
            order_ids.append(group.entry_order_id)

        if group.stop_order_id is not None:
            order_ids.append(group.stop_order_id)

        for order_id in order_ids:
            order = self._orders.get(order_id)

            if order is not None and order.status is OrderStatus.WORKING:
                return True

        return False

    def _reduce_stop_quantity(self, group: Optional[OrderGroup], amount: float) -> None:
        if group is None or group.stop_order_id is None:
            return

        stop = self._orders.get(group.stop_order_id)

        if stop is None or stop.status != OrderStatus.WORKING:
            return

        stop.quantity = max(0.0, stop.quantity - amount)

        if stop.quantity <= QUANTITY_EPSILON:
            self.cancel(stop.id)

    def _cancel_children(self, group: Optional[OrderGroup], exclude: Optional[int] = None) -> None:
        if group is None:
            return

        order_ids = list(group.target_order_ids)
        if group.stop_order_id is not None:
            order_ids.append(group.stop_order_id)
        order_ids.extend(group.exit_order_ids)

        for order_id in order_ids:
            if order_id == exclude:
                continue
            self.cancel(order_id)

    def cancel(self, order_id: int) -> None:
        order = self._orders.get(order_id)

        if order is None or order.status != OrderStatus.WORKING:
            return

        order.status = OrderStatus.CANCELLED
        self._working.pop(order_id, None)