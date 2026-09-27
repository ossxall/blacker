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

from orders.models import Fill, Order, OrderStatus, OrderType, Side
from orders.order_manager import OrderManager


class Execution:
    """
    Simulates order matching against the tick stream.

    The trigger pass runs *before* the strategy is evaluated, so
    protective orders (stop-loss / targets) are resolved first and
    the strategy only ever sees the resulting, consistent state.

    Orders submitted by the strategy on tick *t* start working and
    are matched starting on tick *t + 1*.

    ``working_orders()`` is a snapshot, so an order can be retired by an
    earlier fill of the same pass (OCO) and still be present in the list
    being iterated. Every order is therefore re-checked against its live
    status before it is triggered, and a fill is only reported once the
    OrderManager has actually applied it.
    """

    def __init__(self, order_manager: OrderManager):
        self._order_manager = order_manager

    def submit(self, orders: list[Order]) -> None:
        self._order_manager.submit(orders)

    def update(self, state, tick) -> list[Fill]:
        self._order_manager.update_trailing_stops(tick.price)

        fills: list[Fill] = []

        for order in self._order_manager.working_orders():
            # Cancelled by a sibling fill earlier in this same pass.
            if order.status is not OrderStatus.WORKING:
                continue

            fill_price = self._trigger_price(order, tick)

            if fill_price is None:
                continue

            quantity = self._order_manager.fill_quantity(order)

            if quantity <= 0.0:
                # The group is already flat: retire the order instead of
                # leaving a working order in the published book that can
                # never fill.
                self._order_manager.cancel(order.id)
                continue

            fill = self._order_manager.make_fill(
                order=order,
                price=fill_price,
                quantity=quantity,
                tick=tick,
            )

            new_orders = self._order_manager.on_fill(fill)

            # on_fill refuses orders it no longer considers working, so this
            # is the only point at which the fill is known to be real.
            if order.status is not OrderStatus.FILLED:
                continue

            fills.append(fill)

            self._order_manager.submit(new_orders)

        return fills

    def _trigger_price(self, order: Order, tick) -> Optional[float]:
        price = tick.price

        if order.type == OrderType.MARKET:
            return price

        if order.price is None or price is None:
            return None

        if order.side == Side.BUY:
            if order.type == OrderType.LIMIT:
                # Improvement: a buy limit never pays more than its limit.
                return price if price <= order.price else None

            if order.type == OrderType.STOP:
                # A triggered buy stop becomes a market order.
                return price if price >= order.price else None

            return None

        if order.type == OrderType.LIMIT:
            # Improvement: a sell limit crossed by a gap fills at the limit,
            # not at the print. Filling take-profits at the tick price books
            # the whole gap as profit and flatters every backtest.
            return order.price if price >= order.price else None

        if order.type == OrderType.STOP:
            # A triggered sell stop becomes a market order: no improvement.
            return price if price <= order.price else None

        return None
