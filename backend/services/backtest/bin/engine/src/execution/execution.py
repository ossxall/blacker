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

from orders.models import Fill, Order, OrderType, Side
from orders.order_manager import OrderManager


class Execution:
    """
    Simulates order matching against the tick stream.

    The trigger pass runs *before* the strategy is evaluated, so
    protective orders (stop-loss / targets) are resolved first and
    the strategy only ever sees the resulting, consistent state.

    Orders submitted by the strategy on tick *t* start working and
    are matched starting on tick *t + 1*.
    """

    def __init__(self, order_manager: OrderManager):
        self._order_manager = order_manager

    def submit(self, orders: list[Order]) -> None:
        self._order_manager.submit(orders)

    def update(self, state, tick) -> list[Fill]:
        self._order_manager.update_trailing_stops(tick.price)

        fills: list[Fill] = []

        for order in self._order_manager.working_orders():
            fill_price = self._trigger_price(order, tick)

            if fill_price is None:
                continue

            fill = self._order_manager.make_fill(
                order=order,
                price=fill_price,
                quantity=order.quantity,
                tick=tick,
            )

            if fill.quantity <= 0.0:
                continue

            fills.append(fill)

            new_orders = self._order_manager.on_fill(fill)
            self._order_manager.submit(new_orders)

        return fills

    def _trigger_price(self, order: Order, tick) -> Optional[float]:
        price = tick.price

        if order.type == OrderType.MARKET:
            return price

        if order.type == OrderType.LIMIT:
            if order.side == Side.BUY and price <= order.price:
                return price
            if order.side == Side.SELL and price >= order.price:
                return price
            return None

        if order.type == OrderType.STOP:
            if order.side == Side.BUY and price >= order.price:
                return price
            if order.side == Side.SELL and price <= order.price:
                return price
            return None

        return None