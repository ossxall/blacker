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

from core.engine_state import EngineState
from .base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):

    def evaluate(
        self,
        state: EngineState
    ):

        tf = state.timeframes.get("1m")

        ema_55 = tf.get_series(
            "EMA",
            "EMA 55"
        )

        ema_200 = tf.get_series(
            "EMA",
            "EMA 200"
        )

        if not ema_55.history or not ema_200.history:
            return None

        previous_55 = ema_55.history[-1]
        previous_200 = ema_200.history[-1]

        current_55 = ema_55.live
        current_200 = ema_200.live

        cross_up = (
            previous_55.value <= previous_200.value
            and current_55.value > current_200.value
        )

        cross_down = (
            previous_55.value >= previous_200.value
            and current_55.value < current_200.value
        )

        # --------------------------------------------------
        # The position is owned by the portfolio.
        # --------------------------------------------------

        portfolio = state.portfolio

        position = portfolio.position if portfolio is not None else None

        # --------------------------------------------------
        # FLAT
        # --------------------------------------------------

        if position is None:

            if cross_up:
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            if cross_down:
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # --------------------------------------------------
        # LONG
        # --------------------------------------------------

        if position.side == Side.BUY:

            if cross_down:
                return Signal(
                    action="EXIT",
                )

            return None

        # --------------------------------------------------
        # SHORT
        # --------------------------------------------------

        if position.side == Side.SELL:

            if cross_up:
                return Signal(
                    action="EXIT",
                )

            return None

        return None