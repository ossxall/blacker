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
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):

    def evaluate(
        self,
        state: EngineState
    ):

        tf = state.timeframes.get("30m")

        if tf is None:
            return None

        rt = self._get_reversal_trap(tf)

        # The ReversalTrap series is not part of the layout or is still
        # inside its warm-up window (live is suppressed).
        if rt is None or rt.live is None:
            return None

        live = rt.live

        portfolio = state.portfolio

        position = portfolio.position if portfolio is not None else None

        # --------------------------------------------------
        # FLAT
        # --------------------------------------------------

        if position is None:

            if not live.allowed_by_limits:
                return None

            if live.bull_trap:
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            if live.bear_trap:
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # --------------------------------------------------
        # LONG
        # --------------------------------------------------

        if position.side == Side.BUY:

            if live.bear_trap or not live.active_bull:
                return Signal(
                    action="EXIT",
                )

            return None

        # --------------------------------------------------
        # SHORT
        # --------------------------------------------------

        if position.side == Side.SELL:

            if live.bull_trap or not live.active_bear:
                return Signal(
                    action="EXIT",
                )

            return None

        return None

    def _get_reversal_trap(self, tf):
        """
        Resolves the ReversalTrap series of the timeframe.

        The canonical label matches the dashboard's series registry entry;
        if it is absent the first ReversalTrap series found is used so the
        strategy still works under any configured label.
        """
        try:
            return tf.get_series("ReversalTrap", "Reversal Trap")
        except KeyError:
            pass

        for name in ("Reversal Trap", "RT", "ReversalTrap"):
            try:
                return tf.get_series("ReversalTrap", name)
            except KeyError:
                continue

        return None