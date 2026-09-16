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
    """
    Strategy driven exclusively by ReversalTrap signals on 30m.

    Entries:
        - bull_trap -> BUY
        - bear_trap -> SELL

    Exits:
        - LONG: opposite bear_trap, or ReversalTrap's active_bull becomes
          false after its target/stop evaluation.
        - SHORT: opposite bull_trap, or ReversalTrap's active_bear becomes
          false after its target/stop evaluation.

    EMA 55 / EMA 200 trend filtering and crossover exits are intentionally
    removed: ReversalTrap is the sole signal source for direction.
    """

    def evaluate(self, state: EngineState):
        tf = state.timeframes.get("30m")

        if tf is None:
            return None

        rt = self._get_reversal_trap(tf)

        # ReversalTrap is the only indicator required by this strategy.
        if rt is None:
            return None

        live = rt.live

        # ReversalTrap has no usable signal during its warm-up period.
        if live is None:
            return None

        portfolio = state.portfolio
        position = portfolio.position if portfolio is not None else None

        # --------------------------------------------------
        # FLAT -> ENTRY
        # --------------------------------------------------

        if position is None:
            if not live.allowed_by_limits:
                return None

            # ReversalTrap bull signal = long entry.
            if live.bull_trap:
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            # ReversalTrap bear signal = short entry.
            if live.bear_trap:
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # --------------------------------------------------
        # LONG -> EXIT
        # --------------------------------------------------

        if position.side == Side.BUY:
            # Opposite ReversalTrap signal closes the long.
            if live.bear_trap:
                return Signal(action="EXIT")

            # ReversalTrap internally deactivates the bull trade when its
            # target or stop is reached (with target-before-stop
            # ordering). Mirror that exit at strategy level.
            if not live.active_bull:
                return Signal(action="EXIT")

            return None

        # --------------------------------------------------
        # SHORT -> EXIT
        # --------------------------------------------------

        if position.side == Side.SELL:
            # Opposite ReversalTrap signal closes the short.
            if live.bull_trap:
                return Signal(action="EXIT")

            # Mirror ReversalTrap's internal target/stop completion.
            if not live.active_bear:
                return Signal(action="EXIT")

            return None

        return None

    def _get_reversal_trap(self, tf):
        """
        Resolves the ReversalTrap series without depending on one exact UI
        label. The canonical dashboard label is tried first, followed by
        the common aliases used by the engine configuration.
        """
        found = self._get_series(tf, "ReversalTrap", "Reversal Trap")

        if found is not None:
            return found

        for name in ("RT", "ReversalTrap"):
            found = self._get_series(tf, "ReversalTrap", name)

            if found is not None:
                return found

        return None

    def _get_series(self, tf, kind, label):
        """
        Resolves a configured series by kind and label.

        Returns None instead of raising when the series is not part of
        the layout so the strategy degrades gracefully.
        """
        try:
            return tf.get_series(kind, label)
        except KeyError:
            return None