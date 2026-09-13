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

from aggregator.bar_aggregator import BarAggregator
from strategy.base import Strategy
from core.engine_state import EngineState
from core.portfolio import Portfolio
from strategy.registry import STRATEGY_REGISTRY
from ingestion.tick import Tick
from timeframes.timeframe import Timeframe
from series.registry import SERIES_REGISTRY
from orders.order_manager import OrderManager
from orders.risk_manager import RiskManager
from execution.execution import Execution

class TradingEngine:
    def __init__(self):
        self.status: str = 'init'
        self.boot_id: str | None = None
        self.config_id: str | None = None
        self.state: EngineState | None = None
        self.timeframes: dict[str, Timeframe] = {}
        self.bar_aggregator: BarAggregator | None = None
        self.strategy: Strategy | None = None

        self.portfolio = Portfolio()
        self.risk = {}
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager(
            portfolio=self.portfolio,
            risk_manager=self.risk_manager,
        )
        self.execution = Execution(self.order_manager)

    def reset(self):
        print("Reseting engine...")

        self.status = "init"
        self.boot_id = None
        self.config_id = None
        self.state = None
        self.timeframes = {}
        self.bar_aggregator = None
        self.strategy = None

        self.portfolio = Portfolio()
        self.risk = {}
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager(
            portfolio=self.portfolio,
            risk_manager=self.risk_manager,
        )
        self.execution = Execution(self.order_manager)

    def set_state(self, boot_id: str, config_id: str, engine_state: dict) -> None:
        """
        Restores the engine state from a serialized dictionary.
        """
        self.boot_id = boot_id
        self.config_id = config_id

        timeframes = {}
        #
        # Setup timeframes and series
        #
        for tf_value in engine_state["timeframes"].values():
            timeframe = Timeframe()
            timeframe.set_state(tf_value)
            #
            # Add Timeframe series
            #
            for state in tf_value["series"].values():

                series = SERIES_REGISTRY[state["kind"]](
                    state["id"],
                    state["kind"],
                    state["level"],
                    state["primary"],
                    state["overlay"],
                    state["params"],
                )
                series.set_state(state)

                timeframe.add_series(series)
            #
            # Build the series execution level
            #                         
            timeframe.build_levels()
            timeframes[timeframe.id] = timeframe
        #
        # Build engine strategy
        #
        strategy = STRATEGY_REGISTRY[engine_state["strategy"]["kind"]](
            engine_state["strategy"]["kind"],
            engine_state["strategy"]["params"]        
        )
        strategy.set_state(engine_state["strategy"])

        self.timeframes = timeframes
        self.bar_aggregator = BarAggregator(
            timeframes=self.timeframes
        )
        self.strategy = strategy
        #
        # Restore portfolio and order pipeline
        #
        self.risk = engine_state.get("risk", {})

        self.portfolio = Portfolio.from_dict(
            engine_state.get("portfolio")
        )
        self.risk_manager = RiskManager(
            self.risk
        )
        self.order_manager = OrderManager.from_dict(
            engine_state.get("orders"),
            portfolio=self.portfolio,
            risk_manager=self.risk_manager,
        )
        self.execution = Execution(self.order_manager)
        #
        # Engine state
        #
        self.state = EngineState(
            boot_id=self.boot_id,
            config_id=self.config_id,
            tick_index=engine_state["tick_index"],
            time=engine_state["time"], 
            timeframes=self.timeframes,
            strategy=self.strategy,
            portfolio=self.portfolio,
            risk=self.risk,
            orders=self.order_manager.to_dict(),
        )

    def on_tick(self, tick: Tick):
        self.bar_aggregator.update(tick)

        for timeframe in self.timeframes.values():
            timeframe.update()

        self.state = EngineState(
            boot_id=self.boot_id,
            config_id=self.config_id,
            tick_index=tick.tick_index,
            time=tick.time,
            timeframes=self.timeframes, 
            strategy=self.strategy,
            portfolio=self.portfolio,
            risk=self.risk,
            orders=self.order_manager.to_dict(),
        )

        # --------------------------------------------------
        # 1. Resolve protective orders first.
        #    Stops and targets are matched before the strategy
        #    sees the updated state, so it never acts on a
        #    position that was already closed by a bracket.
        # --------------------------------------------------

        fills = self.execution.update(self.state, tick)

        # --------------------------------------------------
        # 2. Evaluate the strategy on the consistent state.
        # --------------------------------------------------

        signal = self.strategy.evaluate(self.state)

        # --------------------------------------------------
        # 3. Translate the alpha intent into orders and queue
        #    them for the next tick.
        # --------------------------------------------------

        orders = self.order_manager.handle(signal)

        self.execution.submit(orders)

        # --------------------------------------------------
        # 4. The published state must reflect everything this
        #    tick has done (fills, position, order book).
        # --------------------------------------------------

        self.state.orders = self.order_manager.to_dict()

        return self.state, signal