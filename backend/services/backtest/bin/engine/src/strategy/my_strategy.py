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
    Cruce de EMAs: la estrategia mas simple que se puede escribir sobre
    este motor. No busca alfa, sirve de ejemplo y de banco de pruebas del
    ciclo de ordenes.

    La regla entera es una sola linea: la media rapida por encima de la
    lenta es alcista, por debajo es bajista, y la posicion se abre o se
    cierra en consecuencia.

        - Plana y la rapida encima de la lenta  -> BUY
        - Plana y la rapida debajo de la lenta   -> SELL
        - Long y la estructura pasa a bajista   -> EXIT
        - Short y la estructura pasa a alcista  -> EXIT

    Lo unico que lee es ``state``: no lleva contador de barras, ni
    cooldown, ni banderas por lado. Por eso no necesita ``set_state`` ni
    ``to_dict`` propios -- el estado que el motor restaura es exactamente
    el que hay en ``EngineState``, y dos backtests sobre el mismo feed
    dan el mismo resultado.

    El stop-loss, los objetivos y el trailing no se tocan aqui: los
    coloca el RiskManager a partir de la configuracion de riesgo cuando
    la entrada llena. Esta clase solo emite intenciones.
    """

    DEFAULT_PARAMS = {
        # Timeframe del que se leen las medias.
        "timeframe": "5m",
        # Las etiquetas deben coincidir con las de la configuracion.
        "label_fast": "EMA 20",
        "label_slow": "EMA 50",
        "quantity": 1,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.timeframe = str(merged["timeframe"])
        self.label_fast = str(merged["label_fast"])
        self.label_slow = str(merged["label_slow"])

        # El OrderManager rechaza una cantidad no positiva, asi que una
        # configuracion equivocada cae a 1 en vez de tumbar el motor.
        try:
            quantity = float(merged["quantity"])
        except (TypeError, ValueError):
            quantity = 1.0

        self.quantity = quantity if quantity > 0.0 else 1.0

    def evaluate(self, state: EngineState):
        # ---------------------------------------------------------
        # SERIES
        # ---------------------------------------------------------
        timeframe = state.timeframes.get(self.timeframe)

        if timeframe is None:
            return None

        fast = self._get_series(timeframe, "EMA", self.label_fast)
        slow = self._get_series(timeframe, "EMA", self.label_slow)

        if fast is None or slow is None:
            return None

        if fast.live is None or slow.live is None:
            return None

        fast_value = fast.live.value
        slow_value = slow.live.value

        if fast_value is None or slow_value is None:
            return None

        bullish = fast_value > slow_value
        bearish = fast_value < slow_value

        # ---------------------------------------------------------
        # POSICION
        #
        # El portfolio es la unica fuente de verdad: si una entrada esta
        # pendiente todavia no hay posicion, y aqui no se lleva ninguna
        # cuenta propia que pueda desincronizarse de el.
        # ---------------------------------------------------------
        portfolio = state.portfolio
        position = portfolio.position if portfolio is not None else None

        if position is None:
            if bullish:
                return Signal(action="BUY", quantity=self.quantity)

            if bearish:
                return Signal(action="SELL", quantity=self.quantity)

            return None

        if position.side == Side.BUY and bearish:
            return Signal(action="EXIT")

        if position.side == Side.SELL and bullish:
            return Signal(action="EXIT")

        return None

    # =============================================================
    # HELPERS
    # =============================================================

    @staticmethod
    def _unwrap(params: dict) -> dict:
        out = {}

        for key, value in (params or {}).items():

            if isinstance(value, dict):
                if "value" in value:
                    value = value["value"]
                else:
                    continue

            out[key] = value

        return out

    @staticmethod
    def _get_series(timeframe, kind: str, label: str):
        if timeframe is None:
            return None

        try:
            return timeframe.get_series(kind, label)
        except KeyError:
            return None
