from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Estrategia EMA Multi-Timeframe: 5m + 1m

    TIMEFRAME 5m
    ------------
    Define la tendencia principal:

        EMA 55 > EMA 200 -> tendencia alcista
        EMA 55 < EMA 200 -> tendencia bajista

    TIMEFRAME 1m
    ------------
    Busca confirmación de entrada:

    LONG:
        EMA 9 > EMA 21 > EMA 55

    SHORT:
        EMA 9 < EMA 21 < EMA 55

    ENTRADAS
    --------
    LONG:
        5m alcista + 1m con estructura alcista

    SHORT:
        5m bajista + 1m con estructura bajista

    SALIDAS
    -------
    LONG:
        5m pierde tendencia alcista
        OR
        EMA 9 < EMA 21 en 1m

    SHORT:
        5m pierde tendencia bajista
        OR
        EMA 9 > EMA 21 en 1m
    """

    def evaluate(self, state: EngineState):

        tf5 = state.timeframes.get("5m")
        tf1 = state.timeframes.get("1m")

        if tf5 is None or tf1 is None:
            return None

        # --------------------------------------------------
        # EMA 5m
        # --------------------------------------------------

        ema55_5m = self._get_series(tf5, "EMA", "EMA 55")
        ema200_5m = self._get_series(tf5, "EMA", "EMA 200")

        if ema55_5m is None or ema200_5m is None:
            return None

        if ema55_5m.live is None or ema200_5m.live is None:
            return None

        value55_5m = ema55_5m.live.value
        value200_5m = ema200_5m.live.value

        # Tendencia principal
        if value55_5m > value200_5m:
            trend_5m = "UP"

        elif value55_5m < value200_5m:
            trend_5m = "DOWN"

        else:
            return None

        # --------------------------------------------------
        # EMA 1m
        # --------------------------------------------------

        ema9_1m = self._get_series(tf1, "EMA", "EMA 9")
        ema21_1m = self._get_series(tf1, "EMA", "EMA 21")
        ema55_1m = self._get_series(tf1, "EMA", "EMA 55")

        if (
            ema9_1m is None
            or ema21_1m is None
            or ema55_1m is None
        ):
            return None

        if (
            ema9_1m.live is None
            or ema21_1m.live is None
            or ema55_1m.live is None
        ):
            return None

        value9_1m = ema9_1m.live.value
        value21_1m = ema21_1m.live.value
        value55_1m = ema55_1m.live.value

        # --------------------------------------------------
        # Estructura 1m
        # --------------------------------------------------

        bullish_1m = (
            value9_1m > value21_1m
            and value21_1m > value55_1m
        )

        bearish_1m = (
            value9_1m < value21_1m
            and value21_1m < value55_1m
        )

        # --------------------------------------------------
        # POSICIÓN
        # --------------------------------------------------

        portfolio = state.portfolio
        position = (
            portfolio.position
            if portfolio is not None
            else None
        )

        # ==================================================
        # FLAT -> ENTRY
        # ==================================================

        if position is None:

            # -----------------------------
            # LONG
            # -----------------------------

            if trend_5m == "UP" and bullish_1m:
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            # -----------------------------
            # SHORT
            # -----------------------------

            if trend_5m == "DOWN" and bearish_1m:
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # ==================================================
        # LONG -> EXIT
        # ==================================================

        if position.side == Side.BUY:

            # La tendencia principal cambió
            if trend_5m == "DOWN":
                return Signal(action="EXIT")

            # Se pierde estructura rápida 1m
            if value9_1m < value21_1m:
                return Signal(action="EXIT")

            return None

        # ==================================================
        # SHORT -> EXIT
        # ==================================================

        if position.side == Side.SELL:

            # La tendencia principal cambió
            if trend_5m == "UP":
                return Signal(action="EXIT")

            # Se pierde estructura rápida 1m
            if value9_1m > value21_1m:
                return Signal(action="EXIT")

            return None

        return None

    # ======================================================
    # HELPERS
    # ======================================================

    def _get_series(self, tf, kind, label):
        """
        Obtiene una serie configurada sin provocar
        una excepción si no existe.
        """
        try:
            return tf.get_series(kind, label)
        except KeyError:
            return None