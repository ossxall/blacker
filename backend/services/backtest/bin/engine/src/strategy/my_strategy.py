from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    EMA Multi-Timeframe V3

    5m
    --
    EMA 55 > EMA 200 -> tendencia alcista
    EMA 55 < EMA 200 -> tendencia bajista

    1m
    --
    EMA 21 > EMA 55 -> estructura alcista
    EMA 21 < EMA 55 -> estructura bajista

    ENTRADA LONG
    ------------
    5m alcista
    +
    1m EMA21 > EMA55
    +
    EMA9 cruza EMA21 al alza sobre vela cerrada

    ENTRADA SHORT
    -------------
    5m bajista
    +
    1m EMA21 < EMA55
    +
    EMA9 cruza EMA21 a la baja sobre vela cerrada

    SALIDA LONG
    -----------
    5m cambia a bajista
    OR
    1m EMA21 cruza por debajo de EMA55

    SALIDA SHORT
    ------------
    5m cambia a alcista
    OR
    1m EMA21 cruza por encima de EMA55

    No utiliza __init__ ni estado propio.
    """

    def evaluate(self, state: EngineState):

        tf5 = state.timeframes.get("5m")
        tf1 = state.timeframes.get("1m")

        if tf5 is None or tf1 is None:
            return None

        # ==================================================
        # 5m
        # ==================================================

        ema55_5m = self._get_series(
            tf5,
            "EMA",
            "EMA 55",
        )

        ema200_5m = self._get_series(
            tf5,
            "EMA",
            "EMA 200",
        )

        if ema55_5m is None or ema200_5m is None:
            return None

        if (
            ema55_5m.live is None
            or ema200_5m.live is None
        ):
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

        # ==================================================
        # 1m
        # ==================================================

        ema9 = self._get_series(
            tf1,
            "EMA",
            "EMA 9",
        )

        ema21 = self._get_series(
            tf1,
            "EMA",
            "EMA 21",
        )

        ema55 = self._get_series(
            tf1,
            "EMA",
            "EMA 55",
        )

        if (
            ema9 is None
            or ema21 is None
            or ema55 is None
        ):
            return None

        if (
            ema9.live is None
            or ema21.live is None
            or ema55.live is None
        ):
            return None

        # ==================================================
        # VALORES LIVE
        # ==================================================

        value9 = ema9.live.value
        value21 = ema21.live.value
        value55 = ema55.live.value

        # ==================================================
        # HISTORIAL CERRADO
        # ==================================================

        previous9 = self._previous_closed(ema9)
        previous21 = self._previous_closed(ema21)

        previous21_older = self._older_closed(ema21)
        previous55_older = self._older_closed(ema55)

        if previous9 is None or previous21 is None:
            return None

        # ==================================================
        # CRUCE EMA 9 / EMA 21
        # ==================================================

        bullish_cross = (
            previous9 <= previous21
            and value9 > value21
        )

        bearish_cross = (
            previous9 >= previous21
            and value9 < value21
        )

        # ==================================================
        # ESTRUCTURA 1m
        # ==================================================

        bullish_structure = (
            value21 > value55
        )

        bearish_structure = (
            value21 < value55
        )

        # ==================================================
        # PORTFOLIO
        # ==================================================

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

            # ----------------------------------------------
            # LONG
            # ----------------------------------------------

            if (
                trend_5m == "UP"
                and bullish_structure
                and bullish_cross
            ):
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            # ----------------------------------------------
            # SHORT
            # ----------------------------------------------

            if (
                trend_5m == "DOWN"
                and bearish_structure
                and bearish_cross
            ):
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # ==================================================
        # LONG -> EXIT
        # ==================================================

        if position.side == Side.BUY:

            # Tendencia 5m invertida
            if trend_5m == "DOWN":
                return Signal(action="EXIT")

            # Estructura 1m completamente deteriorada
            if (
                value21 < value55
                and value9 < value21
            ):
                return Signal(action="EXIT")

            return None

        # ==================================================
        # SHORT -> EXIT
        # ==================================================

        if position.side == Side.SELL:

            # Tendencia 5m invertida
            if trend_5m == "UP":
                return Signal(action="EXIT")

            # Estructura 1m completamente invertida
            if (
                value21 > value55
                and value9 > value21
            ):
                return Signal(action="EXIT")

            return None

        return None

    # ======================================================
    # HISTORIAL
    # ======================================================

    def _previous_closed(self, series):
        """
        Último valor de EMA correspondiente a una vela cerrada.
        """

        history = getattr(series, "history", None)

        if not history:
            return None

        return history[-1].value

    def _older_closed(self, series):
        """
        Segundo último valor cerrado.
        """

        history = getattr(series, "history", None)

        if not history or len(history) < 2:
            return None

        return history[-2].value

    # ======================================================
    # SERIES
    # ======================================================

    def _get_series(self, tf, kind, label):
        try:
            return tf.get_series(kind, label)

        except KeyError:
            return None