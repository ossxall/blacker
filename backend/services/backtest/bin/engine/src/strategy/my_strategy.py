from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    EMA Multi-Timeframe V3.1

    MEJORA SOBRE V3:
    ----------------
    Se mantiene intacta la lógica de entrada de V3.

    Único filtro nuevo:
        LONG:
            EMA55 5m > EMA200
            +
            EMA55 5m subiendo

        SHORT:
            EMA55 5m < EMA200
            +
            EMA55 5m bajando

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

        # ==================================================
        # Tendencia 5m
        # ==================================================

        if value55_5m > value200_5m:
            trend_5m = "UP"

        elif value55_5m < value200_5m:
            trend_5m = "DOWN"

        else:
            return None

        # ==================================================
        # Pendiente EMA55 5m
        # ==================================================

        previous55_5m = self._previous_closed(
            ema55_5m
        )

        if previous55_5m is None:
            return None

        ema55_rising = (
            value55_5m > previous55_5m
        )

        ema55_falling = (
            value55_5m < previous55_5m
        )

        # Filtro de tendencia mejorada
        long_trend = (
            trend_5m == "UP"
            and ema55_rising
        )

        short_trend = (
            trend_5m == "DOWN"
            and ema55_falling
        )

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
        # Valores LIVE
        # ==================================================

        value9 = ema9.live.value
        value21 = ema21.live.value
        value55 = ema55.live.value

        # ==================================================
        # Historial cerrado
        # ==================================================

        previous9 = self._previous_closed(ema9)
        previous21 = self._previous_closed(ema21)

        if (
            previous9 is None
            or previous21 is None
        ):
            return None

        # ==================================================
        # Cruce EMA9 / EMA21
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
        # Estructura 1m
        # ==================================================

        bullish_structure = (
            value21 > value55
        )

        bearish_structure = (
            value21 < value55
        )

        # ==================================================
        # Portfolio
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

            # LONG
            if (
                long_trend
                and bullish_structure
                and bullish_cross
            ):
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            # SHORT
            if (
                short_trend
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

            # Mantener el exit original de V3
            if trend_5m == "DOWN":
                return Signal(
                    action="EXIT"
                )

            if (
                value21 < value55
                and value9 < value21
            ):
                return Signal(
                    action="EXIT"
                )

            return None

        # ==================================================
        # SHORT -> EXIT
        # ==================================================

        if position.side == Side.SELL:

            # Mantener el exit original de V3
            if trend_5m == "UP":
                return Signal(
                    action="EXIT"
                )

            if (
                value21 > value55
                and value9 > value21
            ):
                return Signal(
                    action="EXIT"
                )

            return None

        return None

    # ======================================================
    # HISTORIAL
    # ======================================================

    def _previous_closed(self, series):
        """
        Último valor de EMA correspondiente
        a una vela cerrada.
        """

        history = getattr(
            series,
            "history",
            None,
        )

        if not history:
            return None

        return history[-1].value

    # ======================================================
    # SERIES
    # ======================================================

    def _get_series(self, tf, kind, label):
        try:
            return tf.get_series(
                kind,
                label,
            )

        except KeyError:
            return None