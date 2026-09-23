from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Strategy V3.5
    EMA Multi-Timeframe + ADX Directional Filter + Selective Entries

    Objetivo:
        Reducir las entradas de baja calidad que generan muchas pérdidas
        pequeñas y dejan la curva de equity lateral.

    5m:
        LONG:
            EMA55 > EMA200
            EMA55 subiendo
            ADX > 23
            +DI > -DI

        SHORT:
            EMA55 < EMA200
            EMA55 bajando
            ADX > 23
            -DI > +DI

    1m:
        LONG:
            EMA21 > EMA55
            EMA9 > EMA21 mediante cruce
            EMA21 subiendo
            EMA55 subiendo

        SHORT:
            EMA21 < EMA55
            EMA9 < EMA21 mediante cruce
            EMA21 bajando
            EMA55 bajando

    ADX reversal:
        No abre nuevas operaciones mientras el ADX esté marcando
        reversal. No se usa como salida.

    SALIDAS:
        LONG:
            - tendencia 5m pasa a DOWN
            - estructura 1m pasa a bajista

        SHORT:
            - tendencia 5m pasa a UP
            - estructura 1m pasa a alcista

    Nota:
        El objetivo de V3.5 es calidad de entrada, no maximizar el
        número de operaciones. El resultado debe validarse mediante
        backtest con el mismo periodo y costes.
    """

    def evaluate(self, state: EngineState):

        # ============================================================
        # TIMEFRAMES
        # ============================================================

        tf5 = state.timeframes.get("5m")
        tf1 = state.timeframes.get("1m")

        if tf5 is None or tf1 is None:
            return None

        # ============================================================
        # 5M INDICATORS
        # ============================================================

        ema55_5m = self._get_series(tf5, "EMA", "EMA 55")
        ema200_5m = self._get_series(tf5, "EMA", "EMA 200")
        adx_5m = self._get_series(tf5, "ADX", "ADX 14")

        if ema55_5m is None or ema200_5m is None or adx_5m is None:
            return None

        if (
            ema55_5m.live is None
            or ema200_5m.live is None
            or adx_5m.live is None
        ):
            return None

        value55_5m = ema55_5m.live.value
        value200_5m = ema200_5m.live.value

        # ADX personalizado: el valor está en .adx
        adx_live = adx_5m.live
        value_adx_5m = adx_live.adx

        plus_di = getattr(adx_live, "plus_di", None)
        minus_di = getattr(adx_live, "minus_di", None)

        if plus_di is None or minus_di is None:
            return None

        # ============================================================
        # ADX FILTER
        # ============================================================

        # 23 coincide con el key_level por defecto del ADX personalizado
        # y evita aceptar demasiadas fases de tendencia débil.
        adx_threshold = 23.0

        adx_has_strength = value_adx_5m > adx_threshold

        adx_reversal = bool(
            getattr(adx_live, "is_reversal", False)
        )

        di_bullish = plus_di > minus_di
        di_bearish = minus_di > plus_di

        # ============================================================
        # 5M TREND
        # ============================================================

        if value55_5m > value200_5m:
            trend_5m = "UP"
        elif value55_5m < value200_5m:
            trend_5m = "DOWN"
        else:
            return None

        # ============================================================
        # 5M EMA55 SLOPE
        # ============================================================

        previous55_5m = self._previous_closed(ema55_5m)

        if previous55_5m is None:
            return None

        ema55_rising = value55_5m > previous55_5m
        ema55_falling = value55_5m < previous55_5m

        # ============================================================
        # 5M TREND + DIRECTIONAL ADX
        # ============================================================

        long_trend = (
            trend_5m == "UP"
            and ema55_rising
            and adx_has_strength
            and di_bullish
        )

        short_trend = (
            trend_5m == "DOWN"
            and ema55_falling
            and adx_has_strength
            and di_bearish
        )

        # ============================================================
        # 1M INDICATORS
        # ============================================================

        ema9 = self._get_series(tf1, "EMA", "EMA 9")
        ema21 = self._get_series(tf1, "EMA", "EMA 21")
        ema55 = self._get_series(tf1, "EMA", "EMA 55")

        if ema9 is None or ema21 is None or ema55 is None:
            return None

        if (
            ema9.live is None
            or ema21.live is None
            or ema55.live is None
        ):
            return None

        value9 = ema9.live.value
        value21 = ema21.live.value
        value55 = ema55.live.value

        # ============================================================
        # PREVIOUS CLOSED 1M VALUES
        # ============================================================

        previous9 = self._previous_closed(ema9)
        previous21 = self._previous_closed(ema21)
        previous55 = self._previous_closed(ema55)

        if (
            previous9 is None
            or previous21 is None
            or previous55 is None
        ):
            return None

        # ============================================================
        # CROSS
        # ============================================================

        bullish_cross = (
            previous9 <= previous21
            and value9 > value21
        )

        bearish_cross = (
            previous9 >= previous21
            and value9 < value21
        )

        # ============================================================
        # 1M STRUCTURE
        # ============================================================

        bullish_structure = value21 > value55
        bearish_structure = value21 < value55

        # ============================================================
        # 1M SLOPES
        # ============================================================

        ema21_rising = value21 > previous21
        ema21_falling = value21 < previous21

        ema55_rising = value55 > previous55
        ema55_falling = value55 < previous55

        # ============================================================
        # POSITION
        # ============================================================

        portfolio = state.portfolio

        position = (
            portfolio.position
            if portfolio is not None
            else None
        )

        # ============================================================
        # ENTRY
        # ============================================================

        if position is None:

            # --------------------------------------------------------
            # ADX REVERSAL FILTER
            # --------------------------------------------------------
            # Un reversal del ADX significa pérdida de fuerza.
            # No lo interpretamos como cambio de tendencia, pero sí
            # evitamos iniciar una nueva operación en ese momento.

            if adx_reversal:
                return None

            # --------------------------------------------------------
            # LONG
            # --------------------------------------------------------

            if long_trend:

                long_entry = (
                    bullish_cross
                    and bullish_structure
                    and ema21_rising
                    and ema55_rising
                )

                if long_entry:
                    return Signal(
                        action="BUY",
                        quantity=1
                    )

            # --------------------------------------------------------
            # SHORT
            # --------------------------------------------------------

            if short_trend:

                short_entry = (
                    bearish_cross
                    and bearish_structure
                    and ema21_falling
                    and ema55_falling
                )

                if short_entry:
                    return Signal(
                        action="SELL",
                        quantity=1
                    )

            return None

        # ============================================================
        # LONG EXIT
        # ============================================================

        if position.side == Side.BUY:

            # Cambio de tendencia principal
            if trend_5m == "DOWN":
                return Signal(action="EXIT")

            # Pérdida de estructura 1m
            if (
                value21 < value55
                and value9 < value21
            ):
                return Signal(action="EXIT")

            return None

        # ============================================================
        # SHORT EXIT
        # ============================================================

        if position.side == Side.SELL:

            # Cambio de tendencia principal
            if trend_5m == "UP":
                return Signal(action="EXIT")

            # Pérdida de estructura 1m
            if (
                value21 > value55
                and value9 > value21
            ):
                return Signal(action="EXIT")

            return None

        return None

    # ================================================================
    # HELPERS
    # ================================================================

    def _previous_closed(self, series):

        history = getattr(series, "history", None)

        if not history:
            return None

        return history[-1].value

    def _get_series(self, tf, kind, label):

        try:
            return tf.get_series(kind, label)

        except KeyError:
            return None