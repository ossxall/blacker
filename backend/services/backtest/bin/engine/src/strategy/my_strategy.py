from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Strategy V3.4
    EMA Multi-Timeframe + ADX Regime + ADX Reversal Confirmation

    TIMEFRAME 5m
    ----------------
    Tendencia LONG:
        EMA55 > EMA200
        EMA55 subiendo
        ADX > 20

    Tendencia SHORT:
        EMA55 < EMA200
        EMA55 bajando
        ADX > 20

    ADX REVERSAL
    ----------------
    is_reversal NO provoca salida automáticamente.

    Cuando ADX está en reversal, la entrada exige confirmación
    adicional mediante:
        LONG:
            +DI > -DI
            EMA9 subiendo
            EMA21 subiendo
            EMA9 > EMA21 > EMA55
            cruce alcista EMA9/EMA21

        SHORT:
            -DI > +DI
            EMA9 bajando
            EMA21 bajando
            EMA9 < EMA21 < EMA55
            cruce bajista EMA9/EMA21

    TIMEFRAME 1m
    ----------------
    Entrada normal LONG:
        EMA21 > EMA55
        cruce EMA9 sobre EMA21

    Entrada normal SHORT:
        EMA21 < EMA55
        cruce EMA9 bajo EMA21

    SALIDAS
    ----------------
    LONG:
        - tendencia 5m pasa a DOWN
        - EMA21 < EMA55 y EMA9 < EMA21

    SHORT:
        - tendencia 5m pasa a UP
        - EMA21 > EMA55 y EMA9 > EMA21
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

        if ema55_5m is None:
            return None

        if ema200_5m is None:
            return None

        if adx_5m is None:
            return None

        if ema55_5m.live is None:
            return None

        if ema200_5m.live is None:
            return None

        if adx_5m.live is None:
            return None

        # ============================================================
        # 5M VALUES
        # ============================================================

        value55_5m = ema55_5m.live.value
        value200_5m = ema200_5m.live.value

        # ADX usa .adx, NO .value
        adx_live = adx_5m.live
        value_adx_5m = adx_live.adx

        adx_threshold = 20.0

        adx_has_strength = value_adx_5m > adx_threshold

        # ============================================================
        # ADX REVERSAL
        # ============================================================

        adx_reversal = bool(
            getattr(adx_live, "is_reversal", False)
        )

        # DI direction
        plus_di = getattr(adx_live, "plus_di", None)
        minus_di = getattr(adx_live, "minus_di", None)

        if plus_di is None or minus_di is None:
            return None

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
        # 5M TREND FILTER
        # ============================================================

        long_trend = (
            trend_5m == "UP"
            and ema55_rising
            and adx_has_strength
        )

        short_trend = (
            trend_5m == "DOWN"
            and ema55_falling
            and adx_has_strength
        )

        # ============================================================
        # 1M INDICATORS
        # ============================================================

        ema9 = self._get_series(tf1, "EMA", "EMA 9")
        ema21 = self._get_series(tf1, "EMA", "EMA 21")
        ema55 = self._get_series(tf1, "EMA", "EMA 55")

        if ema9 is None or ema21 is None or ema55 is None:
            return None

        if ema9.live is None:
            return None

        if ema21.live is None:
            return None

        if ema55.live is None:
            return None

        # ============================================================
        # 1M VALUES
        # ============================================================

        value9 = ema9.live.value
        value21 = ema21.live.value
        value55 = ema55.live.value

        # ============================================================
        # PREVIOUS CLOSED 1M VALUES
        # ============================================================

        previous9 = self._previous_closed(ema9)
        previous21 = self._previous_closed(ema21)

        if previous9 is None or previous21 is None:
            return None

        # ============================================================
        # 1M CROSSES
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

        bullish_structure = (
            value21 > value55
        )

        bearish_structure = (
            value21 < value55
        )

        # Full structure
        bullish_full_structure = (
            value9 > value21
            and value21 > value55
        )

        bearish_full_structure = (
            value9 < value21
            and value21 < value55
        )

        # ============================================================
        # 1M EMA SLOPES
        # ============================================================

        ema9_rising = value9 > previous9
        ema9_falling = value9 < previous9

        ema21_rising = value21 > previous21
        ema21_falling = value21 < previous21

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
            # LONG
            # --------------------------------------------------------

            if long_trend:

                # ADX normal:
                # Solo necesitamos fuerza ADX + estructura + cruce.
                if not adx_reversal:

                    if (
                        bullish_structure
                        and bullish_cross
                    ):
                        return Signal(
                            action="BUY",
                            quantity=1
                        )

                # ADX reversal:
                # No bloqueamos automáticamente la entrada.
                # Exigimos confirmación adicional.
                else:

                    if (
                        bullish_cross
                        and bullish_full_structure
                        and di_bullish
                        and ema9_rising
                        and ema21_rising
                    ):
                        return Signal(
                            action="BUY",
                            quantity=1
                        )

            # --------------------------------------------------------
            # SHORT
            # --------------------------------------------------------

            if short_trend:

                # ADX normal
                if not adx_reversal:

                    if (
                        bearish_structure
                        and bearish_cross
                    ):
                        return Signal(
                            action="SELL",
                            quantity=1
                        )

                # ADX reversal:
                # Exigimos confirmación adicional.
                else:

                    if (
                        bearish_cross
                        and bearish_full_structure
                        and di_bearish
                        and ema9_falling
                        and ema21_falling
                    ):
                        return Signal(
                            action="SELL",
                            quantity=1
                        )

            return None

        # ============================================================
        # LONG POSITION EXIT
        # ============================================================

        if position.side == Side.BUY:

            # 5m trend reversal
            if trend_5m == "DOWN":
                return Signal(action="EXIT")

            # 1m bearish structure
            if (
                value21 < value55
                and value9 < value21
            ):
                return Signal(action="EXIT")

            return None

        # ============================================================
        # SHORT POSITION EXIT
        # ============================================================

        if position.side == Side.SELL:

            # 5m trend reversal
            if trend_5m == "UP":
                return Signal(action="EXIT")

            # 1m bullish structure
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