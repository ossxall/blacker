from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side

class Strategy1(Strategy):
    """
    Strategy V3.7 - protección de perdedoras
    EMA Multi-Timeframe + ADX Directional Filter + ADX Momentum Confirmation

    Objetivo:
        Corregir la curva de equity lateral: las ganancias se diluyen
        con pérdidas pequeñas y constantes. Se logra actuando sobre
        dos frentes, ambos basados en ADX:

        1. Entradas 1m con confirmación de momentum ADX.
           Cada cruce de EMA en 1m solo se acepta si el ADX de 1m está
           subiendo, supera un suelo de fuerza y el DMI queda alineado
           con la dirección. Elimina los cruces sin seguimiento real.

        2. Salidas 5m anticipadas por pérdida de fuerza.
           El ADX de 5m en reversal (pico) o un cruce de DI en contra
           con ADX débil cierra la posición antes de devolver las
           ganancias acumuladas.

    5m (régimen de tendencia):
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

    ADX 5m - salidas:
        LONG:
            - tendencia 5m pasa a DOWN
            - ADX 5m en reversal (pico de fuerza)
            - -DI >= +DI con ADX 5m < 23
            - estructura 1m pasa a bajista

        SHORT:
            - tendencia 5m pasa a UP
            - ADX 5m en reversal (pico de fuerza)
            - +DI >= -DI con ADX 5m < 23
            - estructura 1m pasa a alcista

    1m (timing de entrada):
        LONG:
            EMA21 > EMA55
            EMA9 > EMA21 mediante cruce
            EMA21 subiendo
            EMA55 subiendo
            ADX 1m >= 12 y subiendo
            +DI 1m > -DI 1m

        SHORT:
            EMA21 < EMA55
            EMA9 < EMA21 mediante cruce
            EMA21 bajando
            EMA55 bajando
            ADX 1m >= 12 y subiendo
            -DI 1m > +DI 1m

    ADX reversal:
        No abre nuevas operaciones mientras el ADX 5m esté marcando
        reversal. A diferencia de V3.5, aquí sí se usa como salida
        para proteger las ganancias de la tendencia.

    Nota:
        Si la serie "ADX 14" no existe en 1m, el filtro de momentum
        1m se omite y la estrategia conserva el comportamiento de
        V3.5 para no detener el trading.
    """

    ADX_5M_THRESHOLD = 23.0
    ADX_1M_FLOOR = 12.0

    # Protección de pérdidas V3.7
    # Porcentaje máximo de pérdida desde el precio de entrada.
    # Ajustar según el instrumento y el backtest.
    MAX_LOSS_PCT = 0.20

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
        # 5M ADX FILTER
        # ============================================================

        adx_has_strength = value_adx_5m > self.ADX_5M_THRESHOLD

        # Pico de fuerza: el ADX acumuló fuerza y empieza a ceder.
        # Es una señal de salida (no de entrada inversa).
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
        # 1M ADX MOMENTUM CONFIRMATION
        # ============================================================
        # Requiere que el movimiento de 1m tenga fuerza real: ADX por
        # encima de un suelo, subiendo y con el DMI alineado.
        #
        # Si la serie "ADX 14" no está configurada en 1m, el filtro se
        # omite y se conserva el comportamiento de V3.5.

        momentum_long_1m = True
        momentum_short_1m = True

        adx_1m = self._get_series(tf1, "ADX", "ADX 14")
        previous_adx_1m = self._last_closed(adx_1m)

        if (
            adx_1m is not None
            and adx_1m.live is not None
            and previous_adx_1m is not None
        ):

            value_adx_1m = adx_1m.live.adx
            plus_di_1m = getattr(adx_1m.live, "plus_di", None)
            minus_di_1m = getattr(adx_1m.live, "minus_di", None)

            if plus_di_1m is not None and minus_di_1m is not None:

                adx_1m_has_strength = (
                    value_adx_1m >= self.ADX_1M_FLOOR
                )

                adx_1m_rising = value_adx_1m > previous_adx_1m.adx

                momentum_long_1m = (
                    adx_1m_has_strength
                    and adx_1m_rising
                    and plus_di_1m > minus_di_1m
                )

                momentum_short_1m = (
                    adx_1m_has_strength
                    and adx_1m_rising
                    and minus_di_1m > plus_di_1m
                )

        # ============================================================
        # POSITION
        # ============================================================

        portfolio = state.portfolio

        position = (
            portfolio.position
            if portfolio is not None
            else None
        )

        # Precio actual: se obtiene de forma tolerante porque el EngineState
        # puede exponerlo con distintos nombres según la implementación.
        current_price = getattr(state, "current_price", None)
        if current_price is None:
            current_price = getattr(state, "price", None)
        if current_price is None:
            current_price = getattr(tf1, "current_price", None)
        if current_price is None:
            current_price = getattr(tf1, "price", None)

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
                    and momentum_long_1m
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
                    and momentum_short_1m
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

            # ============================================================
            # HARD STOP DE PÉRDIDA
            # ============================================================
            # Corta una operación que se mueve inmediatamente en contra,
            # independientemente de las demás condiciones de salida.
            entry_price = getattr(position, "entry_price", None)

            if current_price is not None and entry_price is not None:
                loss_pct = (
                    (current_price - entry_price)
                    / entry_price
                    * 100.0
                )

                if loss_pct <= -self.MAX_LOSS_PCT:
                    return Signal(action="EXIT")

            # Cambio de tendencia principal
            if trend_5m == "DOWN":
                return Signal(action="EXIT")

            # Pérdida de fuerza 5m: el ADX hizo pico (reversal) o el
            # DMI se giró en contra con ADX débil. Protege las
            # ganancias acumuladas en lugar de devolverlas.
            adx_strength_exit = (
                adx_reversal
                or (
                    minus_di >= plus_di
                    and not adx_has_strength
                )
            )

            if adx_strength_exit:
                return Signal(action="EXIT")

            # Pérdida de estructura 1m.
            # V3.7 sale en cuanto EMA21 pierde EMA55; no espera además
            # a que EMA9 confirme la segunda condición.
            if value21 < value55:
                return Signal(action="EXIT")

            return None

        # ============================================================
        # SHORT EXIT
        # ============================================================

        if position.side == Side.SELL:

            # ============================================================
            # HARD STOP DE PÉRDIDA
            # ============================================================
            entry_price = getattr(position, "entry_price", None)

            if current_price is not None and entry_price is not None:
                loss_pct = (
                    (entry_price - current_price)
                    / entry_price
                    * 100.0
                )

                if loss_pct <= -self.MAX_LOSS_PCT:
                    return Signal(action="EXIT")

            # Cambio de tendencia principal
            if trend_5m == "UP":
                return Signal(action="EXIT")

            # Pérdida de fuerza 5m (espejo del LONG)
            adx_strength_exit = (
                adx_reversal
                or (
                    plus_di >= minus_di
                    and not adx_has_strength
                )
            )

            if adx_strength_exit:
                return Signal(action="EXIT")

            # Pérdida de estructura 1m.
            # V3.7 sale en cuanto EMA21 supera EMA55; no espera además
            # a que EMA9 confirme la segunda condición.
            if value21 > value55:
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

    def _last_closed(self, series):

        history = getattr(series, "history", None)

        if not history:
            return None

        return history[-1]

    def _get_series(self, tf, kind, label):

        try:
            return tf.get_series(kind, label)

        except KeyError:
            return None
