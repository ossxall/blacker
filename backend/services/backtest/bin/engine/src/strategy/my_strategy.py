from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Strategy V3.8 - Equity Curve Protection
    
    Mejoras aplicadas basadas en backtest:
    1. Filtro Anti-FOMO: Bloquea entradas si ADX 5m > 45 (movimiento exhausto).
    2. Emergency Stop 1m: Corta pérdidas masivas al instante si la EMA9 
       rompe la EMA55 en contra (corte de flash-crashes/spikes).
    3. Dynamic Trailing Profit: Si el ADX de 5m hace reversal, sale con 
       el primer cruce rápido en contra (EMA9 vs EMA21) para asegurar PnL.
    """

    ADX_5M_THRESHOLD = 23.0
    ADX_5M_MAX = 45.0      # Límite superior para evitar comprar/vender tarde
    ADX_1M_FLOOR = 15.0    # Suelo elevado para exigir mayor momentum inicial

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

        if not all([ema55_5m, ema200_5m, adx_5m]) or not all([ema55_5m.live, ema200_5m.live, adx_5m.live]):
            return None

        value55_5m = ema55_5m.live.value
        value200_5m = ema200_5m.live.value

        adx_live = adx_5m.live
        value_adx_5m = adx_live.adx

        plus_di = getattr(adx_live, "plus_di", None)
        minus_di = getattr(adx_live, "minus_di", None)

        if plus_di is None or minus_di is None:
            return None

        # ============================================================
        # 5M TREND & SLOPES
        # ============================================================
        previous55_5m = self._previous_closed(ema55_5m)
        if previous55_5m is None:
            return None

        ema55_rising = value55_5m > previous55_5m
        ema55_falling = value55_5m < previous55_5m

        if value55_5m > value200_5m:
            trend_5m = "UP"
        elif value55_5m < value200_5m:
            trend_5m = "DOWN"
        else:
            return None

        # ============================================================
        # 5M ADX FILTERS
        # ============================================================
        adx_has_strength = value_adx_5m > self.ADX_5M_THRESHOLD
        adx_overextended = value_adx_5m > self.ADX_5M_MAX
        adx_reversal = bool(getattr(adx_live, "is_reversal", False))

        di_bullish = plus_di > minus_di
        di_bearish = minus_di > plus_di

        long_trend = trend_5m == "UP" and ema55_rising and adx_has_strength and di_bullish
        short_trend = trend_5m == "DOWN" and ema55_falling and adx_has_strength and di_bearish

        # ============================================================
        # 1M INDICATORS
        # ============================================================
        ema9 = self._get_series(tf1, "EMA", "EMA 9")
        ema21 = self._get_series(tf1, "EMA", "EMA 21")
        ema55 = self._get_series(tf1, "EMA", "EMA 55")

        if not all([ema9, ema21, ema55]) or not all([ema9.live, ema21.live, ema55.live]):
            return None

        value9 = ema9.live.value
        value21 = ema21.live.value
        value55 = ema55.live.value

        previous9 = self._previous_closed(ema9)
        previous21 = self._previous_closed(ema21)
        previous55 = self._previous_closed(ema55)

        if not all([previous9, previous21, previous55]):
            return None

        # ============================================================
        # 1M STRUCTURE
        # ============================================================
        bullish_cross = previous9 <= previous21 and value9 > value21
        bearish_cross = previous9 >= previous21 and value9 < value21

        bullish_structure = value21 > value55
        bearish_structure = value21 < value55

        ema21_rising = value21 > previous21
        ema21_falling = value21 < previous21

        ema55_rising = value55 > previous55
        ema55_falling = value55 < previous55

        # ============================================================
        # 1M ADX MOMENTUM CONFIRMATION
        # ============================================================
        momentum_long_1m = True
        momentum_short_1m = True

        adx_1m = self._get_series(tf1, "ADX", "ADX 14")
        previous_adx_1m = self._last_closed(adx_1m)

        if adx_1m and adx_1m.live and previous_adx_1m:
            value_adx_1m = adx_1m.live.adx
            plus_di_1m = getattr(adx_1m.live, "plus_di", None)
            minus_di_1m = getattr(adx_1m.live, "minus_di", None)

            if plus_di_1m is not None and minus_di_1m is not None:
                adx_1m_has_strength = value_adx_1m >= self.ADX_1M_FLOOR
                adx_1m_rising = value_adx_1m > previous_adx_1m.adx

                momentum_long_1m = adx_1m_has_strength and adx_1m_rising and (plus_di_1m > minus_di_1m)
                momentum_short_1m = adx_1m_has_strength and adx_1m_rising and (minus_di_1m > plus_di_1m)

        # ============================================================
        # POSITION & EXITS EVALUATION
        # ============================================================
        portfolio = state.portfolio
        position = portfolio.position if portfolio is not None else None

        if position is None:
            # Reversal o mercado sobre-extendido bloquean nuevas entradas
            if adx_reversal or adx_overextended:
                return None

            if long_trend:
                if bullish_cross and bullish_structure and ema21_rising and ema55_rising and momentum_long_1m:
                    return Signal(action="BUY", quantity=1)

            if short_trend:
                if bearish_cross and bearish_structure and ema21_falling and ema55_falling and momentum_short_1m:
                    return Signal(action="SELL", quantity=1)

            return None

        # ============================================================
        # EXIT LOGIC
        # ============================================================
        
        # Condiciones base de pérdida de fuerza 5m evaluadas una vez
        adx_strength_exit_long = adx_reversal or (minus_di >= plus_di and not adx_has_strength)
        adx_strength_exit_short = adx_reversal or (plus_di >= minus_di and not adx_has_strength)

        # ------------------- LONG EXIT -------------------
        if position.side == Side.BUY:
            
            # 1. EMERGENCY STOP (Corte rápido de flash crash)
            # La EMA rápida cruzando violentamente la estructura mayor
            if value9 < value55:
                return Signal(action="EXIT")
                
            # 2. Cambio Estructural 5m
            if trend_5m == "DOWN":
                return Signal(action="EXIT")

            # 3. Dynamic Trailing Stop (Toma de ganancias anticipada)
            # Si en 5m hay debilidad, no espera a perder la estructura 55, sale al primer cruce 9/21
            if adx_strength_exit_long and (value9 < value21):
                return Signal(action="EXIT")

            return None

        # ------------------- SHORT EXIT -------------------
        if position.side == Side.SELL:
            
            # 1. EMERGENCY STOP (Corte rápido de spikes)
            if value9 > value55:
                return Signal(action="EXIT")
                
            # 2. Cambio Estructural 5m
            if trend_5m == "UP":
                return Signal(action="EXIT")

            # 3. Dynamic Trailing Stop (Toma de ganancias anticipada)
            if adx_strength_exit_short and (value9 > value21):
                return Signal(action="EXIT")

            return None

        return None

    # ================================================================
    # HELPERS
    # ================================================================
    def _previous_closed(self, series):
        history = getattr(series, "history", None)
        if not history: return None
        return history[-1].value

    def _last_closed(self, series):
        history = getattr(series, "history", None)
        if not history: return None
        return history[-1]

    def _get_series(self, tf, kind, label):
        try:
            return tf.get_series(kind, label)
        except KeyError:
            return None