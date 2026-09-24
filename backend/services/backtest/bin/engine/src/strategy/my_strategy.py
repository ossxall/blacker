from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Multi-Frame EMA + ADX -- 5m main timeframe.

    - 15m : Macro trend gate. EMA 21/55 structure + EMA55 slope + +DI/-DI.
    - 5m  : Main timeframe. Setup engine: EMA 9/21 cross, EMA 21/55 anchor,
            ADX 14 strength, +DI/-DI direction and ADX momentum.
    - 1m  : Entry trigger. Fresh EMA 9/21 cross aligned with the 5m setup.

    Only EMA and ADX series are consumed.
    """

    ADX_STRENGTH = 20.0
    ADX_OVEREXTENDED = 50.0

    def evaluate(self, state: EngineState):

        tf15 = state.timeframes.get("15m")
        tf5 = state.timeframes.get("5m")
        tf1 = state.timeframes.get("1m")

        if tf15 is None or tf5 is None or tf1 is None:
            return None

        # ============================================================
        # 15M MACRO TREND (context gate)
        # ============================================================
        ema21_15 = self._get_series(tf15, "EMA", "EMA 21")
        ema55_15 = self._get_series(tf15, "EMA", "EMA 55")
        adx_15 = self._get_series(tf15, "ADX", "ADX 14")

        if not all([ema21_15, ema55_15, adx_15]) or not all([
            ema21_15.live,
            ema55_15.live,
            adx_15.live,
        ]):
            return None

        value21_15 = ema21_15.live.value
        value55_15 = ema55_15.live.value
        previous55_15 = self._previous_closed(ema55_15)
        if previous55_15 is None:
            return None

        adx_live_15 = adx_15.live
        plus_di_15 = getattr(adx_live_15, "plus_di", None)
        minus_di_15 = getattr(adx_live_15, "minus_di", None)
        if plus_di_15 is None or minus_di_15 is None:
            return None

        macro_up = (
            value21_15 > value55_15
            and value55_15 > previous55_15
            and plus_di_15 > minus_di_15
        )
        macro_down = (
            value21_15 < value55_15
            and value55_15 < previous55_15
            and minus_di_15 > plus_di_15
        )

        # ============================================================
        # 5M MAIN TIMEFRAME (setup engine)
        # ============================================================
        ema9 = self._get_series(tf5, "EMA", "EMA 9")
        ema21 = self._get_series(tf5, "EMA", "EMA 21")
        ema55 = self._get_series(tf5, "EMA", "EMA 55")
        adx = self._get_series(tf5, "ADX", "ADX 14")

        if not all([ema9, ema21, ema55, adx]) or not all([
            ema9.live,
            ema21.live,
            ema55.live,
            adx.live,
        ]):
            return None

        value9 = ema9.live.value
        value21 = ema21.live.value
        value55 = ema55.live.value
        previous9 = self._previous_closed(ema9)
        previous21 = self._previous_closed(ema21)
        previous55 = self._previous_closed(ema55)
        if not all([previous9, previous21, previous55]):
            return None

        adx_live = adx.live
        value_adx = adx_live.adx
        previous_adx = self._last_closed(adx)

        plus_di = getattr(adx_live, "plus_di", None)
        minus_di = getattr(adx_live, "minus_di", None)
        if plus_di is None or minus_di is None:
            return None

        adx_has_strength = value_adx > self.ADX_STRENGTH
        adx_overextended = value_adx > self.ADX_OVEREXTENDED
        adx_reversal = bool(getattr(adx_live, "is_reversal", False))
        adx_rising = previous_adx is not None and value_adx > previous_adx.adx

        bullish_cross_5m = previous9 <= previous21 and value9 > value21
        bearish_cross_5m = previous9 >= previous21 and value9 < value21

        # ============================================================
        # 1M ENTRY CONFIRMATION (alignment, not an event)
        # ============================================================
        ema9_1 = self._get_series(tf1, "EMA", "EMA 9")
        ema21_1 = self._get_series(tf1, "EMA", "EMA 21")

        if not all([ema9_1, ema21_1]) or not all([ema9_1.live, ema21_1.live]):
            return None

        value9_1 = ema9_1.live.value
        value21_1 = ema21_1.live.value
        previous21_1 = self._previous_closed(ema21_1)
        if previous21_1 is None:
            return None

        bullish_aligned_1m = value9_1 > value21_1 and value21_1 > previous21_1
        bearish_aligned_1m = value9_1 < value21_1 and value21_1 < previous21_1

        # ============================================================
        # POSITION & SIGNALS
        # ============================================================
        portfolio = state.portfolio
        position = portfolio.position if portfolio is not None else None

        if position is None:
            if adx_reversal or adx_overextended or not adx_rising:
                return None

            if (
                macro_up
                and bullish_cross_5m
                and value21 > value55
                and adx_has_strength
                and plus_di > minus_di
                and bullish_aligned_1m
            ):
                return Signal(action="BUY", quantity=1)

            if (
                macro_down
                and bearish_cross_5m
                and value21 < value55
                and adx_has_strength
                and minus_di > plus_di
                and bearish_aligned_1m
            ):
                return Signal(action="SELL", quantity=1)

            return None

        # ============================================================
        # EXITS
        # ============================================================
        if position.side == Side.BUY:
            if (
                adx_reversal
                or (minus_di >= plus_di and value9 < value21)
                or value21 < value55
            ):
                return Signal(action="EXIT")
            return None

        if position.side == Side.SELL:
            if (
                adx_reversal
                or (plus_di >= minus_di and value9 > value21)
                or value21 > value55
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