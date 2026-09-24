from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Multi-Frame Trend-Following strategy. EMA + ADX only.

      15m  Macro trend gate: EMA 21/55 structure, EMA55 slope and
            strong +DI/-DI direction (ADX > strength).
       5m  Setup engine: fresh EMA 9/21 cross aligned with the macro
            trend, EMA 21 > EMA 55, ADX 14 between strength and
            overextended, +DI/-DI direction and ADX momentum.
       1m  Entry trigger: fresh EMA 9/21 cross fired in the same
            direction as the 5m setup, with the slow EMA still rising.

    Exits ride the trend while ADX is alive and break as soon as the
    5m structure (EMA 9/21 cross, EMA 21/55 anchor or ADX reversal)
    gives up.
    """

    # ------------------------------------------------------------------
    # DEFAULT PARAMETERS (override via strategy "params" in the config)
    # ------------------------------------------------------------------

    DEFAULT_PARAMS = {
        "adx_strength": 20.0,
        "adx_overextended": 50.0,
        "adx_exit": 20.0,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.adx_strength = float(merged["adx_strength"])
        self.adx_overextended = float(merged["adx_overextended"])
        self.adx_exit = float(merged["adx_exit"])

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
            and adx_live_15.adx > self.adx_strength
        )
        macro_down = (
            value21_15 < value55_15
            and value55_15 < previous55_15
            and minus_di_15 > plus_di_15
            and adx_live_15.adx > self.adx_strength
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
        if previous9 is None or previous21 is None:
            return None

        adx_live = adx.live
        previous_adx = self._last_closed(adx)

        plus_di = getattr(adx_live, "plus_di", None)
        minus_di = getattr(adx_live, "minus_di", None)
        if plus_di is None or minus_di is None:
            return None

        adx_rising = previous_adx is not None and adx_live.adx > previous_adx.adx
        adx_in_range = self.adx_strength < adx_live.adx < self.adx_overextended
        adx_alive = adx_live.adx > self.adx_exit

        bullish_cross_5m = previous9 <= previous21 and value9 > value21
        bearish_cross_5m = previous9 >= previous21 and value9 < value21

        # ============================================================
        # 1M ENTRY TRIGGER (fresh same-direction cross)
        # ============================================================

        ema9_1 = self._get_series(tf1, "EMA", "EMA 9")
        ema21_1 = self._get_series(tf1, "EMA", "EMA 21")

        if not all([ema9_1, ema21_1]) or not all([ema9_1.live, ema21_1.live]):
            return None

        value9_1 = ema9_1.live.value
        value21_1 = ema21_1.live.value
        previous9_1 = self._previous_closed(ema9_1)
        previous21_1 = self._previous_closed(ema21_1)
        if previous9_1 is None or previous21_1 is None:
            return None

        bullish_cross_1m = previous9_1 <= previous21_1 and value9_1 > value21_1
        bearish_cross_1m = previous9_1 >= previous21_1 and value9_1 < value21_1

        # ============================================================
        # POSITION & SIGNALS
        # ============================================================

        portfolio = state.portfolio
        position = portfolio.position if portfolio is not None else None

        if position is None:
            # ------------------------------------------------------
            # ENTRIES
            # ------------------------------------------------------
            if (
                macro_up
                and bullish_cross_5m
                and value21 > value55
                and adx_in_range
                and adx_rising
                and plus_di > minus_di
                and bullish_cross_1m
                and value21_1 >= previous21_1
            ):
                return Signal(action="BUY", quantity=1)

            if (
                macro_down
                and bearish_cross_5m
                and value21 < value55
                and adx_in_range
                and adx_rising
                and minus_di > plus_di
                and bearish_cross_1m
                and value21_1 <= previous21_1
            ):
                return Signal(action="SELL", quantity=1)

            return None

        # ============================================================
        # EXITS
        # ============================================================

        adx_reversal = bool(getattr(adx_live, "is_reversal", False))

        if position.side == Side.BUY:
            if (
                adx_reversal
                or bearish_cross_5m
                or value21 < value55
                or not adx_alive
                or minus_di >= plus_di
            ):
                return Signal(action="EXIT")
            return None

        if position.side == Side.SELL:
            if (
                adx_reversal
                or bullish_cross_5m
                or value21 > value55
                or not adx_alive
                or plus_di >= minus_di
            ):
                return Signal(action="EXIT")
            return None

        return None

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _unwrap(params: dict) -> dict:
        out = {}
        for key, value in (params or {}).items():
            if isinstance(value, dict):
                value = value.get("value", 0)
            out[key] = value
        return out

    @staticmethod
    def _previous_closed(series):
        history = getattr(series, "history", None)
        if not history:
            return None
        return history[-1].value

    @staticmethod
    def _last_closed(series):
        history = getattr(series, "history", None)
        if not history:
            return None
        return history[-1]

    @staticmethod
    def _get_series(tf, kind, label):
        try:
            return tf.get_series(kind, label)
        except KeyError:
            return None