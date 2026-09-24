from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Multi-Timeframe Trend Pullback Strategy.

    15m:
        Regime / macro trend filter.

    5m:
        Trend structure + ADX confirmation.

    1m:
        Fresh EMA 9/21 cross used as the actual entry trigger.

    Important:
        The 5m does NOT need to cross at the exact same moment as
        the 1m. The 5m establishes the setup and the 1m provides
        the trigger.

    Exits:
        - 5m structure failure
        - opposite EMA 9/21 structure
        - DI reversal
        - ADX falling below the minimum trend threshold
        - ADX reversal flag when supplied by the engine
    """

    DEFAULT_PARAMS = {
        "adx_strength": 18.0,
        "adx_overextended": 55.0,
        "adx_exit": 15.0,
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
        # 15M MACRO TREND
        # ============================================================

        ema21_15 = self._get_series(tf15, "EMA", "EMA 21")
        ema55_15 = self._get_series(tf15, "EMA", "EMA 55")
        adx_15 = self._get_series(tf15, "ADX", "ADX 14")

        if not all([ema21_15, ema55_15, adx_15]):
            return None

        if not all([
            ema21_15.live,
            ema55_15.live,
            adx_15.live,
        ]):
            return None

        previous55_15 = self._previous_closed(ema55_15)

        if previous55_15 is None:
            return None

        value21_15 = ema21_15.live.value
        value55_15 = ema55_15.live.value

        adx_live_15 = adx_15.live

        plus_di_15 = getattr(adx_live_15, "plus_di", None)
        minus_di_15 = getattr(adx_live_15, "minus_di", None)

        if plus_di_15 is None or minus_di_15 is None:
            return None

        macro_up = (
            value21_15 > value55_15
            and value55_15 > previous55_15
            and plus_di_15 > minus_di_15
            and adx_live_15.adx >= self.adx_strength
        )

        macro_down = (
            value21_15 < value55_15
            and value55_15 < previous55_15
            and minus_di_15 > plus_di_15
            and adx_live_15.adx >= self.adx_strength
        )

        # ============================================================
        # 5M TREND / SETUP
        # ============================================================

        ema9 = self._get_series(tf5, "EMA", "EMA 9")
        ema21 = self._get_series(tf5, "EMA", "EMA 21")
        ema55 = self._get_series(tf5, "EMA", "EMA 55")
        adx = self._get_series(tf5, "ADX", "ADX 14")

        if not all([ema9, ema21, ema55, adx]):
            return None

        if not all([
            ema9.live,
            ema21.live,
            ema55.live,
            adx.live,
        ]):
            return None

        previous21 = self._previous_closed(ema21)
        previous55 = self._previous_closed(ema55)
        previous_adx = self._last_closed(adx)

        if previous21 is None or previous55 is None:
            return None

        value9 = ema9.live.value
        value21 = ema21.live.value
        value55 = ema55.live.value

        adx_live = adx.live

        plus_di = getattr(adx_live, "plus_di", None)
        minus_di = getattr(adx_live, "minus_di", None)

        if plus_di is None or minus_di is None:
            return None

        # ADX should preferably be rising for new entries.
        adx_rising = (
            previous_adx is not None
            and adx_live.adx > previous_adx.adx
        )

        # ADX > 55 is considered too extended for a NEW entry.
        # It does NOT force an existing trade to exit.
        adx_entry_ok = (
            adx_live.adx >= self.adx_strength
            and adx_live.adx < self.adx_overextended
        )

        # Existing trend remains alive while ADX stays above exit level.
        adx_alive = adx_live.adx >= self.adx_exit

        # ------------------------------------------------------------
        # Bullish 5m structure
        # ------------------------------------------------------------

        bullish_structure = (
            value9 > value21
            and value21 > value55
            and value21 >= previous21
            and value55 >= previous55
            and plus_di > minus_di
        )

        # ------------------------------------------------------------
        # Bearish 5m structure
        # ------------------------------------------------------------

        bearish_structure = (
            value9 < value21
            and value21 < value55
            and value21 <= previous21
            and value55 <= previous55
            and minus_di > plus_di
        )

        # ============================================================
        # 1M ENTRY TRIGGER
        # ============================================================

        ema9_1 = self._get_series(tf1, "EMA", "EMA 9")
        ema21_1 = self._get_series(tf1, "EMA", "EMA 21")

        if not all([ema9_1, ema21_1]):
            return None

        if not all([
            ema9_1.live,
            ema21_1.live,
        ]):
            return None

        previous9_1 = self._previous_closed(ema9_1)
        previous21_1 = self._previous_closed(ema21_1)

        if previous9_1 is None or previous21_1 is None:
            return None

        value9_1 = ema9_1.live.value
        value21_1 = ema21_1.live.value

        # Fresh 1m bullish cross.
        bullish_cross_1m = (
            previous9_1 <= previous21_1
            and value9_1 > value21_1
        )

        # Fresh 1m bearish cross.
        bearish_cross_1m = (
            previous9_1 >= previous21_1
            and value9_1 < value21_1
        )

        # Require the slow EMA to point in the same direction.
        bullish_trigger = (
            bullish_cross_1m
            and value21_1 >= previous21_1
        )

        bearish_trigger = (
            bearish_cross_1m
            and value21_1 <= previous21_1
        )

        # ============================================================
        # POSITION
        # ============================================================

        portfolio = state.portfolio
        position = portfolio.position if portfolio is not None else None

        # ============================================================
        # ENTRIES
        # ============================================================

        if position is None:

            # --------------------------------------------------------
            # LONG
            # --------------------------------------------------------

            if (
                macro_up
                and bullish_structure
                and adx_entry_ok
                and adx_rising
                and bullish_trigger
            ):
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            # --------------------------------------------------------
            # SHORT
            # --------------------------------------------------------

            if (
                macro_down
                and bearish_structure
                and adx_entry_ok
                and adx_rising
                and bearish_trigger
            ):
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # ============================================================
        # EXITS
        # ============================================================

        adx_reversal = bool(
            getattr(adx_live, "is_reversal", False)
        )

        # ============================================================
        # LONG EXIT
        # ============================================================

        if position.side == Side.BUY:

            if (
                adx_reversal
                or not adx_alive
                or value21 < value55
                or value9 < value21
                or minus_di > plus_di
            ):
                return Signal(action="EXIT")

            return None

        # ============================================================
        # SHORT EXIT
        # ============================================================

        if position.side == Side.SELL:

            if (
                adx_reversal
                or not adx_alive
                or value21 > value55
                or value9 > value21
                or plus_di > minus_di
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
                if "value" in value:
                    value = value["value"]
                else:
                    continue

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