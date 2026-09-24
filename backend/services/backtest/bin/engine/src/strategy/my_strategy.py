from core.engine_state import EngineState

from strategy.base import Strategy

from orders import Signal, Side


class Strategy1(Strategy):
    """
    Multi-Timeframe EMA 20/50 + ADX Strategy.

    15m:
        Macro direction only.

    5m:
        Trend confirmation + ADX strength.

    1m:
        Entry execution / pullback continuation.

    Indicators:
        - EMA 20
        - EMA 50
        - ADX 14

    No RSI, MACD, DI or other indicators.

    Philosophy:
        15m decides the direction.
        5m confirms the trend.
        1m provides the execution trigger.

    LONG:
        15m EMA20 > EMA50
        15m ADX >= 20

        5m EMA20 > EMA50
        5m ADX >= 20
        5m ADX rising

        1m EMA20 > EMA50
        1m close > EMA20

    SHORT:
        Opposite conditions.

    The 1m does NOT require a fresh EMA cross.
    This allows continuation entries after pullbacks.

    Exits:
        - 5m EMA20/EMA50 structure failure
        - 5m ADX falls below exit threshold
        - ADX reversal flag when supplied by engine
    """

    DEFAULT_PARAMS = {
        "adx_strength": 20.0,
        "adx_exit": 15.0,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.adx_strength = float(merged["adx_strength"])
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

        ema20_15 = self._get_series(tf15, "EMA", "EMA 20")
        ema50_15 = self._get_series(tf15, "EMA", "EMA 50")
        adx_15 = self._get_series(tf15, "ADX", "ADX 14")

        if not all([ema20_15, ema50_15, adx_15]):
            return None

        if not all([
            ema20_15.live,
            ema50_15.live,
            adx_15.live,
        ]):
            return None

        value20_15 = ema20_15.live.value
        value50_15 = ema50_15.live.value
        adx_live_15 = adx_15.live

        # ------------------------------------------------------------
        # 15m direction
        #
        # Deliberately simple:
        # only EMA relationship + ADX.
        # ------------------------------------------------------------

        macro_up = (
            value20_15 > value50_15
            and adx_live_15.adx >= self.adx_strength
        )

        macro_down = (
            value20_15 < value50_15
            and adx_live_15.adx >= self.adx_strength
        )

        # ============================================================
        # 5M TREND CONFIRMATION
        # ============================================================

        ema20_5 = self._get_series(tf5, "EMA", "EMA 20")
        ema50_5 = self._get_series(tf5, "EMA", "EMA 50")
        adx_5 = self._get_series(tf5, "ADX", "ADX 14")

        if not all([ema20_5, ema50_5, adx_5]):
            return None

        if not all([
            ema20_5.live,
            ema50_5.live,
            adx_5.live,
        ]):
            return None

        previous_adx_5 = self._last_closed(adx_5)

        if previous_adx_5 is None:
            return None

        value20_5 = ema20_5.live.value
        value50_5 = ema50_5.live.value
        adx_live_5 = adx_5.live

        # ------------------------------------------------------------
        # ADX confirmation
        #
        # Only 5m needs ADX to be rising.
        # This removes a major source of over-filtering.
        # ------------------------------------------------------------

        adx_rising_5 = (
            adx_live_5.adx > previous_adx_5.adx
        )

        adx_entry_ok_5 = (
            adx_live_5.adx >= self.adx_strength
        )

        # ------------------------------------------------------------
        # 5m structure
        # ------------------------------------------------------------

        bullish_structure_5 = (
            value20_5 > value50_5
        )

        bearish_structure_5 = (
            value20_5 < value50_5
        )

        # ============================================================
        # 1M EXECUTION
        # ============================================================

        ema20_1 = self._get_series(tf1, "EMA", "EMA 20")
        ema50_1 = self._get_series(tf1, "EMA", "EMA 50")

        if not all([ema20_1, ema50_1]):
            return None

        if not all([
            ema20_1.live,
            ema50_1.live,
        ]):
            return None

        value20_1 = ema20_1.live.value
        value50_1 = ema50_1.live.value

        # ------------------------------------------------------------
        # 1m continuation trigger
        #
        # No fresh cross required.
        #
        # This allows entries while the trend is already established.
        # ------------------------------------------------------------

        bullish_trigger = (
            value20_1 > value50_1
            and value20_1 >= value20_5
        )

        bearish_trigger = (
            value20_1 < value50_1
            and value20_1 <= value20_5
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

        # ============================================================
        # ENTRIES
        # ============================================================

        if position is None:

            # --------------------------------------------------------
            # LONG
            # --------------------------------------------------------

            if (
                macro_up
                and bullish_structure_5
                and adx_entry_ok_5
                and adx_rising_5
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
                and bearish_structure_5
                and adx_entry_ok_5
                and adx_rising_5
                and bearish_trigger
            ):
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # ============================================================
        # EXIT CONDITIONS
        # ============================================================

        adx_reversal = bool(
            getattr(adx_live_5, "is_reversal", False)
        )

        adx_alive = (
            adx_live_5.adx >= self.adx_exit
        )

        # ============================================================
        # LONG EXIT
        # ============================================================

        if position.side == Side.BUY:

            if (
                adx_reversal
                or not adx_alive
                or value20_5 < value50_5
            ):
                return Signal(
                    action="EXIT"
                )

            return None

        # ============================================================
        # SHORT EXIT
        # ============================================================

        if position.side == Side.SELL:

            if (
                adx_reversal
                or not adx_alive
                or value20_5 > value50_5
            ):
                return Signal(
                    action="EXIT"
                )

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

