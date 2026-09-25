from core.engine_state import EngineState

from strategy.base import Strategy

from orders import Signal, Side


class Strategy1(Strategy):
    """
    MTF EMA 20/50 + ADX 14

    15m = dirección macro
    5m  = tendencia + fuerza
    1m  = entrada después de recuperación

    Indicadores:
        EMA 20
        EMA 50
        ADX 14

    Diseño:
        - Menos entradas en zonas laterales.
        - No compara EMAs entre diferentes timeframes.
        - Utiliza pendiente de EMA20.
        - Short ligeramente más permisivo.
        - Long ligeramente más selectivo.
    """

    DEFAULT_PARAMS = {
        "adx_strength": 20.0,
        "adx_exit": 15.0,

        # Pendiente mínima relativa de EMA20.
        # 0.0 = solamente exige que esté subiendo/bajando.
        "ema_slope_5": 0.0,

        # Número de barras 1m que utilizamos para comprobar
        # la dirección de EMA20.
        "execution_lookback": 1,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.adx_strength = float(
            merged["adx_strength"]
        )

        self.adx_exit = float(
            merged["adx_exit"]
        )

        self.ema_slope_5 = float(
            merged["ema_slope_5"]
        )

        self.execution_lookback = int(
            merged["execution_lookback"]
        )

    def evaluate(self, state: EngineState):

        # ============================================================
        # TIMEFRAMES
        # ============================================================

        tf15 = state.timeframes.get("15m")
        tf5 = state.timeframes.get("5m")
        tf1 = state.timeframes.get("1m")

        if tf15 is None or tf5 is None or tf1 is None:
            return None

        # ============================================================
        # 15M
        # MACRO TREND
        # ============================================================

        ema20_15 = self._get_series(
            tf15,
            "EMA",
            "EMA 20"
        )

        ema50_15 = self._get_series(
            tf15,
            "EMA",
            "EMA 50"
        )

        adx_15 = self._get_series(
            tf15,
            "ADX",
            "ADX 14"
        )

        if not all([
            ema20_15,
            ema50_15,
            adx_15,
        ]):
            return None

        if not all([
            ema20_15.live,
            ema50_15.live,
            adx_15.live,
        ]):
            return None

        ema20_value_15 = ema20_15.live.value
        ema50_value_15 = ema50_15.live.value

        adx_value_15 = adx_15.live.adx

        # ------------------------------------------------------------
        # MACRO LONG
        # ------------------------------------------------------------

        macro_long = (
            ema20_value_15 > ema50_value_15
            and adx_value_15 >= self.adx_strength
        )

        # ------------------------------------------------------------
        # MACRO SHORT
        # ------------------------------------------------------------

        macro_short = (
            ema20_value_15 < ema50_value_15
            and adx_value_15 >= self.adx_strength
        )

        # ============================================================
        # 5M
        # TREND + ADX
        # ============================================================

        ema20_5 = self._get_series(
            tf5,
            "EMA",
            "EMA 20"
        )

        ema50_5 = self._get_series(
            tf5,
            "EMA",
            "EMA 50"
        )

        adx_5 = self._get_series(
            tf5,
            "ADX",
            "ADX 14"
        )

        if not all([
            ema20_5,
            ema50_5,
            adx_5,
        ]):
            return None

        if not all([
            ema20_5.live,
            ema50_5.live,
            adx_5.live,
        ]):
            return None

        previous_ema20_5 = self._previous_value(
            ema20_5
        )

        previous_ema50_5 = self._previous_value(
            ema50_5
        )

        previous_adx_5 = self._last_closed(
            adx_5
        )

        if (
            previous_ema20_5 is None
            or previous_ema50_5 is None
            or previous_adx_5 is None
        ):
            return None

        ema20_value_5 = ema20_5.live.value
        ema50_value_5 = ema50_5.live.value
        adx_value_5 = adx_5.live.adx

        # ------------------------------------------------------------
        # 5M STRUCTURE
        # ------------------------------------------------------------

        bullish_5 = (
            ema20_value_5 > ema50_value_5
        )

        bearish_5 = (
            ema20_value_5 < ema50_value_5
        )

        # ------------------------------------------------------------
        # 5M EMA20 SLOPE
        # ------------------------------------------------------------

        ema20_slope_5 = (
            ema20_value_5 - previous_ema20_5
        )

        bullish_slope_5 = (
            ema20_slope_5 > self.ema_slope_5
        )

        bearish_slope_5 = (
            ema20_slope_5 < -self.ema_slope_5
        )

        # ------------------------------------------------------------
        # ADX
        # ------------------------------------------------------------

        adx_strong_5 = (
            adx_value_5 >= self.adx_strength
        )

        # ============================================================
        # 1M
        # EXECUTION
        # ============================================================

        ema20_1 = self._get_series(
            tf1,
            "EMA",
            "EMA 20"
        )

        ema50_1 = self._get_series(
            tf1,
            "EMA",
            "EMA 50"
        )

        if not all([
            ema20_1,
            ema50_1,
        ]):
            return None

        if not all([
            ema20_1.live,
            ema50_1.live,
        ]):
            return None

        previous_ema20_1 = self._previous_value(
            ema20_1
        )

        previous_ema50_1 = self._previous_value(
            ema50_1
        )

        if (
            previous_ema20_1 is None
            or previous_ema50_1 is None
        ):
            return None

        ema20_value_1 = ema20_1.live.value
        ema50_value_1 = ema50_1.live.value

        # ============================================================
        # 1M LONG
        # ============================================================

        long_structure_1 = (
            ema20_value_1 > ema50_value_1
        )

        long_slope_1 = (
            ema20_value_1 > previous_ema20_1
        )

        # Recuperación:
        # EMA20 está por encima de EMA50 y continúa subiendo.
        long_trigger = (
            long_structure_1
            and long_slope_1
        )

        # ============================================================
        # 1M SHORT
        # ============================================================

        short_structure_1 = (
            ema20_value_1 < ema50_value_1
        )

        short_slope_1 = (
            ema20_value_1 < previous_ema20_1
        )

        short_trigger = (
            short_structure_1
            and short_slope_1
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
        # ENTRY
        # ============================================================

        if position is None:

            # ========================================================
            # LONG
            # ========================================================
            #
            # Más selectivo:
            # macro + 5m estructura + 5m pendiente + ADX
            # + 1m recuperación.
            #

            if (
                macro_long
                and bullish_5
                and bullish_slope_5
                and adx_strong_5
                and long_trigger
            ):
                return Signal(
                    action="BUY",
                    quantity=1,
                )

            # ========================================================
            # SHORT
            # ========================================================
            #
            # El histórico actual muestra mucha más actividad
            # rentable en este lado.
            #

            if (
                macro_short
                and bearish_5
                and bearish_slope_5
                and adx_strong_5
                and short_trigger
            ):
                return Signal(
                    action="SELL",
                    quantity=1,
                )

            return None

        # ============================================================
        # EXIT
        # ============================================================

        adx_reversal = bool(
            getattr(
                adx_5.live,
                "is_reversal",
                False
            )
        )

        adx_alive = (
            adx_value_5 >= self.adx_exit
        )

        # ============================================================
        # EXIT LONG
        # ============================================================

        if position.side == Side.BUY:

            structure_failed = (
                ema20_value_5 < ema50_value_5
            )

            trend_lost = (
                ema20_value_5 < previous_ema20_5
                and ema20_value_5 < ema50_value_5
            )

            if (
                structure_failed
                or trend_lost
                or not adx_alive
                or adx_reversal
            ):
                return Signal(
                    action="EXIT"
                )

            return None

        # ============================================================
        # EXIT SHORT
        # ============================================================

        if position.side == Side.SELL:

            structure_failed = (
                ema20_value_5 > ema50_value_5
            )

            trend_lost = (
                ema20_value_5 > previous_ema20_5
                and ema20_value_5 > ema50_value_5
            )

            if (
                structure_failed
                or trend_lost
                or not adx_alive
                or adx_reversal
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
    def _previous_value(series):

        history = getattr(
            series,
            "history",
            None
        )

        if not history:
            return None

        return history[-1].value

    @staticmethod
    def _last_closed(series):

        history = getattr(
            series,
            "history",
            None
        )

        if not history:
            return None

        return history[-1]

    @staticmethod
    def _get_series(tf, kind, label):

        try:
            return tf.get_series(
                kind,
                label
            )

        except KeyError:
            return None