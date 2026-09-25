from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    DEFAULT_PARAMS = {
        "adx_strength": 20.0,
        "adx_exit": 15.0,
        "ema_slope_5": 0.0,
        "cooldown_bars": 3,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.adx_strength = float(merged["adx_strength"])
        self.adx_exit = float(merged["adx_exit"])
        self.ema_slope_5 = float(merged["ema_slope_5"])
        self.cooldown_bars = int(merged["cooldown_bars"])

        # Estado interno
        self.cooldown = 0

        # Evita volver a entrar en el mismo impulso
        self.long_trigger_used = False
        self.short_trigger_used = False

    def evaluate(self, state: EngineState):
        # ---------------------------------------------------------
        # TIMEFRAMES
        # ---------------------------------------------------------
        tf15 = state.timeframes.get("15m")
        tf5 = state.timeframes.get("5m")
        tf1 = state.timeframes.get("1m")

        if tf15 is None or tf5 is None or tf1 is None:
            return None

        # ---------------------------------------------------------
        # 15m - MACRO
        # ---------------------------------------------------------
        ema20_15 = self._get_series(tf15, "EMA", "EMA 20")
        ema50_15 = self._get_series(tf15, "EMA", "EMA 50")
        adx_15 = self._get_series(tf15, "ADX", "ADX 14")

        if not all([ema20_15, ema50_15, adx_15]):
            return None

        if not all([
            ema20_15.live,
            ema50_15.live,
            adx_15.live
        ]):
            return None

        ema20_value_15 = ema20_15.live.value
        ema50_value_15 = ema50_15.live.value
        adx_value_15 = adx_15.live.adx

        macro_long = (
            ema20_value_15 > ema50_value_15
            and adx_value_15 >= self.adx_strength
        )

        macro_short = (
            ema20_value_15 < ema50_value_15
            and adx_value_15 >= self.adx_strength
        )

        # ---------------------------------------------------------
        # 5m - CONFIRMACION
        # ---------------------------------------------------------
        ema20_5 = self._get_series(tf5, "EMA", "EMA 20")
        ema50_5 = self._get_series(tf5, "EMA", "EMA 50")
        adx_5 = self._get_series(tf5, "ADX", "ADX 14")

        if not all([ema20_5, ema50_5, adx_5]):
            return None

        if not all([
            ema20_5.live,
            ema50_5.live,
            adx_5.live
        ]):
            return None

        previous_ema20_5 = self._previous_value(ema20_5)
        previous_ema50_5 = self._previous_value(ema50_5)
        previous_adx_5 = self._last_closed(adx_5)

        if (
            previous_ema20_5 is None
            or previous_ema50_5 is None
            or previous_adx_5 is None
        ):
            return None

        ema20_value_5 = ema20_5.live.value
        ema50_value_5 = ema50_5.live.value
        adx_value_5 = adx_5.live.adx

        bullish_5 = ema20_value_5 > ema50_value_5
        bearish_5 = ema20_value_5 < ema50_value_5

        ema20_slope_5 = (
            ema20_value_5 - previous_ema20_5
        )

        bullish_slope_5 = (
            ema20_slope_5 > self.ema_slope_5
        )

        bearish_slope_5 = (
            ema20_slope_5 < -self.ema_slope_5
        )

        adx_strong_5 = (
            adx_value_5 >= self.adx_strength
        )

        # ---------------------------------------------------------
        # 1m - TRIGGER
        # ---------------------------------------------------------
        ema20_1 = self._get_series(tf1, "EMA", "EMA 20")
        ema50_1 = self._get_series(tf1, "EMA", "EMA 50")

        if not all([ema20_1, ema50_1]):
            return None

        if not all([
            ema20_1.live,
            ema50_1.live
        ]):
            return None

        previous_ema20_1 = self._previous_value(ema20_1)
        previous_ema50_1 = self._previous_value(ema50_1)

        if (
            previous_ema20_1 is None
            or previous_ema50_1 is None
        ):
            return None

        ema20_value_1 = ema20_1.live.value
        ema50_value_1 = ema50_1.live.value

        # ---------------------------------------------------------
        # NUEVO IMPULSO 1m
        #
        # No basta con que EMA20 > EMA50.
        # Exigimos que EMA20 esté acelerando y que la estructura
        # haya cambiado respecto a la vela anterior.
        # ---------------------------------------------------------

        bullish_structure_now = (
            ema20_value_1 > ema50_value_1
        )

        bullish_structure_previous = (
            previous_ema20_1 > previous_ema50_1
        )

        bearish_structure_now = (
            ema20_value_1 < ema50_value_1
        )

        bearish_structure_previous = (
            previous_ema20_1 < previous_ema50_1
        )

        ema20_rising_1 = (
            ema20_value_1 > previous_ema20_1
        )

        ema20_falling_1 = (
            ema20_value_1 < previous_ema20_1
        )

        # Trigger por NUEVO cambio de estructura
        long_cross = (
            bullish_structure_now
            and not bullish_structure_previous
        )

        short_cross = (
            bearish_structure_now
            and not bearish_structure_previous
        )

        # Trigger alternativo:
        # si la estructura ya estaba establecida, permitimos
        # un nuevo impulso después de que EMA20 pierda pendiente.
        long_impulse = (
            bullish_structure_now
            and ema20_rising_1
        )

        short_impulse = (
            bearish_structure_now
            and ema20_falling_1
        )

        # ---------------------------------------------------------
        # PORTFOLIO
        # ---------------------------------------------------------
        portfolio = state.portfolio
        position = (
            portfolio.position
            if portfolio is not None
            else None
        )

        # ---------------------------------------------------------
        # COOLDOWN
        # ---------------------------------------------------------
        if self.cooldown > 0:
            self.cooldown -= 1

        # ---------------------------------------------------------
        # SIN POSICION -> BUSCAR ENTRADA
        # ---------------------------------------------------------
        if position is None:

            # Si estamos en cooldown no hacemos nada.
            if self.cooldown > 0:
                return None

            # -----------------------------------------------------
            # LONG
            # -----------------------------------------------------
            long_setup = (
                macro_long
                and bullish_5
                and bullish_slope_5
                and adx_strong_5
            )

            long_trigger = (
                long_cross
                or (
                    long_impulse
                    and not self.long_trigger_used
                )
            )

            if (
                long_setup
                and long_trigger
                and not self.long_trigger_used
            ):
                self.long_trigger_used = True
                self.short_trigger_used = False

                return Signal(
                    action="BUY",
                    quantity=1
                )

            # -----------------------------------------------------
            # SHORT
            # -----------------------------------------------------
            short_setup = (
                macro_short
                and bearish_5
                and bearish_slope_5
                and adx_strong_5
            )

            short_trigger = (
                short_cross
                or (
                    short_impulse
                    and not self.short_trigger_used
                )
            )

            if (
                short_setup
                and short_trigger
                and not self.short_trigger_used
            ):
                self.short_trigger_used = True
                self.long_trigger_used = False

                return Signal(
                    action="SELL",
                    quantity=1
                )

            return None

        # ---------------------------------------------------------
        # GESTION DE POSICION
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # LONG
        # ---------------------------------------------------------
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
                self.cooldown = self.cooldown_bars

                # Permitimos buscar un nuevo impulso
                # después de que aparezca una nueva condición.
                self.long_trigger_used = True

                return Signal(action="EXIT")

            return None

        # ---------------------------------------------------------
        # SHORT
        # ---------------------------------------------------------
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
                self.cooldown = self.cooldown_bars

                self.short_trigger_used = True

                return Signal(action="EXIT")

            return None

        return None

    # =============================================================
    # HELPERS
    # =============================================================

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

