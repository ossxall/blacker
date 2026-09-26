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


class MTFPullback(Strategy):
    """
    Reversion de pullback con filtro de regimen multitimeframe.

    Idea:
      1h  -> regime.  Solo se compra un dip si el par rapido/lento de la
             hora no esta en tendencia bajista.  Este filtro es lo que
             separa la estrategia de un "comprar la caida" sin mas: sin el,
             la estrategia pierde dinero.
      15m -> setup.   Precio estirado al menos `stretch` ATR por debajo de
             su EMA, medido en unidades de volatilidad (no en porcentaje),
             de modo que el umbral se adapta al regime.
      1m  -> trigger.  Opcional: entrar al giro y no a la caida.  Por
             defecto desactivado porque reduce demasiado el numero de
             operaciones.

    Salida: revierte a la media, se invalida el regimen, o se agota el
    limite de barras.

    NOTA HONESTA: esta estrategia NO esta verificada como rentable.  Con los
    valores por defecto (stretch 3.25, exit_dev -0.5) da +3,291 brutos en 27
    operaciones sobre las 826,907 ticks de junio+ julio, es decir ~+19 bps por
    operacion, y +1,569 neto assumiendo 10 bps por viaje.  Eso sobrevive a
    costes y, a diferencia de los otros candidatos, mantiene el signo en junio
    y en julio por separado y en todo el vecindario de parametros.  Pero con
    solo 27 operaciones el intervalo de confianza del 95% es [-2,629, +5,165]
    y la probabilidad de que el edge real sea nulo es ~0.21.  No es una
    estrategia probada: es un candidato de baja frecuencia (0.5 operaciones al
    dia) que preserva capital (max drawdown -1.5% frente a -11.7% de buy &
    hold) mas que una que genere alpha demostrable.
    """

    DEFAULT_PARAMS = {
        # --- etiquetas de las series (deben coincidir con la config) ---
        "label_regime_fast": "EMA 20",
        "label_regime_slow": "EMA 50",
        "label_ema": "EMA 34",
        "label_atr": "ATR 14",
        "label_trigger": "EMA 9",
        # --- regimen 1h ---
        "regime_fast": 20,
        "regime_slow": 50,
        # --- setup 15m ---
        "ema_period": 34,
        "atr_period": 14,
        # centro de la meseta robusta medida sobre las 826,907 ticks de
        # junio+ julio (3.0 - 3.75 da resultado positivo en ambos meses)
        "stretch": 3.25,
        # --- salida ---
        "exit_dev": -0.5,
        "max_bars": 34,
        "cooldown_bars": 2,
        # --- opciones ---
        "allow_short": False,
        "use_trigger": False,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.label_regime_fast = str(merged["label_regime_fast"])
        self.label_regime_slow = str(merged["label_regime_slow"])
        self.label_ema = str(merged["label_ema"])
        self.label_atr = str(merged["label_atr"])
        self.label_trigger = str(merged["label_trigger"])

        self.ema_period = int(merged["ema_period"])
        self.atr_period = int(merged["atr_period"])
        self.stretch = float(merged["stretch"])
        self.exit_dev = float(merged["exit_dev"])
        self.max_bars = int(merged["max_bars"])
        self.cooldown_bars = int(merged["cooldown_bars"])

        self.allow_short = bool(merged["allow_short"])
        self.use_trigger = bool(merged["use_trigger"])

        # ¿Estamos ya estirados?  Sirve para entrar una sola vez por
        # episodio de estiramiento y no en cada tick mientras dura.
        self._was_stretched = False

        # Barra 15m de entrada, para contar barras whilst in position.
        self._entry_time: int | None = None

        # Barras de calma tras salir.
        self._cooldown = 0

    def evaluate(self, state: EngineState):

        # ---------------------------------------------------------
        # TIMEFRAMES
        # ---------------------------------------------------------
        tf15 = state.timeframes.get("15m")
        tf1h = state.timeframes.get("1h")

        if tf15 is None or tf1h is None:
            return None

        # ---------------------------------------------------------
        # 1h - REGIMEN
        # ---------------------------------------------------------
        regime_fast = self._get_series(
            tf1h,
            "EMA",
            self.label_regime_fast
        )
        regime_slow = self._get_series(
            tf1h,
            "EMA",
            self.label_regime_slow
        )

        if not all([regime_fast, regime_slow]):
            return None

        if not all([
            regime_fast.live,
            regime_slow.live
        ]):
            return None

        uptrend = (
            regime_fast.live.value
            >= regime_slow.live.value
        )

        downtrend = (
            regime_fast.live.value
            <= regime_slow.live.value
        )

        # ---------------------------------------------------------
        # 15m - SETUP
        # ---------------------------------------------------------
        ema = self._get_series(tf15, "EMA", self.label_ema)
        atr = self._get_series(tf15, "ATR", self.label_atr)

        if not all([ema, atr]):
            return None

        if not all([ema.live, atr.live]):
            return None

        atr_value = atr.live.atr

        # El ATR se necesita para normalizar; sin el no hay setup.
        if atr_value is None or atr_value <= 0:
            return None

        close = atr.live.close
        ema_value = ema.live.value

        if close is None or close <= 0:
            return None

        # Desviacion en unidades de volatilidad.
        dev = (close - ema_value) / atr_value

        # ---------------------------------------------------------
        # 1m - TRIGGER OPCIONAL
        # ---------------------------------------------------------
        trigger_up = True
        trigger_down = True

        if self.use_trigger:
            trigger = self._get_series(
                state.timeframes.get("1m"),
                "EMA",
                self.label_trigger
            ) if state.timeframes.get("1m") is not None else None

            if not trigger or not trigger.live:
                return None

            previous = self._previous_value(trigger)

            if previous is None:
                return None

            # Giro: el trigger supera su propio valor anterior.
            trigger_up = (
                trigger.live.value > previous
            )
            trigger_down = (
                trigger.live.value < previous
            )

        # ---------------------------------------------------------
        # ESTIRAMIENTO
        # ---------------------------------------------------------
        stretched_long = dev <= -self.stretch
        stretched_short = dev >= self.stretch

        # Solo el primer tick de cada episodio.
        fresh_long = stretched_long and not self._was_stretched
        fresh_short = stretched_short and not self._was_stretched

        self._was_stretched = stretched_long or stretched_short

        # ---------------------------------------------------------
        # PUESTO
        # ---------------------------------------------------------
        portfolio = state.portfolio
        position = (
            portfolio.position
            if portfolio is not None
            else None
        )

        if self._cooldown > 0:
            self._cooldown -= 1

        # ---------------------------------------------------------
        # FLAT -> ENTRADA
        # ---------------------------------------------------------
        if position is None:

            if self._cooldown > 0:
                return None

            # No se compra un dip en un regimen bajista de 1h.
            if (
                fresh_long
                and uptrend
                and trigger_up
            ):
                self._entry_time = atr.live.time

                return Signal(
                    action="BUY",
                    quantity=1
                )

            # En mirror, no se vende un rebote en un regimen alcista.
            if (
                self.allow_short
                and fresh_short
                and downtrend
                and trigger_down
            ):
                self._entry_time = atr.live.time

                return Signal(
                    action="SELL",
                    quantity=1
                )

            return None

        # ---------------------------------------------------------
        # EN POSICION -> SALIR
        # ---------------------------------------------------------
        bars_held = self._bars_held(tf15, atr)

        if position.side == Side.BUY:

            reverted = dev >= -self.exit_dev
            regime_lost = not uptrend
            expired = bars_held >= self.max_bars

            if reverted or regime_lost or expired:
                self._cooldown = self.cooldown_bars
                self._entry_time = None

                return Signal(action="EXIT")

            return None

        if position.side == Side.SELL:

            reverted = dev <= self.exit_dev
            regime_lost = not downtrend
            expired = bars_held >= self.max_bars

            if reverted or regime_lost or expired:
                self._cooldown = self.cooldown_bars
                self._entry_time = None

                return Signal(action="EXIT")

            return None

        return None

    def _bars_held(
        self,
        tf15,
        atr,
    ) -> int:
        if self._entry_time is None:
            return 0

        now = getattr(atr.live, "time", None)

        if now is None:
            return 0

        span = tf15.timeframe_ms

        if not span:
            return 0

        return max(0, int((now - self._entry_time) * 1000 // span))

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
    def _get_series(tf, kind, label):
        if tf is None:
            return None

        try:
            return tf.get_series(
                kind,
                label
            )
        except KeyError:
            return None

