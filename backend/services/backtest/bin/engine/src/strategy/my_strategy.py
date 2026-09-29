from core.engine_state import EngineState
from core.portfolio import QUANTITY_EPSILON
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Entrada por confirmacion multi-timeframe y salida en dos etapas.

    La entrada exige que los cinco timeframes estén de acuerdo: 1H da
    la tendencia, 30m la confirma, 15m exige un pullback, 5m marca el
    setup y 1m el gatillo.

    La salida se decide por cuál es el timeframe que se ha girado:

        - 1H en contra        -> cierre completo. La tendencia se
                                rompió y no queda nada que sostener.
        - 1H igual, 30m en
          contra             -> salida parcial (``scale_out_fraction``
                                de la posición). Se asegura parte y se
                                deja correr el resto, que sigue bajo su
                                stop y sus targets originales.
        - 30m de acuerdo      -> mantener.

    El orden importa: 1H manda porque su reverso ya cierra entero, así
    que un 30m en contra no llegaría a ejecutarse nunca. La salida
    parcial solo se pide mientras la posición esté completa, de modo
    que no se repite cada tick mientras el 30m siga girado.
    """

    DEFAULT_PARAMS = {
        # Timeframes
        "tf_trend": "1h",
        "tf_confirm": "30m",
        "tf_pullback": "15m",
        "tf_setup": "5m",
        "tf_trigger": "1m",

        # EMA
        "label_fast": "EMA 20",
        "label_slow": "EMA 50",

        # ADX
        "adx_label": "ADX 14",
        "adx_threshold": 25.0,

        # Position
        "quantity": 1,

        # Cuanto se sale en el cross de 30m, como fraccion de la
        # posicion. Es una fraccion y no una cantidad absoluta para que
        # la misma cuenta sirva con cualquier tamano de posicion.
        "scale_out_fraction": 0.5,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.tf_trend = str(merged["tf_trend"])
        self.tf_confirm = str(merged["tf_confirm"])
        self.tf_pullback = str(merged["tf_pullback"])
        self.tf_setup = str(merged["tf_setup"])
        self.tf_trigger = str(merged["tf_trigger"])

        self.label_fast = str(merged["label_fast"])
        self.label_slow = str(merged["label_slow"])
        self.adx_label = str(merged["adx_label"])

        try:
            self.adx_threshold = float(merged["adx_threshold"])
        except (TypeError, ValueError):
            self.adx_threshold = 25.0

        try:
            quantity = float(merged["quantity"])
        except (TypeError, ValueError):
            quantity = 1.0

        self.quantity = quantity if quantity > 0.0 else 1.0

        try:
            scale_out = float(merged["scale_out_fraction"])
        except (TypeError, ValueError):
            scale_out = 0.5

        # Una fraccion >= 1 convertiria el cross de 30m en un cierre
        # completo, que es lo que ya hace el reverso de 1H, y una <= 0 no
        # cerraria nada. En los dos casos el valor cae a la mitad.
        self.scale_out_fraction = scale_out if 0.0 < scale_out < 1.0 else 0.5

    def evaluate(self, state: EngineState):
        # ---------------------------------------------------------
        # Obtener timeframes
        # ---------------------------------------------------------

        tf_1h = state.timeframes.get(self.tf_trend)
        tf_30m = state.timeframes.get(self.tf_confirm)
        tf_15m = state.timeframes.get(self.tf_pullback)
        tf_5m = state.timeframes.get(self.tf_setup)
        tf_1m = state.timeframes.get(self.tf_trigger)

        if (
            tf_1h is None
            or tf_30m is None
            or tf_15m is None
            or tf_5m is None
            or tf_1m is None
        ):
            return None

        # ---------------------------------------------------------
        # 1H - TENDENCIA PRINCIPAL
        #
        # LONG:
        # EMA20 > EMA50
        # ADX >= threshold
        # +DI > -DI
        #
        # SHORT:
        # EMA20 < EMA50
        # ADX >= threshold
        # -DI > +DI
        # ---------------------------------------------------------

        ema20_1h = self._get_series(
            tf_1h,
            "EMA",
            self.label_fast,
        )

        ema50_1h = self._get_series(
            tf_1h,
            "EMA",
            self.label_slow,
        )

        adx_1h = self._get_series(
            tf_1h,
            "ADX",
            self.adx_label,
        )

        if ema20_1h is None or ema50_1h is None or adx_1h is None:
            return None

        ema20_1h_value = self._ema_closed_value(ema20_1h)
        ema50_1h_value = self._ema_closed_value(ema50_1h)

        adx_value = self._adx_confirmed_value(
            adx_1h,
            tf_1h,
            "adx",
        )

        plus_di = self._adx_confirmed_value(
            adx_1h,
            tf_1h,
            "plus_di",
        )

        minus_di = self._adx_confirmed_value(
            adx_1h,
            tf_1h,
            "minus_di",
        )

        if (
            ema20_1h_value is None
            or ema50_1h_value is None
            or adx_value is None
            or plus_di is None
            or minus_di is None
        ):
            return None

        bullish_1h = (
            ema20_1h_value > ema50_1h_value
            and adx_value >= self.adx_threshold
            and plus_di > minus_di
        )

        bearish_1h = (
            ema20_1h_value < ema50_1h_value
            and adx_value >= self.adx_threshold
            and minus_di > plus_di
        )

        # ---------------------------------------------------------
        # Portfolio
        # ---------------------------------------------------------

        portfolio = state.portfolio
        position = portfolio.position if portfolio is not None else None

        # ---------------------------------------------------------
        # Si existe posición:
        #
        # 1H en contra      -> cierre completo.
        # 1H igual, 30m en contra -> salida parcial.
        # En otro caso      -> mantener.
        # ---------------------------------------------------------

        if position is not None:

            if position.side == Side.BUY and bearish_1h:
                return Signal(action="EXIT")

            if position.side == Side.SELL and bullish_1h:
                return Signal(action="EXIT")

            scale_out = self._scale_out_signal(state, position)

            if scale_out is not None:
                return scale_out

            return None

        # ---------------------------------------------------------
        # Si 1H está neutral:
        # NO abrir operaciones.
        # ---------------------------------------------------------

        if not bullish_1h and not bearish_1h:
            return None

        # ---------------------------------------------------------
        # 30m - CONFIRMACIÓN
        #
        # LONG:
        # EMA20 > EMA50
        #
        # SHORT:
        # EMA20 < EMA50
        #
        # Se lee con el mismo helper que usa la salida parcial, para
        # que el cross que abre y el que sale se juzguen con la misma
        # regla y no puedan discrepar.
        # ---------------------------------------------------------

        structure_30m = self._ema_structure(tf_30m)

        if structure_30m is None:
            return None

        bullish_30m, bearish_30m = structure_30m

        # ---------------------------------------------------------
        # 15m - PULLBACK + TENDENCIA
        #
        # IMPORTANTE:
        # Aquí SI usamos EMA20 Y EMA50.
        #
        # LONG:
        # EMA20 > EMA50
        # Precio cerrado <= EMA20
        #
        # SHORT:
        # EMA20 < EMA50
        # Precio cerrado >= EMA20
        # ---------------------------------------------------------

        ema20_15m = self._get_series(
            tf_15m,
            "EMA",
            self.label_fast,
        )

        ema50_15m = self._get_series(
            tf_15m,
            "EMA",
            self.label_slow,
        )

        if ema20_15m is None or ema50_15m is None:
            return None

        ema20_15m_value = self._ema_closed_value(ema20_15m)
        ema50_15m_value = self._ema_closed_value(ema50_15m)

        close_15m = self._closed_close(tf_15m)

        if (
            ema20_15m_value is None
            or ema50_15m_value is None
            or close_15m is None
        ):
            return None

        bullish_15m = ema20_15m_value > ema50_15m_value
        bearish_15m = ema20_15m_value < ema50_15m_value

        long_pullback = (
            bullish_15m
            and close_15m <= ema20_15m_value
        )

        short_pullback = (
            bearish_15m
            and close_15m >= ema20_15m_value
        )

        # ---------------------------------------------------------
        # 5m - SETUP
        #
        # LONG:
        # EMA20 > EMA50
        #
        # SHORT:
        # EMA20 < EMA50
        # ---------------------------------------------------------

        ema20_5m = self._get_series(
            tf_5m,
            "EMA",
            self.label_fast,
        )

        ema50_5m = self._get_series(
            tf_5m,
            "EMA",
            self.label_slow,
        )

        if ema20_5m is None or ema50_5m is None:
            return None

        ema20_5m_value = self._ema_closed_value(ema20_5m)
        ema50_5m_value = self._ema_closed_value(ema50_5m)

        if ema20_5m_value is None or ema50_5m_value is None:
            return None

        bullish_5m = ema20_5m_value > ema50_5m_value
        bearish_5m = ema20_5m_value < ema50_5m_value

        # ---------------------------------------------------------
        # 1m - TRIGGER
        #
        # LONG:
        # EMA20 > EMA50
        # Cierre de la vela cerrada > EMA20
        # Cierre rompe el máximo de la vela anterior
        #
        # SHORT:
        # EMA20 < EMA50
        # Cierre de la vela cerrada < EMA20
        # Cierre rompe el mínimo de la vela anterior
        # ---------------------------------------------------------

        ema20_1m = self._get_series(
            tf_1m,
            "EMA",
            self.label_fast,
        )

        ema50_1m = self._get_series(
            tf_1m,
            "EMA",
            self.label_slow,
        )

        if ema20_1m is None or ema50_1m is None:
            return None

        ema20_1m_value = self._ema_closed_value(ema20_1m)
        ema50_1m_value = self._ema_closed_value(ema50_1m)

        # ---------------------------------------------------------
        # Candlestick 1m
        #
        # history[-1] = última vela cerrada
        # history[-2] = vela cerrada inmediatamente anterior
        # ---------------------------------------------------------

        current_1m, previous_1m = self._get_last_two_candles(tf_1m)

        if (
            ema20_1m_value is None
            or ema50_1m_value is None
            or current_1m is None
            or previous_1m is None
        ):
            return None

        close_1m = current_1m.close
        previous_high_1m = previous_1m.high
        previous_low_1m = previous_1m.low

        # ---------------------------------------------------------
        # Trigger LONG
        # ---------------------------------------------------------

        bullish_1m = (
            ema20_1m_value > ema50_1m_value
            and close_1m > ema20_1m_value
            and close_1m > previous_high_1m
        )

        # ---------------------------------------------------------
        # Trigger SHORT
        # ---------------------------------------------------------

        bearish_1m = (
            ema20_1m_value < ema50_1m_value
            and close_1m < ema20_1m_value
            and close_1m < previous_low_1m
        )

        # ---------------------------------------------------------
        # ENTRADA LONG
        # ---------------------------------------------------------

        if (
            bullish_1h
            and bullish_30m
            and long_pullback
            and bullish_5m
            and bullish_1m
        ):
            return Signal(
                action="BUY",
                quantity=self.quantity,
            )

        # ---------------------------------------------------------
        # ENTRADA SHORT
        # ---------------------------------------------------------

        if (
            bearish_1h
            and bearish_30m
            and short_pullback
            and bearish_5m
            and bearish_1m
        ):
            return Signal(
                action="SELL",
                quantity=self.quantity,
            )

        return None

    # =============================================================
    # SCALE OUT
    # =============================================================

    def _scale_out_signal(self, state, position):
        """
        Salida parcial cuando el cross de 30m va contra la posición.

        Devuelve None cuando no hay que salir, y la señal con la
        cantidad cuando sí.

        La estrategia es sin estado a propósito: el portfolio es la
        única fuente de verdad y el motor restaura exactamente el estado
        que publica. Por eso la idempotencia no se guarda en un flag
        sino que se deduce de la propia posición.

        Si no, el 30m en contra se leería en cada tick y la posición
        menguaría 1.0 -> 0.5 -> 0.25 -> 0.125 ... hasta quedar en polvo,
        que es una pérdida por goteo y no una salida parcial.

        La condición es que la posición siga completa: en cuanto una
        salida (parcial o de bracket) la reduce, ya no vuelve a
        escalar. Un target que llenó antes también cuenta como salida
        ya realizada, que es lo conservador.
        """

        if position.quantity < self.quantity - QUANTITY_EPSILON:
            return None

        tf_30m = state.timeframes.get(self.tf_confirm)

        if tf_30m is None:
            return None

        structure = self._ema_structure(tf_30m)

        if structure is None:
            return None

        bullish_30m, bearish_30m = structure

        if position.side == Side.BUY and bearish_30m:
            against = True
        elif position.side == Side.SELL and bullish_30m:
            against = True
        else:
            against = False

        if not against:
            return None

        return Signal(
            action="EXIT",
            quantity=position.quantity * self.scale_out_fraction,
        )

    def _ema_structure(self, timeframe):
        """
        (alcista, bajista) según el cruce de las dos EMA en el
        timeframe, o None si todavía no se pueden leer.

        Estar en None es distinto de estar en neutral: neutral es una
        decisión, None es "todavía no hay datos" y no debe abrir ni
        cerrar nada.
        """

        ema_fast = self._get_series(timeframe, "EMA", self.label_fast)
        ema_slow = self._get_series(timeframe, "EMA", self.label_slow)

        if ema_fast is None or ema_slow is None:
            return None

        fast = self._ema_closed_value(ema_fast)
        slow = self._ema_closed_value(ema_slow)

        if fast is None or slow is None:
            return None

        return fast > slow, fast < slow

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
    def _get_series(timeframe, kind: str, label: str):
        if timeframe is None:
            return None

        try:
            return timeframe.get_series(kind, label)
        except KeyError:
            return None

    @staticmethod
    def _ema_closed_value(series):
        """
        EMA:
        _closed = último valor confirmado.
        """

        if series is None:
            return None

        closed = getattr(series, "_closed", None)

        if closed is None:
            return None

        return closed.value

    @staticmethod
    def _adx_confirmed_value(series, timeframe, field: str):
        """
        ADX no tiene _closed como EMA.

        Si ADX.live corresponde a la última vela cerrada,
        usamos live.

        Si live corresponde a la vela actualmente abierta,
        usamos history[-1].
        """

        if series is None:
            return None

        closed_bar = getattr(timeframe, "closed", None)
        live = getattr(series, "live", None)

        if closed_bar is not None and live is not None:

            live_start_ts = getattr(
                live,
                "start_ts",
                None,
            )

            closed_start_ts = getattr(
                closed_bar,
                "start_ts",
                None,
            )

            if (
                live_start_ts is not None
                and closed_start_ts is not None
                and live_start_ts == closed_start_ts
            ):
                return getattr(
                    live,
                    field,
                    None,
                )

        history = getattr(series, "history", None)

        if not history:
            return None

        return getattr(
            history[-1],
            field,
            None,
        )

    @staticmethod
    def _closed_close(timeframe):
        """
        Devuelve el cierre de la última vela cerrada.
        """

        if timeframe is None:
            return None

        candle = getattr(
            timeframe,
            "closed",
            None,
        )

        if candle is None:
            return None

        return getattr(
            candle,
            "close",
            None,
        )

    @staticmethod
    def _get_last_two_candles(timeframe):
        """
        Devuelve:

        current  = history[-1] -> última vela cerrada
        previous = history[-2] -> vela cerrada anterior
        """

        if timeframe is None:
            return None, None

        for series in getattr(timeframe, "_series", {}).values():

            if getattr(series, "kind", None) != "Candlestick":
                continue

            history = getattr(series, "history", None)

            if history is None or len(history) < 2:
                return None, None

            return history[-1], history[-2]

        return None, None    