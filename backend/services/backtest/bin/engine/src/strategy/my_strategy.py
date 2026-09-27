from core.engine_state import EngineState
from strategy.base import Strategy
from orders import Signal, Side


class Strategy1(Strategy):
    """
    Estrategia MTF de continuación de tendencia.

    ================================================================
    1H - REGIMEN
    ================================================================

        EMA rápida > EMA lenta
        ADX >= threshold
        +DI > -DI

            => Tendencia alcista

        EMA rápida < EMA lenta
        ADX >= threshold
        -DI > +DI

            => Tendencia bajista

    ================================================================
    30M - CONFIRMACION
    ================================================================

        LONG  -> EMA20 > EMA50
        SHORT -> EMA20 < EMA50

    ================================================================
    15M - PULLBACK
    ================================================================

        LONG  -> precio <= EMA20
        SHORT -> precio >= EMA20

    ================================================================
    5M - SETUP
    ================================================================

        LONG  -> EMA20 > EMA50
        SHORT -> EMA20 < EMA50

    ================================================================
    1M - TRIGGER
    ================================================================

        LONG  -> EMA20 > EMA50
        SHORT -> EMA20 < EMA50

    ================================================================
    FILOSOFIA
    ================================================================

    1H manda.

    Los timeframes inferiores NO pueden cambiar el régimen.

    Si 1H es alcista:
        solamente se buscan LONG.

    Si 1H es bajista:
        solamente se buscan SHORT.

    Si ADX está por debajo del threshold:
        no se abren nuevas posiciones.

    El RiskManager continúa gestionando SL / TP / trailing.
    """

    DEFAULT_PARAMS = {

        # =========================================================
        # 1H
        # =========================================================

        "trend_timeframe": "1h",

        "label_fast_1h": "EMA 20",
        "label_slow_1h": "EMA 50",

        "label_adx_1h": "ADX 14",

        "adx_threshold": 25.0,

        # =========================================================
        # 30M
        # =========================================================

        "confirm_timeframe": "30m",

        "label_fast_30m": "EMA 20",
        "label_slow_30m": "EMA 50",

        # =========================================================
        # 15M
        # =========================================================

        "pullback_timeframe": "15m",

        "label_fast_15m": "EMA 20",

        # =========================================================
        # 5M
        # =========================================================

        "setup_timeframe": "5m",

        "label_fast_5m": "EMA 20",
        "label_slow_5m": "EMA 50",

        # =========================================================
        # 1M
        # =========================================================

        "trigger_timeframe": "1m",

        "label_fast_1m": "EMA 20",
        "label_slow_1m": "EMA 50",

        # =========================================================
        # ORDER
        # =========================================================

        "quantity": 1,
    }

    def __init__(self, kind: str, params: dict):

        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        # ---------------------------------------------------------
        # Timeframes
        # ---------------------------------------------------------

        self.trend_timeframe = str(
            merged["trend_timeframe"]
        )

        self.confirm_timeframe = str(
            merged["confirm_timeframe"]
        )

        self.pullback_timeframe = str(
            merged["pullback_timeframe"]
        )

        self.setup_timeframe = str(
            merged["setup_timeframe"]
        )

        self.trigger_timeframe = str(
            merged["trigger_timeframe"]
        )

        # ---------------------------------------------------------
        # 1H
        # ---------------------------------------------------------

        self.label_fast_1h = str(
            merged["label_fast_1h"]
        )

        self.label_slow_1h = str(
            merged["label_slow_1h"]
        )

        self.label_adx_1h = str(
            merged["label_adx_1h"]
        )

        try:
            self.adx_threshold = float(
                merged["adx_threshold"]
            )
        except (TypeError, ValueError):
            self.adx_threshold = 25.0

        # ---------------------------------------------------------
        # 30M
        # ---------------------------------------------------------

        self.label_fast_30m = str(
            merged["label_fast_30m"]
        )

        self.label_slow_30m = str(
            merged["label_slow_30m"]
        )

        # ---------------------------------------------------------
        # 15M
        # ---------------------------------------------------------

        self.label_fast_15m = str(
            merged["label_fast_15m"]
        )

        # ---------------------------------------------------------
        # 5M
        # ---------------------------------------------------------

        self.label_fast_5m = str(
            merged["label_fast_5m"]
        )

        self.label_slow_5m = str(
            merged["label_slow_5m"]
        )

        # ---------------------------------------------------------
        # 1M
        # ---------------------------------------------------------

        self.label_fast_1m = str(
            merged["label_fast_1m"]
        )

        self.label_slow_1m = str(
            merged["label_slow_1m"]
        )

        # ---------------------------------------------------------
        # Quantity
        # ---------------------------------------------------------

        try:
            quantity = float(
                merged["quantity"]
            )
        except (TypeError, ValueError):
            quantity = 1.0

        self.quantity = (
            quantity
            if quantity > 0.0
            else 1.0
        )

    # =============================================================
    # EVALUATE
    # =============================================================

    def evaluate(self, state: EngineState):

        # =========================================================
        # 1H - TENDENCIA
        # =========================================================

        trend_tf = state.timeframes.get(
            self.trend_timeframe
        )

        if trend_tf is None:
            return None

        ema_fast_1h = self._get_series(
            trend_tf,
            "EMA",
            self.label_fast_1h,
        )

        ema_slow_1h = self._get_series(
            trend_tf,
            "EMA",
            self.label_slow_1h,
        )

        adx_1h = self._get_series(
            trend_tf,
            "ADX",
            self.label_adx_1h,
        )

        if (
            ema_fast_1h is None
            or ema_slow_1h is None
            or adx_1h is None
        ):
            return None

        # =========================================================
        # Usamos valores CONFIRMADOS.
        #
        # EMA:
        #     _closed
        #
        # ADX:
        #     history[-1]
        #
        # No usamos el valor live para determinar la tendencia.
        # =========================================================

        fast_1h = self._ema_closed_value(
            ema_fast_1h
        )

        slow_1h = self._ema_closed_value(
            ema_slow_1h
        )

        adx_value = self._adx_closed_value(
            adx_1h,
            "adx",
        )

        plus_di = self._adx_closed_value(
            adx_1h,
            "plus_di",
        )

        minus_di = self._adx_closed_value(
            adx_1h,
            "minus_di",
        )

        if any(
            value is None
            for value in (
                fast_1h,
                slow_1h,
                adx_value,
                plus_di,
                minus_di,
            )
        ):
            return None

        # =========================================================
        # REGIMEN
        # =========================================================

        trend_bullish = (
            fast_1h > slow_1h
            and adx_value >= self.adx_threshold
            and plus_di > minus_di
        )

        trend_bearish = (
            fast_1h < slow_1h
            and adx_value >= self.adx_threshold
            and minus_di > plus_di
        )

        # =========================================================
        # POSITION
        # =========================================================

        portfolio = state.portfolio

        position = (
            portfolio.position
            if portfolio is not None
            else None
        )

        # =========================================================
        # EXIT
        #
        # Unicamente cambiamos de posicion cuando 1H confirma
        # el regimen contrario.
        # =========================================================

        if position is not None:

            if (
                position.side == Side.BUY
                and trend_bearish
            ):
                return Signal(
                    action="EXIT"
                )

            if (
                position.side == Side.SELL
                and trend_bullish
            ):
                return Signal(
                    action="EXIT"
                )

            # Nunca abrimos una segunda posicion.
            return None

        # =========================================================
        # SIN TENDENCIA
        # =========================================================

        if not trend_bullish and not trend_bearish:
            return None

        # =========================================================
        # 30M - CONFIRMACION
        # =========================================================

        confirm_tf = state.timeframes.get(
            self.confirm_timeframe
        )

        if confirm_tf is None:
            return None

        ema_fast_30m = self._get_series(
            confirm_tf,
            "EMA",
            self.label_fast_30m,
        )

        ema_slow_30m = self._get_series(
            confirm_tf,
            "EMA",
            self.label_slow_30m,
        )

        fast_30m = self._ema_closed_value(
            ema_fast_30m
        )

        slow_30m = self._ema_closed_value(
            ema_slow_30m
        )

        if (
            fast_30m is None
            or slow_30m is None
        ):
            return None

        confirm_bullish = (
            fast_30m > slow_30m
        )

        confirm_bearish = (
            fast_30m < slow_30m
        )

        # La tendencia del 30m debe coincidir
        # con la tendencia del 1H.

        if (
            trend_bullish
            and not confirm_bullish
        ):
            return None

        if (
            trend_bearish
            and not confirm_bearish
        ):
            return None

        # =========================================================
        # 15M - PULLBACK
        # =========================================================

        pullback_tf = state.timeframes.get(
            self.pullback_timeframe
        )

        if pullback_tf is None:
            return None

        ema_15m = self._get_series(
            pullback_tf,
            "EMA",
            self.label_fast_15m,
        )

        ema_15m_value = self._ema_closed_value(
            ema_15m
        )

        close_15m = self._closed_close(
            pullback_tf
        )

        if (
            ema_15m_value is None
            or close_15m is None
        ):
            return None

        # ---------------------------------------------------------
        # LONG
        # ---------------------------------------------------------

        if trend_bullish:

            pullback_valid = (
                close_15m <= ema_15m_value
            )

        # ---------------------------------------------------------
        # SHORT
        # ---------------------------------------------------------

        else:

            pullback_valid = (
                close_15m >= ema_15m_value
            )

        if not pullback_valid:
            return None

        # =========================================================
        # 5M - SETUP
        # =========================================================

        setup_tf = state.timeframes.get(
            self.setup_timeframe
        )

        if setup_tf is None:
            return None

        ema_fast_5m = self._get_series(
            setup_tf,
            "EMA",
            self.label_fast_5m,
        )

        ema_slow_5m = self._get_series(
            setup_tf,
            "EMA",
            self.label_slow_5m,
        )

        fast_5m = self._ema_closed_value(
            ema_fast_5m
        )

        slow_5m = self._ema_closed_value(
            ema_slow_5m
        )

        if (
            fast_5m is None
            or slow_5m is None
        ):
            return None

        setup_bullish = (
            fast_5m > slow_5m
        )

        setup_bearish = (
            fast_5m < slow_5m
        )

        if (
            trend_bullish
            and not setup_bullish
        ):
            return None

        if (
            trend_bearish
            and not setup_bearish
        ):
            return None

        # =========================================================
        # 1M - TRIGGER
        # =========================================================

        trigger_tf = state.timeframes.get(
            self.trigger_timeframe
        )

        if trigger_tf is None:
            return None

        ema_fast_1m = self._get_series(
            trigger_tf,
            "EMA",
            self.label_fast_1m,
        )

        ema_slow_1m = self._get_series(
            trigger_tf,
            "EMA",
            self.label_slow_1m,
        )

        fast_1m = self._ema_closed_value(
            ema_fast_1m
        )

        slow_1m = self._ema_closed_value(
            ema_slow_1m
        )

        if (
            fast_1m is None
            or slow_1m is None
        ):
            return None

        trigger_bullish = (
            fast_1m > slow_1m
        )

        trigger_bearish = (
            fast_1m < slow_1m
        )

        # =========================================================
        # LONG
        # =========================================================

        if (
            trend_bullish
            and confirm_bullish
            and pullback_valid
            and setup_bullish
            and trigger_bullish
        ):
            return Signal(
                action="BUY",
                quantity=self.quantity,
            )

        # =========================================================
        # SHORT
        # =========================================================

        if (
            trend_bearish
            and confirm_bearish
            and pullback_valid
            and setup_bearish
            and trigger_bearish
        ):
            return Signal(
                action="SELL",
                quantity=self.quantity,
            )

        return None

    # =============================================================
    # EMA HELPERS
    # =============================================================

    @staticmethod
    def _ema_closed_value(series):

        if series is None:
            return None

        # Tu EMA guarda la ultima vela confirmada
        # en _closed.

        closed = getattr(
            series,
            "_closed",
            None,
        )

        if closed is None:
            return None

        return closed.value

    # =============================================================
    # ADX HELPERS
    # =============================================================

    @staticmethod
    def _adx_closed_value(
        series,
        field,
    ):

        if series is None:
            return None

        # ADX no tiene .closed.
        #
        # Su ultimo valor confirmado esta en:
        #
        #     history[-1]
        #
        # El objeto Adx contiene:
        #
        #     adx
        #     plus_di
        #     minus_di

        if not series.history:
            return None

        value = series.history[-1]

        return getattr(
            value,
            field,
            None,
        )

    # =============================================================
    # BAR
    # =============================================================

    @staticmethod
    def _closed_close(timeframe):

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

    # =============================================================
    # SERIES
    # =============================================================

    @staticmethod
    def _get_series(
        timeframe,
        kind: str,
        label: str,
    ):

        if timeframe is None:
            return None

        try:
            return timeframe.get_series(
                kind,
                label,
            )
        except KeyError:
            return None

    # =============================================================
    # PARAMS
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