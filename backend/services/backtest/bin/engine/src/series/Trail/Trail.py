from collections import deque
from dataclasses import asdict, dataclass
from series.series import Series


MAX_HISTORY = 500


@dataclass(frozen=True)
class TrailValue:
    time: int

    # Estado principal
    trend: int

    # Señales
    up_signal: bool
    down_signal: bool

    # Core
    basis: float
    atr: float
    safe_atr: float

    momentum: float
    distance: float

    efficiency: float
    chop: float
    efficiency_field: float

    # Volatilidad
    normal_atr: float
    vol_ratio: float
    vol_expansion: float
    vol_deviation: float

    # Supertrend
    fast_adaptive_factor: float
    mid_adaptive_factor: float
    slow_adaptive_factor: float

    st_fast: float
    st_mid: float
    st_slow: float

    st_fast_direction: float
    st_mid_direction: float
    st_slow_direction: float

    bull_st_votes: int
    bear_st_votes: int

    bull_st_bars: int
    bear_st_bars: int

    # Regime
    distance_field: float
    momentum_field: float
    slope_field: float
    rsi: float
    rsi_field: float

    st_field: float
    slow_slope_field: float

    structure_field: float
    pressure_field: float

    raw_regime: float
    regime: float
    dynamic_gate: float

    # Confirmation
    bull_candidate: bool
    bear_candidate: bool

    bull_confirm: int
    bear_confirm: int

    bull_persistent: bool
    bear_persistent: bool

    strong_bull: bool
    strong_bear: bool

    bull_takeover: bool
    bear_takeover: bool

    bull_ready: bool
    bear_ready: bool

    # Overlay
    smooth_basis: float
    smooth_atr: float

    inner_trail: float
    outer_trail: float
    depth_trail: float

    # Valuation
    valuation: float


def _get_param(params: dict, key: str, default):
    """The frontend can send a parameter either as a plain value or as a
    parameter descriptor object carrying the value (``affectsCompute``)."""
    value = params.get(key, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    return value


class Trail(Series):

    def __init__(
        self,
        id: str,
        kind: str,
        level: int,
        params: dict,
        parent_id: str | None,
    ):
        super().__init__(
            id,
            kind,
            level,
            params,
            parent_id,
        )

        # =====================================================
        # INPUTS — MISMAS CONFIGURACIONES DEL PINE
        # =====================================================

        self.trend_length = int(
            _get_param(params, "trend_length", 34)
        )

        self.momentum_length = int(
            _get_param(params, "momentum_length", 12)
        )

        self.sensitivity = float(
            _get_param(params, "sensitivity", 0.35)
        )

        self.st_fast_length = int(
            _get_param(params, "st_fast_length", 9)
        )

        self.st_fast_factor = float(
            _get_param(params, "st_fast_factor", 1.45)
        )

        self.st_mid_length = int(
            _get_param(params, "st_mid_length", 14)
        )

        self.st_mid_factor = float(
            _get_param(params, "st_mid_factor", 1.95)
        )

        self.st_slow_length = int(
            _get_param(params, "st_slow_length", 21)
        )

        self.st_slow_factor = float(
            _get_param(params, "st_slow_factor", 2.55)
        )

        self.trail_size = float(
            _get_param(params, "trail_size", 1.00)
        )

        self.smoothness = int(
            _get_param(params, "smoothness", 5)
        )

        # =====================================================
        # STATE
        # =====================================================

        self._live: TrailValue | None = None
        self._closed: TrailValue | None = None

        self.history: deque[TrailValue] = deque(
            maxlen=MAX_HISTORY
        )

        # =====================================================
        # INTERNAL STATE
        #
        # Estos estados pertenecen a la máquina de señales.
        # =====================================================

        self._trend = 0

        self._last_flip_time: int | None = None

        self._bull_st_bars = 0
        self._bear_st_bars = 0

        self._bull_confirm = 0
        self._bear_confirm = 0

        # =====================================================
        # SERIES INTERNAS
        #
        # Se mantienen separadas para poder reproducir
        # exactamente el comportamiento secuencial de Pine.
        # =====================================================

        self._basis_closed: float | None = None
        self._basis_history = deque(maxlen=MAX_HISTORY)

        self._atr_closed: float | None = None
        self._atr_history = deque(maxlen=MAX_HISTORY)

        self._rsi_closed: float | None = None
        self._rsi_history = deque(maxlen=MAX_HISTORY)

        self._regime_closed: float | None = None
        self._regime_history = deque(maxlen=MAX_HISTORY)

        self._smooth_basis_closed: float | None = None
        self._smooth_atr_closed: float | None = None

        self._normal_atr_closed: float | None = None

        # =====================================================
        # HISTORIAL DE CLOSE / OHLC
        # =====================================================

        self._closes = deque(maxlen=MAX_HISTORY)
        self._highs = deque(maxlen=MAX_HISTORY)
        self._lows = deque(maxlen=MAX_HISTORY)
        self._opens = deque(maxlen=MAX_HISTORY)

    @property
    def live(self) -> TrailValue | None:
        return self._live

    @live.setter
    def live(self, value: TrailValue | None):
        self._live = value

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(maximum, value),
        )

    @staticmethod
    def _ema(
        value: float,
        previous: float | None,
        period: int,
    ) -> float:

        alpha = 2.0 / (
            period + 1.0
        )

        if previous is None:
            return value

        return (
            alpha * value
            + (1.0 - alpha) * previous
        )

    @staticmethod
    def _rma(
        value: float,
        previous: float | None,
        period: int,
    ) -> float:

        alpha = 1.0 / period

        if previous is None:
            return value

        return (
            alpha * value
            + (1.0 - alpha) * previous
        )

    # =========================================================
    # EMA
    # =========================================================

    def _calculate_ema(
        self,
        value: float,
        previous: float | None,
        period: int,
    ) -> float:

        return self._ema(
            value,
            previous,
            period,
        )

    # =========================================================
    # ATR
    # =========================================================

    def _calculate_true_range(
        self,
        high: float,
        low: float,
        previous_close: float | None,
    ) -> float:

        if previous_close is None:
            return high - low

        return max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )

    def _calculate_atr(
        self,
        high: float,
        low: float,
        close: float,
        previous_close: float | None,
        previous_atr: float | None,
    ) -> float:

        tr = self._calculate_true_range(
            high,
            low,
            previous_close,
        )

        return self._rma(
            tr,
            previous_atr,
            14,
        )

    # =========================================================
    # MOMENTUM
    # =========================================================

    def _calculate_momentum(
        self,
        close: float,
        close_n: float | None,
        previous_momentum_ema: float | None,
        safe_atr: float,
    ) -> float:

        if close_n is None:
            return float("nan")

        raw = close - close_n

        smoothed = self._ema(
            raw,
            previous_momentum_ema,
            5,
        )

        return smoothed / safe_atr

    # =========================================================
    # EFFICIENCY
    # =========================================================

    def _calculate_efficiency(
        self,
        close: float,
        previous_closes: list[float],
    ) -> tuple[float, float, float]:

        if len(previous_closes) < 10:
            return (
                0.0,
                0.0,
                0.0,
            )

        old_close = previous_closes[-10]

        net_movement = abs(
            close - old_close
        )

        travel_path = 0.0

        # Equivalente a:
        #
        # math.sum(
        #     math.abs(close - close[1]),
        #     10
        # )
        #
        # utilizando las últimas 10 diferencias.

        values = previous_closes[-10:]

        if len(values) >= 2:

            for i in range(1, len(values)):
                travel_path += abs(
                    values[i] - values[i - 1]
                )

            travel_path += abs(
                close - values[-1]
            )

        if travel_path > 0.0:
            efficiency = self._clamp(
                net_movement / travel_path,
                0.0,
                1.0,
            )
        else:
            efficiency = 0.0

        if close > old_close:
            direction = 1.0
        elif close < old_close:
            direction = -1.0
        else:
            direction = 0.0

        efficiency_field = (
            efficiency * direction
        )

        return (
            efficiency,
            1.0 - efficiency,
            efficiency_field,
        )

    # =========================================================
    # ACCESSORS
    # =========================================================

    def to_dict(self) -> dict:

        return {
            "id": self.id,
            "kind": self.kind,
            "level": self.level,
            "params": self.params,
            "parent_id": self.parent_id,

            "live": (
                asdict(self.live)
                if self.live is not None
                else None
            ),

            "history": [
                asdict(value)
                for value in self.history
            ],
        }

    # =========================================================
    # STATE RESTORE
    # =========================================================

    def set_state(
        self,
        state: dict,
    ) -> None:

        live_state = state.get("live")

        self.live = (
            TrailValue(**live_state)
            if live_state is not None
            else None
        )

        history = [
            TrailValue(**value)
            for value in (
                state.get("history") or []
            )
        ]

        self.history = deque(
            history,
            maxlen=MAX_HISTORY,
        )

        self._closed = (
            history[-1]
            if history
            else None
        )

        # -----------------------------------------------------
        # Restaurar máquina de estado desde el último valor
        # cerrado.
        # -----------------------------------------------------

        if self._closed is not None:

            self._trend = (
                self._closed.trend
            )

            self._bull_st_bars = (
                self._closed.bull_st_bars
            )

            self._bear_st_bars = (
                self._closed.bear_st_bars
            )

            self._bull_confirm = (
                self._closed.bull_confirm
            )

            self._bear_confirm = (
                self._closed.bear_confirm
            )

            self._basis_closed = (
                self._closed.basis
            )

            self._atr_closed = (
                self._closed.atr
            )

            self._rsi_closed = (
                self._closed.rsi
            )

            self._regime_closed = (
                self._closed.regime
            )

            self._smooth_basis_closed = (
                self._closed.smooth_basis
            )

            self._smooth_atr_closed = (
                self._closed.smooth_atr
            )

            self._normal_atr_closed = (
                self._closed.normal_atr
            )