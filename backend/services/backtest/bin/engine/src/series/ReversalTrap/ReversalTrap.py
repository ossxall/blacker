# Reversal Trap
#
# Engine-compatible Series port of the Script  indicator whose
# reference lives in `reversal`. The mathematical logic is kept as
# close to 1:1 with as the engine's bar-driven model allows:
#
#   - basis = ta.ema(close, envelope_len)          (SMA-seeded EMA)
#   - vola  = ta.atr(envelope_len)                 (RMA of true range)
#   - rsi   = ta.rsi(close, rsi_len)               (Wilder up/down RMA)
#   - upper_band = basis + multiplier * vola
#   - lower_band = basis - multiplier * vola
#   - rsi_bucket = max(0, min(10, round(rsi / 10)))
#   - atr        = ta.atr(atr_len) * stop_mult     (atr_len = 100)
#   - raw_bear/raw_bull trap detection with trap_window counters
#   - signal_gap cooldown on last_signal_bar
#   - independent active_bull / active_bear trackers
#   - target pinned to basis (or upper/lower band)
#   - stop pinned to lowest/highest(hl, 2) +/- atr at the signal bar
#   - per-bucket (RSI) win/loss database + total_trades_count guardrails
#
# Engine adaptations:
#   * `bars_from_live = last_bar_index - bar_index` is not known
#     to a streaming series, so the engine always operates at the live
#     edge (`bars_from_live == 0`); `max_bars` therefore has no effect
#     and `max_trades` is the active guardrail.
#   * The label/line drawing primitives are dropped; the trap flags,
#     active trackers and target/stop prices are exposed as data so
#     downstream consumers can build signals themselves.
#   * The chain (EMA / ATR / RSI) is advanced once per closed bar and the
#     forming bar offers a provisional value, mirroring the EMA series.
#
# Warm-up: the exposed `live` stays None until every chain used by the
# indicator is seeded (EMA, ATR(envelope), RSI and ATR(stop) incl. the
# hardcoded atr_len = 100), matching the ADX series behaviour.

from collections import deque
from dataclasses import asdict, dataclass
import math

from series.series import Series


MAX_HISTORY = 500

# `math.round`: half away from zero.
def _round(x: float) -> int:
    if x >= 0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


def _feed_chain(acc_sum: float, count: int, value, x: float, length: int, alpha: float):
    """
    Incremental chain (ta.ema / ta.rma).

    Both built-ins share the same seeding convention:
    while the chain is not seeded, an SMA of the first `length` samples
    is accumulated; once seeded, the recursive formula applies.

        ta.ema:  alpha = 2 / (length + 1)
        ta.rma:  alpha = 1 / length
    """
    if count < length:
        acc_sum += x
        count += 1
        if count == length:
            return acc_sum, count, acc_sum / length
        return acc_sum, count, None
    return acc_sum, count, alpha * x + (1.0 - alpha) * value


def _true_range(prev_close, high: float, low: float) -> float:
    if prev_close is None:
        return high - low
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


@dataclass(frozen=True)
class ReversalTrapValue:
    time: int
    start_ts: int
    end_ts: int
    bar_index: int

    # ----------------------------------------------------------------
    # Public output (bands / momentum)
    # ----------------------------------------------------------------
    basis: float | None
    upper_band: float | None
    lower_band: float | None
    atr_envelope: float | None
    rsi: float | None
    rsi_bucket: int | None
    atr_stop: float | None  # ta.atr(atr_len) * stop_mult
    bull_stop_level: float | None  # ta.lowest(low, 2) - atr
    bear_stop_level: float | None  # ta.highest(high, 2) + atr

    # ----------------------------------------------------------------
    # Trap flags / envelope counters
    # ----------------------------------------------------------------
    raw_bull_trap: bool
    raw_bear_trap: bool
    bull_trap: bool
    bear_trap: bool
    close_above_envelope_count: int
    close_below_envelope_count: int

    # ----------------------------------------------------------------
    # Trade tracking
    # ----------------------------------------------------------------
    active_bull: bool
    active_bear: bool
    bull_bucket: int | None
    bear_bucket: int | None
    bull_target_price: float | None
    bull_stop_price: float | None
    bear_target_price: float | None
    bear_stop_price: float | None
    bull_entry_bar: int | None
    bear_entry_bar: int | None
    allowed_by_limits: bool

    # ----------------------------------------------------------------
    # Historical win/loss database (var arrays of 11 buckets)
    # ----------------------------------------------------------------
    bull_total: tuple[int, ...]
    bull_wins: tuple[int, ...]
    bear_total: tuple[int, ...]
    bear_wins: tuple[int, ...]
    total_trades_count: int

    # ----------------------------------------------------------------
    # Internal continuity state (needed to compute the next bar)
    # ----------------------------------------------------------------
    high: float
    low: float
    close: float
    last_signal_bar: int
    ema_value: float | None
    ema_sum: float
    ema_count: int
    atr55_value: float | None
    atr55_sum: float
    atr55_count: int
    atr100_value: float | None
    atr100_sum: float
    atr100_count: int
    rsi_up_value: float | None
    rsi_up_sum: float
    rsi_down_value: float | None
    rsi_down_sum: float
    rsi_count: int


def _get_param(params: dict, key: str, default):
    """The frontend may send a value either as a plain number or as a
    parameter descriptor object carrying the value."""
    value = params.get(key, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    return value


def _zero_arrays() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    zero = (0,) * 11
    return zero, zero, zero, zero


def _inc(t: tuple[int, ...], idx: int) -> tuple[int, ...]:
    lst = list(t)
    lst[idx] += 1
    return tuple(lst)


def _bucket_index(bucket: int | None) -> int:
    return bucket if bucket is not None else 0


class ReversalTrap(Series):
    def __init__(
        self,
        id: str,
        kind: str,
        level: int,
        primary: bool,
        overlay: bool,
        params: dict,
    ):
        super().__init__(
            id,
            kind,
            level,
            primary,
            overlay,
            params,
        )

        self.envelope_len = int(_get_param(params, "envelope_len", 55))
        self.multiplier = float(_get_param(params, "multiplier", 4.0))
        self.trap_window = int(_get_param(params, "trap_window", 10))
        self.signal_gap = int(_get_param(params, "signal_gap", 10))
        self.rsi_len = int(_get_param(params, "rsi_len", 20))
        self.stop_mult = float(_get_param(params, "stop_mult", 0.5))
        self.max_bars = int(_get_param(params, "max_bars", 5000))
        self.max_trades = int(_get_param(params, "max_trades", 500))
        self.atr_len = int(_get_param(params, "atr_len", 100))
        self.target_source = str(_get_param(params, "target_source", "Basis Line"))

        if self.envelope_len <= 0:
            raise ValueError("envelope_len must be greater than 0")
        if self.rsi_len <= 0:
            raise ValueError("rsi_len must be greater than 0")
        if self.atr_len <= 0:
            raise ValueError("atr_len must be greater than 0")
        if self.trap_window <= 0:
            raise ValueError("trap_window must be greater than 0")
        if self.signal_gap <= 0:
            raise ValueError("signal_gap must be greater than 0")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be greater than 0")
        if self.stop_mult <= 0:
            raise ValueError("stop_mult must be greater than 0")

        # First bar index at which every chain used by the indicator is
        # already seeded:
        #   - EMA/ATR(envelope): length - 1
        #   - RSI: rsi_len  (delta starts at bar 1, RMA seeds one bar later)
        #   - ATR(stop): atr_len - 1
        #   - lowest/highest(hl, 2): 1
        self._warmup = max(
            self.envelope_len - 1,
            self.rsi_len,
            self.atr_len - 1,
            1,
        )

        # Last fully evaluated CLOSED bar (never suppressed by warm-up).
        self._closed: ReversalTrapValue | None = None

        # Visible state (None during warm-up).
        self._live: ReversalTrapValue | None = None

        self.history: deque[ReversalTrapValue] = deque(
            maxlen=MAX_HISTORY
        )

    @property
    def live(self) -> ReversalTrapValue | None:
        return self._live

    @live.setter
    def live(self, value: ReversalTrapValue | None):
        self._live = value

    @staticmethod
    def _coerce(state: dict) -> "ReversalTrapValue":
        for key in ("bull_total", "bull_wins", "bear_total", "bear_wins"):
            value = state.get(key)
            if value is not None:
                state[key] = tuple(value)
        return ReversalTrapValue(**state)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "level": self.level,
            "primary": self.primary,
            "overlay": self.overlay,
            "params": self.params,

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

    def set_state(self, state: dict) -> None:
        live_state = state.get("live")

        self.live = (
            self._coerce(dict(live_state))
            if live_state is not None
            else None
        )

        self.history = deque(
            (
                self._coerce(dict(value))
                for value in (state.get("history") or [])
            ),
            maxlen=MAX_HISTORY,
        )

        # The confirmed chain is rebuilt from live, or from the last
        # history entry when live is suppressed by the warm-up period.
        self._closed = (
            self.live
            or (self.history[-1] if self.history else None)
        )

    def update(self) -> None:
        timeframe = self._timeframe

        # --------------------------------------------------
        # Candle cerrada -> avanzar el estado confirmado.
        # --------------------------------------------------

        if timeframe.is_closed:

            closed = timeframe.closed

            if closed is not None:

                value = self._evaluate(
                    self._closed,
                    closed,
                )

                if self._closed is not None:
                    self.history.append(
                        self._closed
                    )

                self._closed = value

        # --------------------------------------------------
        # Candle viva -> estimación provisional.
        # --------------------------------------------------

        candle = timeframe.live

        if candle is None:
            return

        if (
            timeframe.is_closed
            and timeframe.closed is not None
            and candle.start_ts == timeframe.closed.start_ts
        ):
            # Timeframe.flush(): the last bar was just confirmed and no
            # new bar replaced it, so the live value is that final state.
            live = self._closed
        else:
            live = self._evaluate(
                self._closed,
                candle,
            )

        if live is not None and live.bar_index < self._warmup:
            live = None

        self.live = live

    def _evaluate(
        self,
        prev: "ReversalTrapValue | None",
        candle,
    ) -> "ReversalTrapValue":
        high = float(candle.high)
        low = float(candle.low)
        close = float(candle.close)

        # ----------------------------------------------------------------
        # Continuity chain ( `var` values carried bar to bar).
        # ----------------------------------------------------------------

        if prev is None:
            bar_index = 0
            prev_close = None
            prev_high = None
            prev_low = None
            close_above_prev = 0
            close_below_prev = 0
            last_signal_bar = -1000

            active_bull = False
            active_bear = False
            bull_bucket = None
            bear_bucket = None
            bull_target_price = None
            bull_stop_price = None
            bear_target_price = None
            bear_stop_price = None
            bull_entry_bar = None
            bear_entry_bar = None

            bull_total, bull_wins, bear_total, bear_wins = _zero_arrays()
            total_trades_count = 0

            ema_value = None
            ema_sum = 0.0
            ema_count = 0
            atr55_value = None
            atr55_sum = 0.0
            atr55_count = 0
            atr100_value = None
            atr100_sum = 0.0
            atr100_count = 0
            rsi_up_value = None
            rsi_up_sum = 0.0
            rsi_down_value = None
            rsi_down_sum = 0.0
            rsi_count = 0
        else:
            bar_index = prev.bar_index + 1
            prev_close = prev.close
            prev_high = prev.high
            prev_low = prev.low
            close_above_prev = prev.close_above_envelope_count
            close_below_prev = prev.close_below_envelope_count
            last_signal_bar = prev.last_signal_bar

            active_bull = prev.active_bull
            active_bear = prev.active_bear
            bull_bucket = prev.bull_bucket
            bear_bucket = prev.bear_bucket
            bull_target_price = prev.bull_target_price
            bull_stop_price = prev.bull_stop_price
            bear_target_price = prev.bear_target_price
            bear_stop_price = prev.bear_stop_price
            bull_entry_bar = prev.bull_entry_bar
            bear_entry_bar = prev.bear_entry_bar

            bull_total = prev.bull_total
            bull_wins = prev.bull_wins
            bear_total = prev.bear_total
            bear_wins = prev.bear_wins
            total_trades_count = prev.total_trades_count

            ema_value = prev.ema_value
            ema_sum = prev.ema_sum
            ema_count = prev.ema_count
            atr55_value = prev.atr55_value
            atr55_sum = prev.atr55_sum
            atr55_count = prev.atr55_count
            atr100_value = prev.atr100_value
            atr100_sum = prev.atr100_sum
            atr100_count = prev.atr100_count
            rsi_up_value = prev.rsi_up_value
            rsi_up_sum = prev.rsi_up_sum
            rsi_down_value = prev.rsi_down_value
            rsi_down_sum = prev.rsi_down_sum
            rsi_count = prev.rsi_count

        # ----------------------------------------------------------------
        # CALCULATIONS (section "CALCULATIONS" of the script)
        # ----------------------------------------------------------------

        # basis = ta.ema(close, envelope_len)
        ema_sum, ema_count, ema_value = _feed_chain(
            ema_sum,
            ema_count,
            ema_value,
            close,
            self.envelope_len,
            2.0 / (self.envelope_len + 1.0),
        )

        tr = _true_range(prev_close, high, low)

        # vola = ta.atr(envelope_len)
        atr55_sum, atr55_count, atr55_value = _feed_chain(
            atr55_sum,
            atr55_count,
            atr55_value,
            tr,
            self.envelope_len,
            1.0 / self.envelope_len,
        )

        # atr = ta.atr(atr_len) * stop_mult
        atr100_sum, atr100_count, atr100_value = _feed_chain(
            atr100_sum,
            atr100_count,
            atr100_value,
            tr,
            self.atr_len,
            1.0 / self.atr_len,
        )

        # rsi = ta.rsi(close, rsi_len)
        if prev_close is not None:
            delta = close - prev_close
            up = max(delta, 0.0)
            down = max(-delta, 0.0)

            # Both Wilder accumulators advance in lockstep (they are fed
            # on exactly the same bars), so they share one counter.
            rsi_up_sum, rsi_up_count, rsi_up_value = _feed_chain(
                rsi_up_sum,
                rsi_count,
                rsi_up_value,
                up,
                self.rsi_len,
                1.0 / self.rsi_len,
            )
            rsi_down_sum, _, rsi_down_value = _feed_chain(
                rsi_down_sum,
                rsi_count,
                rsi_down_value,
                down,
                self.rsi_len,
                1.0 / self.rsi_len,
            )
            rsi_count = rsi_up_count

        if (
            rsi_count == self.rsi_len
            and rsi_up_value is not None
            and rsi_down_value is not None
        ):
            if rsi_down_value == 0.0:
                rsi = 100.0
            elif rsi_up_value == 0.0:
                rsi = 0.0
            else:
                rsi = 100.0 - 100.0 / (
                    1.0 + rsi_up_value / rsi_down_value
                )

            # rsi_bucket = max(0, min(10, round(rsi / 10)))
            rsi_bucket = max(
                0,
                min(10, _round(rsi / 10.0)),
            )
        else:
            rsi = None
            rsi_bucket = None

        # Envelope bands.
        if ema_value is not None and atr55_value is not None:
            basis = ema_value
            vola = atr55_value
            upper_band = basis + self.multiplier * vola
            lower_band = basis - self.multiplier * vola
        else:
            basis = None
            upper_band = None
            lower_band = None
            vola = atr55_value

        atr_stop = (
            atr100_value * self.stop_mult
            if atr100_value is not None
            else None
        )

        # Bull_Stop  = ta.lowest(low, 2) - atr
        # Bear_Stop  = ta.highest(high, 2) + atr
        if prev_low is not None:
            lowest2 = min(low, prev_low)
        else:
            lowest2 = None

        if prev_high is not None:
            highest2 = max(high, prev_high)
        else:
            highest2 = None

        bull_stop_level = (
            lowest2 - atr_stop
            if (lowest2 is not None and atr_stop is not None)
            else None
        )
        bear_stop_level = (
            highest2 + atr_stop
            if (highest2 is not None and atr_stop is not None)
            else None
        )

        # ----------------------------------------------------------------
        # RAW BEAR TRAP + envelope counter.
        # ----------------------------------------------------------------

        raw_bear_trap = False

        if upper_band is not None and close < upper_band:

            if prev_close is not None:

                before_window = (
                    close_above_prev <= self.trap_window
                )

                if (
                    high > upper_band
                    and prev_close < upper_band
                    and before_window
                ):
                    raw_bear_trap = True

                elif (
                    prev_close > upper_band
                    and before_window
                ):
                    raw_bear_trap = True

        # if high > upper_band
        #     close_above_envelope_count += 1
        # else
        #     close_above_envelope_count := 0
        if upper_band is not None and high > upper_band:
            close_above_envelope_count = close_above_prev + 1
        else:
            close_above_envelope_count = 0

        # ----------------------------------------------------------------
        # RAW BULL TRAP + envelope counter.
        # ----------------------------------------------------------------

        raw_bull_trap = False

        if lower_band is not None and close > lower_band:

            if prev_close is not None:

                before_window = (
                    close_below_prev <= self.trap_window
                )

                if (
                    low < lower_band
                    and prev_close > lower_band
                    and before_window
                ):
                    raw_bull_trap = True

                elif (
                    prev_close < lower_band
                    and before_window
                ):
                    raw_bull_trap = True

        # if low < lower_band
        #     close_below_envelope_count += 1
        # else
        #     close_below_envelope_count := 0
        if lower_band is not None and low < lower_band:
            close_below_envelope_count = close_below_prev + 1
        else:
            close_below_envelope_count = 0

        # ----------------------------------------------------------------
        # SIGNAL GAP.
        # ----------------------------------------------------------------

        # bars_from_live = last_bar_index - bar_index
        # The engine always runs at the live edge, so bars_from_live == 0.
        bars_from_live = 0
        allowed_by_limits = (
            bars_from_live <= self.max_bars
            and total_trades_count < self.max_trades
        )

        can_fire = (
            bar_index - last_signal_bar
        ) >= self.signal_gap

        bull_trap = raw_bull_trap and can_fire
        bear_trap = raw_bear_trap and can_fire

        if bull_trap or bear_trap:
            last_signal_bar = bar_index

        # ----------------------------------------------------------------
        # BULL TRACKING ENGINE.
        # ----------------------------------------------------------------

        if bull_trap and not active_bull and allowed_by_limits:
            active_bull = True
            bull_bucket = rsi_bucket
            total_trades_count += 1
            bull_entry_bar = bar_index

            # calculated_target = target_source == "Basis Line"
            #                       ? basis : upper_band
            calculated_target = (
                basis
                if self.target_source == "Basis Line"
                else upper_band
            )

            bull_target_price = calculated_target
            bull_stop_price = bull_stop_level

        if active_bull:
            # Target is evaluated before stop on the same bar.
            if (
                bull_target_price is not None
                and high >= bull_target_price
                and bar_index - last_signal_bar > 1
            ):
                bucket = _bucket_index(bull_bucket)
                bull_total = _inc(bull_total, bucket)
                bull_wins = _inc(bull_wins, bucket)
                active_bull = False
                bull_entry_bar = None

            elif (
                bull_stop_price is not None
                and low < bull_stop_price
                and bar_index - last_signal_bar > 1
            ):
                bucket = _bucket_index(bull_bucket)
                bull_total = _inc(bull_total, bucket)
                active_bull = False
                bull_entry_bar = None

        # ----------------------------------------------------------------
        # BEAR TRACKING ENGINE.
        # ----------------------------------------------------------------

        if bear_trap and not active_bear and allowed_by_limits:
            active_bear = True
            bear_bucket = rsi_bucket
            total_trades_count += 1
            bear_entry_bar = bar_index

            # calculated_target = target_source == "Basis Line"
            #                       ? basis : lower_band
            calculated_target = (
                basis
                if self.target_source == "Basis Line"
                else lower_band
            )

            bear_target_price = calculated_target
            bear_stop_price = bear_stop_level

        if active_bear:
            if (
                bear_target_price is not None
                and low <= bear_target_price
                and bar_index - last_signal_bar > 1
            ):
                bucket = _bucket_index(bear_bucket)
                bear_total = _inc(bear_total, bucket)
                bear_wins = _inc(bear_wins, bucket)
                active_bear = False
                bear_entry_bar = None

            elif (
                bear_stop_price is not None
                and high > bear_stop_price
                and bar_index - last_signal_bar > 1
            ):
                bucket = _bucket_index(bear_bucket)
                bear_total = _inc(bear_total, bucket)
                active_bear = False
                bear_entry_bar = None

        # ----------------------------------------------------------------
        # Output state.
        # ----------------------------------------------------------------

        return ReversalTrapValue(
            time=candle.time,
            start_ts=candle.start_ts,
            end_ts=candle.end_ts,
            bar_index=bar_index,
            basis=basis,
            upper_band=upper_band,
            lower_band=lower_band,
            atr_envelope=vola,
            rsi=rsi,
            rsi_bucket=rsi_bucket,
            atr_stop=atr_stop,
            bull_stop_level=bull_stop_level,
            bear_stop_level=bear_stop_level,
            raw_bull_trap=raw_bull_trap,
            raw_bear_trap=raw_bear_trap,
            bull_trap=bull_trap,
            bear_trap=bear_trap,
            close_above_envelope_count=close_above_envelope_count,
            close_below_envelope_count=close_below_envelope_count,
            active_bull=active_bull,
            active_bear=active_bear,
            bull_bucket=bull_bucket,
            bear_bucket=bear_bucket,
            bull_target_price=bull_target_price,
            bull_stop_price=bull_stop_price,
            bear_target_price=bear_target_price,
            bear_stop_price=bear_stop_price,
            bull_entry_bar=bull_entry_bar,
            bear_entry_bar=bear_entry_bar,
            allowed_by_limits=allowed_by_limits,
            bull_total=bull_total,
            bull_wins=bull_wins,
            bear_total=bear_total,
            bear_wins=bear_wins,
            total_trades_count=total_trades_count,
            high=high,
            low=low,
            close=close,
            last_signal_bar=last_signal_bar,
            ema_value=ema_value,
            ema_sum=ema_sum,
            ema_count=ema_count,
            atr55_value=atr55_value,
            atr55_sum=atr55_sum,
            atr55_count=atr55_count,
            atr100_value=atr100_value,
            atr100_sum=atr100_sum,
            atr100_count=atr100_count,
            rsi_up_value=rsi_up_value,
            rsi_up_sum=rsi_up_sum,
            rsi_down_value=rsi_down_value,
            rsi_down_sum=rsi_down_sum,
            rsi_count=rsi_count,
        )