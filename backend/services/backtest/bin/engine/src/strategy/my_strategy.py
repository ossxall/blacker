from core.engine_state import EngineState

from strategy.base import Strategy

from orders import Signal, Side


class Strategy1(Strategy):
    """
    Multi-Timeframe EMA 20/50 + ADX Strategy (confirmed-candle revision).

    15m:
        Macro direction only.

    5m:
        Trend confirmation + ADX strength + DI conviction.

    1m:
        Momentum confirmation at entry / pullback-resume trigger.

    ------------------------------------------------------------------
    Why this revision is different from the live-tick version.
    ------------------------------------------------------------------

    The previous implementation decided entries and exits from the
    *live* EMA/ADX values.  Because the engine re-evaluates the strategy
    on every tick, the 1m EMA of the forming candle moves tick by tick,
    so the same signal could flip on and off dozens of times inside a
    single minute (entry -> exit -> entry ...).  In backtesting this
    produced hundreds of zero-duration round trips and a strongly
    negative / choppy equity curve.

    This revision only acts on CONFIRMED (closed candle) values:

        - EMA 20 / EMA 50 are read from the last *closed* candle;
        - ADX / +DI / -DI are read from the last confirmed ADX value;
        - a FRESH 5m EMA 20/50 cross is required for the main entry,
          and it can fire at most once per confirmed 5m candle;
        - continuation entries (1m pullback-resume) are also gated to
          once per confirmed 5m candle;
        - after any exit a minimum number of confirmed 5m candles must
          elapse before the strategy may enter again.

    Exits are also computed from confirmed candles, plus two profit
    managers that protect green trades:
        - an optional hard take-profit (take_profit_pct), and
        - a momentum-stall exit (green + 5m ADX declining + 1m structure
          turned against the position).

    LONG:
        15m EMA20 > EMA50, 15m ADX >= adx_strength           (confirmed)
        5m  EMA20 > EMA50, 5m ADX >= adx_strength             (confirmed)
        5m  ADX rising, +DI above -DI by at least di_gap      (confirmed)
        5m  fresh bull cross, or 1m fresh bull cross          (confirmed)
        1m  EMA20 > EMA50                                     (confirmed)

    SHORT:
        Mirrored conditions.

    Exits:
        - 5m EMA20/EMA50 structure failure
        - 5m ADX below adx_exit
        - 5m ADX reversal flag
        - take-profit at take_profit_pct above entry (0 = disabled)
        - momentum stall on a profitable position
    """

    DEFAULT_PARAMS = {
        "adx_strength": 20.0,
        "adx_exit": 15.0,
        "di_gap": 4.0,
        "take_profit_pct": 0.0,
        "cooldown_bars": 3,
        "allow_continuations": True,
    }

    def __init__(self, kind: str, params: dict):
        super().__init__(kind, params)

        merged = dict(self.DEFAULT_PARAMS)
        merged.update(self._unwrap(params))

        self.adx_strength = float(merged["adx_strength"])
        self.adx_exit = float(merged["adx_exit"])
        self.di_gap = float(merged["di_gap"])
        self.take_profit_pct = float(merged.get("take_profit_pct", 0.0))
        self.cooldown_bars = int(merged["cooldown_bars"])
        self.allow_continuations = bool(merged["allow_continuations"])

        # ------------------------------------------------------------------
        # In-memory guards. They prevent duplicate / whipsaw trading within
        # the same confirmed 5m candle. They are intentionally not persisted
        # through engine restores: a freshly restored engine simply waits one
        # confirmed candle before it may fire again.
        # ------------------------------------------------------------------

        self._prev20_5: float | None = None
        self._prev50_5: float | None = None
        self._prev20_1: float | None = None
        self._prev50_1: float | None = None

        self._last_signal_bar: int | None = None
        self._last_entry_bar: int | None = None
        self._exit_bar: int | None = None

    def evaluate(self, state: EngineState):

        tf15 = state.timeframes.get("15m")
        tf5 = state.timeframes.get("5m")
        tf1 = state.timeframes.get("1m")

        if tf15 is None or tf5 is None or tf1 is None:
            return None

        # ============================================================
        # SERIES RESOLUTION
        # ============================================================

        ema20_15 = self._get_series(tf15, "EMA", "EMA 20")
        ema50_15 = self._get_series(tf15, "EMA", "EMA 50")
        adx_15 = self._get_series(tf15, "ADX", "ADX 14")

        ema20_5 = self._get_series(tf5, "EMA", "EMA 20")
        ema50_5 = self._get_series(tf5, "EMA", "EMA 50")
        adx_5 = self._get_series(tf5, "ADX", "ADX 14")

        ema20_1 = self._get_series(tf1, "EMA", "EMA 20")
        ema50_1 = self._get_series(tf1, "EMA", "EMA 50")

        if not all([ema20_15, ema50_15, adx_15, ema20_5, ema50_5,
                    adx_5, ema20_1, ema50_1]):
            return None

        # ADX only exposes values after its warm-up period.
        if adx_15.live is None or adx_5.live is None:
            return None

        # ============================================================
        # CONFIRMED VALUES (closed candles only)
        # ============================================================

        c20_15 = self._closed_value(ema20_15)
        c50_15 = self._closed_value(ema50_15)
        c20_5 = self._closed_value(ema20_5)
        c50_5 = self._closed_value(ema50_5)
        c20_1 = self._closed_value(ema20_1)
        c50_1 = self._closed_value(ema50_1)

        if not all([c20_15, c50_15, c20_5, c50_5, c20_1, c50_1]):
            return None

        a15 = self._adx_latest(adx_15)
        a5 = self._adx_latest(adx_5)
        a5p = self._adx_previous(adx_5)

        if a15 is None or a5 is None:
            return None

        # Previous confirmed values, tracked by the strategy itself. They
        # are captured on every tick so a fresh cross is only detected on
        # the tick where a candle closes.
        prev20_5 = self._capture_prev(ema20_5, "_prev20_5")
        prev50_5 = self._capture_prev(ema50_5, "_prev50_5")
        prev20_1 = self._capture_prev(ema20_1, "_prev20_1")
        prev50_1 = self._capture_prev(ema50_1, "_prev50_1")

        # ============================================================
        # CONDITION PRIMITIVES
        # ============================================================

        macro_up = (
            c20_15 > c50_15
            and a15.adx >= self.adx_strength
        )

        macro_down = (
            c20_15 < c50_15
            and a15.adx >= self.adx_strength
        )

        struct_up_5 = c20_5 > c50_5
        struct_down_5 = c20_5 < c50_5

        adx_ok_5 = a5.adx >= self.adx_strength
        adx_rising_5 = a5p is not None and a5.adx > a5p.adx

        di_bull_5 = (a5.plus_di - a5.minus_di) >= self.di_gap
        di_bear_5 = (a5.minus_di - a5.plus_di) >= self.di_gap

        # A fresh cross only counts when it happens between the previous
        # and the current CONFIRMED candle.
        fresh_bull_5 = bool(
            prev20_5 is not None
            and prev50_5 is not None
            and c20_5 > c50_5
            and prev20_5 <= prev50_5
        )

        fresh_bear_5 = bool(
            prev20_5 is not None
            and prev50_5 is not None
            and c20_5 < c50_5
            and prev20_5 >= prev50_5
        )

        fresh_bull_1 = bool(
            prev20_1 is not None
            and prev50_1 is not None
            and c20_1 > c50_1
            and prev20_1 <= prev50_1
        )

        fresh_bear_1 = bool(
            prev20_1 is not None
            and prev50_1 is not None
            and c20_1 < c50_1
            and prev20_1 >= prev50_1
        )

        momentum_bull_1 = c20_1 > c50_1
        momentum_bear_1 = c20_1 < c50_1

        # Identity of the latest confirmed 5m candle (ms bucket start).
        current_bar = a5.start_ts

        portfolio = state.portfolio

        position = (
            portfolio.position
            if portfolio is not None
            else None
        )

        # ============================================================
        # POSITION MANAGEMENT
        # ============================================================

        if position is not None:

            if self._last_entry_bar is None:
                self._last_entry_bar = current_bar

            price = self._last_price(tf1)
            avg = position.avg_price

            # --------------------------------------------------------
            # LONG EXIT
            # --------------------------------------------------------

            if position.side == Side.BUY:

                if (
                    not struct_up_5
                    or a5.adx < self.adx_exit
                    or a5.is_reversal
                ):
                    return self._exit(current_bar)

                if (
                    self.take_profit_pct > 0.0
                    and price is not None
                    and price >= avg * (1.0 + self.take_profit_pct)
                ):
                    return self._exit(current_bar)

                if (
                    price is not None
                    and price > avg
                    and a5p is not None
                    and a5.adx < a5p.adx
                    and momentum_bear_1
                ):
                    return self._exit(current_bar)

                return None

            # --------------------------------------------------------
            # SHORT EXIT
            # --------------------------------------------------------

            if (
                not struct_down_5
                or a5.adx < self.adx_exit
                or a5.is_reversal
            ):
                return self._exit(current_bar)

            if (
                self.take_profit_pct > 0.0
                and price is not None
                and price <= avg * (1.0 - self.take_profit_pct)
            ):
                return self._exit(current_bar)

            if (
                price is not None
                and price < avg
                and a5p is not None
                and a5.adx < a5p.adx
                and momentum_bull_1
            ):
                return self._exit(current_bar)

            return None

        # ============================================================
        # FLAT: ENTRIES
        # ============================================================

        # If the position no longer exists, the last entry was closed by
        # a protective order (stop / target) without the strategy knowing.
        # Start the cooldown window so we do not instantly re-enter.
        if self._last_entry_bar is not None:

            if self._exit_bar is None:
                self._exit_bar = self._last_entry_bar

            self._last_entry_bar = None

        # Cooldown: wait a few confirmed 5m candles after any exit.
        if self._exit_bar is not None:

            if current_bar < self._exit_bar + self.cooldown_bars * 300_000:
                return None

            self._exit_bar = None

        # One shot per confirmed 5m candle.
        if current_bar == self._last_signal_bar:
            return None

        # ------------------------------------------------------------
        # FRESH 5m CROSS ENTRY
        # ------------------------------------------------------------

        if (
            macro_up
            and fresh_bull_5
            and adx_ok_5
            and adx_rising_5
            and di_bull_5
            and momentum_bull_1
        ):
            return self._enter(
                current_bar,
                Signal(action="BUY", quantity=1),
            )

        if (
            macro_down
            and fresh_bear_5
            and adx_ok_5
            and adx_rising_5
            and di_bear_5
            and momentum_bear_1
        ):
            return self._enter(
                current_bar,
                Signal(action="SELL", quantity=1),
            )

        # ------------------------------------------------------------
        # PULLBACK CONTINUATION ENTRY (optional)
        # ------------------------------------------------------------

        if self.allow_continuations:

            if (
                macro_up
                and struct_up_5
                and adx_ok_5
                and adx_rising_5
                and di_bull_5
                and fresh_bull_1
                and c20_1 > c20_5
            ):
                return self._enter(
                    current_bar,
                    Signal(action="BUY", quantity=1),
                )

            if (
                macro_down
                and struct_down_5
                and adx_ok_5
                and adx_rising_5
                and di_bear_5
                and fresh_bear_1
                and c20_1 < c20_5
            ):
                return self._enter(
                    current_bar,
                    Signal(action="SELL", quantity=1),
                )

        return None

    # ================================================================
    # STATE HELPERS
    # ================================================================

    def _enter(self, bar: int, signal: Signal) -> Signal:
        self._last_signal_bar = bar
        self._last_entry_bar = bar
        return signal

    def _exit(self, bar: int) -> Signal:
        self._exit_bar = bar
        self._last_entry_bar = None
        return Signal(action="EXIT")

    def _capture_prev(self, series, attr: str) -> float | None:
        latest = getattr(series, "_closed", None)

        if latest is None:
            return None

        previous = getattr(self, attr, None)

        if previous is None:
            previous = latest.value

        setattr(self, attr, latest.value)

        return previous

    @staticmethod
    def _closed_value(series) -> float | None:
        latest = getattr(series, "_closed", None)

        if latest is None:
            return None

        return latest.value

    @staticmethod
    def _adx_latest(series):
        history = getattr(series, "history", None)

        if not history:
            return None

        return history[-1]

    @staticmethod
    def _adx_previous(series):
        history = getattr(series, "history", None)

        if not history or len(history) < 2:
            return None

        return history[-2]

    @staticmethod
    def _last_price(tf) -> float | None:
        live = getattr(tf, "live", None)

        if live is not None:
            return live.close

        closed = getattr(tf, "closed", None)

        if closed is not None:
            return closed.close

        return None

    # ================================================================
    # MISC HELPERS
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
    def _get_series(tf, kind, label):
        try:
            return tf.get_series(kind, label)

        except KeyError:
            return None