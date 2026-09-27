# BLACKER
# Copyright (C) 2026 Juan José Caballero Rey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import deque
from dataclasses import asdict, dataclass

from series.series import Series

MAX_HISTORY_LEN = 500


@dataclass(frozen=True)
class StochRsiValue:
    time: int
    start_ts: int
    end_ts: int
    value: float
    k: float
    d: float
    is_oversold: bool
    is_overbought: bool
    crossed_up: bool
    crossed_down: bool


class StochRSI(Series):
    """
    Stochastic RSI.

    The RSI is rescaled into a 0-100 oscillator over `rsi_period` candles
    and then smoothed with %K / %D moving averages.
    """

    def __init__(self, id: str, kind: str, level: int,
                 primary: bool, overlay: bool, params: dict):
        super().__init__(id, kind, level, primary, overlay, params)

        def _p(name, default):
            v = params.get(name, default)
            return v.get("value", default) if isinstance(v, dict) else v

        self.rsi_period = int(_p("rsi_period", 14))
        self.stoch_period = int(_p("stoch_period", 14))
        self.k_period = int(_p("k_period", 3))
        self.d_period = int(_p("d_period", 3))
        self.oversold = float(_p("oversold", 20.0))
        self.overbought = float(_p("overbought", 80.0))

        self._rsis: deque[float] = deque(maxlen=self.stoch_period)
        self._ks: deque[float] = deque(maxlen=self.k_period)
        self._ds: deque[float] = deque(maxlen=self.d_period)
        self._prev_k: float | None = None
        self._prev_d: float | None = None

        self._internal: StochRsiValue | None = None
        self._live: StochRsiValue | None = None
        self.history: deque[StochRsiValue] = deque(maxlen=MAX_HISTORY_LEN)

    @property
    def live(self) -> StochRsiValue | None:
        return self._live

    @live.setter
    def live(self, value: StochRsiValue | None):
        self._live = value

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "level": self.level,
            "primary": self.primary,
            "overlay": self.overlay,
            "params": self.params,
            "live": asdict(self.live) if self.live is not None else None,
            "history": [asdict(s) for s in self.history],
        }

    def set_state(self, state: dict) -> None:
        self.history = deque(
            (StochRsiValue(**s) for s in (state.get("history") or [])),
            maxlen=MAX_HISTORY_LEN,
        )
        live_state = state.get("live")
        self.live = StochRsiValue(**live_state) if live_state is not None else None
        self._internal = self.live or (self.history[-1] if self.history else None)
        # The k/d chains are rebuilt from the retained history.
        self._ks = deque((s.k for s in self.history), maxlen=self.k_period)
        self._ds = deque((s.d for s in self.history), maxlen=self.d_period)
        self._rsis = deque(
            (s.value for s in self.history), maxlen=self.stoch_period
        )
        if len(self.history) >= 2:
            self._prev_k = self.history[-2].k
            self._prev_d = self.history[-2].d

    def update(self) -> None:
        candle = self._timeframe.live
        if candle is None:
            return

        rsi_series = None
        try:
            rsi_series = self._timeframe.get_series("RSI", self._rsi_label)
        except KeyError:
            return
        if rsi_series is None or rsi_series.live is None:
            return

        rsi_now = rsi_series.live.value

        is_same_candle = (
            self._internal is not None
            and self._internal.start_ts == candle.start_ts
        )

        if self._internal is None:
            prev = None
        elif is_same_candle:
            prev = self.history[-1] if self.history else self._internal
        else:
            self.history.append(self._internal)
            prev = self._internal
            if prev is not None:
                # Commit the confirmed RSI reading before the new one.
                self._rsis.append(prev.value)

        if not is_same_candle or prev is None:
            self._rsis.append(rsi_now)
        else:
            # Replace the open candle's reading.
            self._rsis[-1] = rsi_now if self._rsis else rsi_now

        self._internal = self._compute(candle)

        if len(self._rsis) >= self.stoch_period:
            self.live = self._internal
        else:
            self.live = None

    @property
    def _rsi_label(self) -> str:
        return f"RSI {self.rsi_period}"

    def _compute(self, candle) -> StochRsiValue:
        window = list(self._rsis)
        lo = min(window) if window else 0.0
        hi = max(window) if window else 0.0
        raw = 0.0 if hi == lo else 100.0 * (window[-1] - lo) / (hi - lo)

        self._ks.append(raw)
        k = sum(self._ks) / len(self._ks)
        self._ds.append(k)
        d = sum(self._ds) / len(self._ds)

        crossed_up = (
            self._prev_k is not None and self._prev_d is not None
            and self._prev_k <= self._prev_d and k > d
        )
        crossed_down = (
            self._prev_k is not None and self._prev_d is not None
            and self._prev_k >= self._prev_d and k < d
        )

        value = (
            k if (k <= self.oversold or d <= self.oversold
                  or k >= self.overbought or d >= self.overbought)
            else 50.0
        )

        self._prev_k, self._prev_d = k, d

        return StochRsiValue(
            time=candle.time,
            start_ts=candle.start_ts,
            end_ts=candle.end_ts,
            value=value,
            k=k,
            d=d,
            is_overbought=k >= self.overbought,
            is_oversold=k <= self.oversold,
            crossed_up=crossed_up,
            crossed_down=crossed_down,
        )
