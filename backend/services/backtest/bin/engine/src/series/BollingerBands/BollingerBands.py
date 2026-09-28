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
class BollingerValue:
    time: int
    start_ts: int
    end_ts: int
    middle: float
    upper: float
    lower: float
    bandwidth: float
    percent_b: float

    close: float


class BollingerBands(Series):
    """SMA basis with a standard-deviation band (classic Bollinger)."""

    def __init__(self, id: str, kind: str, level: int,
                 primary: bool, overlay: bool, params: dict):
        super().__init__(id, kind, level, primary, overlay, params)

        period = params.get("period", 20)
        if isinstance(period, dict):
            period = period.get("value", 20)
        self.period = int(period)

        mult = params.get("mult", 2.0)
        if isinstance(mult, dict):
            mult = mult.get("value", 2.0)
        self.mult = float(mult)

        self._closes: deque[float] = deque(maxlen=self.period)
        self._internal: BollingerValue | None = None
        self._live: BollingerValue | None = None
        self.history: deque[BollingerValue] = deque(maxlen=MAX_HISTORY_LEN)

    @property
    def live(self) -> BollingerValue | None:
        return self._live

    @live.setter
    def live(self, value: BollingerValue | None):
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
            "history": [asdict(b) for b in self.history],
            # `live` is suppressed until the close window is full and is only
            # an alias of `_internal` once the series is warm, so it cannot
            # stand in for the band state on restore: an engine rebuilt without
            # it would confirm the previous bar a second time.
            "internal": (
                asdict(self._internal)
                if self._internal is not None
                else None
            ),
        }

    def set_state(self, state: dict) -> None:
        self.history = deque(
            (BollingerValue(**b) for b in (state.get("history") or [])),
            maxlen=MAX_HISTORY_LEN,
        )
        live_state = state.get("live")
        self.live = BollingerValue(**live_state) if live_state is not None else None

        # `internal` supersedes the fallback kept for states serialized
        # before this field existed.
        internal_state = state.get("internal")

        self._internal = (
            BollingerValue(**internal_state)
            if internal_state is not None
            else (
                self.live
                or (self.history[-1] if self.history else None)
            )
        )

        # Rebuild the rolling close window from the retained history.
        self._closes = deque((b.close for b in self.history), maxlen=self.period)

    def update(self) -> None:
        candle = self._timeframe.live
        if candle is None:
            return

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

        # Keep one close per candle: replace the value of the candle that is
        # still open, otherwise append the newly confirmed close.
        if (
            not is_same_candle
            and prev is not None
            and prev.start_ts != candle.start_ts
        ):
            self._closes.append(prev.close)

        self._internal = self._compute(candle)

        if len(self._closes) >= self.period:
            self.live = self._internal
        else:
            self.live = None

    def _compute(self, candle) -> BollingerValue:
        window = list(self._closes) + [candle.close]
        window = window[-self.period:]

        if len(window) < self.period:
            middle = sum(window) / len(window) if window else candle.close
            return BollingerValue(
                time=candle.time, start_ts=candle.start_ts,
                end_ts=candle.end_ts, middle=middle, upper=middle,
                lower=middle, bandwidth=0.0, percent_b=0.5, close=candle.close,
            )

        middle = sum(window) / self.period
        variance = sum((x - middle) ** 2 for x in window) / self.period
        dev = variance ** 0.5
        upper = middle + self.mult * dev
        lower = middle - self.mult * dev
        span = upper - lower
        percent_b = (candle.close - lower) / span if span != 0 else 0.5
        bandwidth = span / middle if middle != 0 else 0.0

        return BollingerValue(
            time=candle.time, start_ts=candle.start_ts,
            end_ts=candle.end_ts, middle=middle, upper=upper, lower=lower,
            bandwidth=bandwidth, percent_b=percent_b, close=candle.close,
        )
