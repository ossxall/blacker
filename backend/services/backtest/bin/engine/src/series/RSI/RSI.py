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
class Rsi:
    time: int
    start_ts: int
    end_ts: int
    value: float

    # Internal continuity state
    close: float
    avg_gain: float
    avg_loss: float


class RSI(Series):
    """Wilder's Relative Strength Index."""

    def __init__(self, id: str, kind: str, level: int,
                 primary: bool, overlay: bool, params: dict):
        super().__init__(id, kind, level, primary, overlay, params)

        period = params.get("period", 14)
        if isinstance(period, dict):
            period = period.get("value", 14)
        self.period = int(period)

        self._internal: Rsi | None = None
        self._live: Rsi | None = None
        self.history: deque[Rsi] = deque(maxlen=MAX_HISTORY_LEN)

    @property
    def live(self) -> Rsi | None:
        return self._live

    @live.setter
    def live(self, value: Rsi | None):
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
            "history": [asdict(r) for r in self.history],
            # `live` is suppressed during the warm-up and is only an alias of
            # `_internal` once the chain is warm, so it cannot stand in for the
            # Wilder chain on restore: an engine rebuilt without it would resume
            # from the previous confirmed bar and confirm it a second time.
            "internal": (
                asdict(self._internal)
                if self._internal is not None
                else None
            ),
        }

    def set_state(self, state: dict) -> None:
        self.history = deque(
            (Rsi(**r) for r in (state.get("history") or [])),
            maxlen=MAX_HISTORY_LEN,
        )
        live_state = state.get("live")
        self.live = Rsi(**live_state) if live_state is not None else None

        # `internal` supersedes the fallback kept for states serialized
        # before this field existed.
        internal_state = state.get("internal")

        self._internal = (
            Rsi(**internal_state)
            if internal_state is not None
            else (
                self.live
                or (self.history[-1] if self.history else None)
            )
        )

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

        self._internal = self._compute(candle, prev)

        if len(self.history) >= self.period:
            self.live = self._internal
        else:
            self.live = None

    def _compute(self, candle, prev: Rsi | None) -> Rsi:
        if prev is None:
            return Rsi(
                time=candle.time,
                start_ts=candle.start_ts,
                end_ts=candle.end_ts,
                value=50.0,
                close=candle.close,
                avg_gain=0.0,
                avg_loss=0.0,
            )

        change = candle.close - prev.close
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0

        alpha = 1.0 / self.period
        avg_gain = prev.avg_gain * (1 - alpha) + gain * alpha
        avg_loss = prev.avg_loss * (1 - alpha) + loss * alpha

        if avg_gain == 0 and avg_loss == 0:
            value = 50.0
        elif avg_loss == 0:
            value = 100.0
        else:
            rs = avg_gain / avg_loss
            value = 100.0 - (100.0 / (1.0 + rs))

        return Rsi(
            time=candle.time,
            start_ts=candle.start_ts,
            end_ts=candle.end_ts,
            value=value,
            close=candle.close,
            avg_gain=avg_gain,
            avg_loss=avg_loss,
        )
