# BLACKER
# Copyright (C) 2026 Juan José Caballero Rey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

from series.series import Series


MAX_HISTORY = 500


@dataclass(frozen=True)
class ATRValue:
    """
    One ATR reading.

    ``close`` travels with the value so a strategy can pair the ATR and the
    price it was computed from without having to reach into the timeframe.
    For the live (in progress) bar both fields are provisional.
    """

    time: int
    atr: Optional[float]
    #: True range of the bar, always available even during warmup.
    tr: float
    #: Close of the same bar the ATR refers to.
    close: float
    #: ``atr / close``, or None while the ATR is still warming up.
    atr_pct: Optional[float]


class ATR(Series):
    """
    Wilder's Average True Range.

    True range of a bar::

        tr = max(high - low, abs(high - previous_close), abs(low - previous_close))

    The ATR is seeded with the simple mean of the first ``period`` true
    ranges and then smoothed with Wilder's recursion::

        atr = (atr * (period - 1) + tr) / period

    ``live`` is recomputed on every tick from the in-progress bar, so it
    changes until the bar closes; ``history`` only ever receives fully
    closed bars.
    """

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

        period = params.get("period", 14)

        # The frontend can send the period either as a plain number
        # or as a parameter descriptor object carrying the value.
        if isinstance(period, dict):
            period = period.get("value", 14)

        self.period = int(period)

        if self.period <= 0:
            raise ValueError(
                "ATR period must be greater than 0"
            )

        self._live: ATRValue | None = None

        # Last ATR computed over a fully closed candle.
        self._closed: ATRValue | None = None

        # True ranges still needed to seed the average.
        self._seed: deque[float] = deque(maxlen=self.period)

        # Close of the last closed candle, needed for the next true range.
        self._prev_close: Optional[float] = None

        self.history: deque[ATRValue] = deque(
            maxlen=MAX_HISTORY
        )

    @property
    def live(self) -> ATRValue | None:
        return self._live

    @live.setter
    def live(self, value: ATRValue | None):
        self._live = value

    def _true_range(
        self,
        high: float,
        low: float,
        prev_close: Optional[float],
    ) -> float:
        if prev_close is None:
            return high - low

        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

    def _smooth(
        self,
        tr: float,
        prev_atr: Optional[float],
        prev_seed: deque[float],
    ) -> tuple[Optional[float], deque[float]]:
        """
        Advance the ATR by one true range.

        Returns the new ATR (None while still seeding) and the updated seed
        queue. The seed queue is copied so a provisional live calculation
        never pollutes the state of the closed series.
        """
        seed = deque(prev_seed)

        if prev_atr is not None:
            return (
                (prev_atr * (self.period - 1.0) + tr) / self.period,
                seed,
            )

        seed.append(tr)

        if len(seed) < self.period:
            return None, seed

        return sum(seed) / self.period, seed

    def update(self) -> None:

        timeframe = self._timeframe

        # --------------------------------------------------
        # Candle cerrada
        # --------------------------------------------------

        if timeframe.is_closed:

            candle = timeframe.closed

            if candle is not None:

                tr = self._true_range(
                    candle.high,
                    candle.low,
                    self._prev_close,
                )

                atr, seed = self._smooth(
                    tr,
                    (
                        self._closed.atr
                        if self._closed is not None
                        else None
                    ),
                    self._seed,
                )

                value = ATRValue(
                    time=candle.time,
                    atr=atr,
                    tr=tr,
                    close=candle.close,
                    atr_pct=(
                        atr / candle.close
                        if atr is not None and candle.close > 0
                        else None
                    ),
                )

                if self._closed is not None:
                    self.history.append(
                        self._closed
                    )

                self._closed = value
                self._seed = seed
                self._prev_close = candle.close

        # --------------------------------------------------
        # Candle viva
        # --------------------------------------------------

        candle = timeframe.live

        if candle is None:
            return

        tr = self._true_range(
            candle.high,
            candle.low,
            self._prev_close,
        )

        atr, _ = self._smooth(
            tr,
            (
                self._closed.atr
                if self._closed is not None
                else None
            ),
            self._seed,
        )

        self.live = ATRValue(
            time=candle.time,
            atr=atr,
            tr=tr,
            close=candle.close,
            atr_pct=(
                atr / candle.close
                if atr is not None and candle.close > 0
                else None
            ),
        )

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
            # `history` lags one bar behind `_closed`, so the newest
            # closed value has to be persisted explicitly or a restored
            # engine would resume from a one-bar-stale ATR.
            "closed": (
                asdict(self._closed)
                if self._closed is not None
                else None
            ),
            "prev_close": self._prev_close,
            "seed": list(self._seed),
        }

    def set_state(self, state: dict) -> None:

        live_state = state.get("live")

        self.live = (
            ATRValue(**live_state)
            if live_state is not None
            else None
        )

        history = [
            ATRValue(**value)
            for value in (
                state.get("history") or []
            )
        ]

        self.history = deque(
            history,
            maxlen=MAX_HISTORY,
        )

        # `closed` supersedes the `history[-1]` fallback kept for states
        # serialized before this field existed.
        closed_state = state.get("closed")

        self._closed = (
            ATRValue(**closed_state)
            if closed_state is not None
            else (
                history[-1]
                if history
                else None
            )
        )

        prev_close = state.get("prev_close")

        self._prev_close = (
            float(prev_close)
            if prev_close is not None
            else None
        )

        self._seed = deque(
            (float(value) for value in (state.get("seed") or [])),
            maxlen=self.period,
        )
