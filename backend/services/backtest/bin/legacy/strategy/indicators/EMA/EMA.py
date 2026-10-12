# BLACKER
# Copyright (C) 2026 Juan José Caballero Rey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3 of the License.
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
from series.series import Series


MAX_HISTORY = 500


@dataclass(frozen=True)
class EMAValue:
    time: int
    value: float


class EMA(Series):

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

        period = params.get("period", 55)

        # The frontend can send the period either as a plain number
        # or as a parameter descriptor object carrying the value.
        if isinstance(period, dict):
            period = period.get("value", 55)

        self.period = int(period)

        if self.period <= 0:
            raise ValueError(
                "EMA period must be greater than 0"
            )

        self._live: EMAValue | None = None

        # Última EMA calculada sobre una candle cerrada.
        self._closed: EMAValue | None = None

        self.history: deque[EMAValue] = deque(
            maxlen=MAX_HISTORY
        )

    @property
    def live(self) -> EMAValue | None:
        return self._live

    @live.setter
    def live(self, value: EMAValue | None):
        self._live = value

    def _calculate(self, close: float) -> float:
        alpha = 2.0 / (self.period + 1.0)

        if self._closed is None:
            return close

        return (
            alpha * close
            + (1.0 - alpha) * self._closed.value
        )

    def update(self) -> None:

        timeframe = self._timeframe

        # --------------------------------------------------
        # Candle cerrada
        # --------------------------------------------------

        if timeframe.is_closed:

            candle = timeframe.closed

            if candle is not None:

                value = EMAValue(
                    time=candle.time,
                    value=self._calculate(
                        candle.close
                    ),
                )

                if self._closed is not None:
                    self.history.append(
                        self._closed
                    )

                self._closed = value

        # --------------------------------------------------
        # Candle viva
        # --------------------------------------------------

        candle = timeframe.live

        if candle is None:
            return

        self.live = EMAValue(
            time=candle.time,
            value=self._calculate(
                candle.close
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
            # engine would resume from a one-bar-stale EMA.
            "closed": (
                asdict(self._closed)
                if self._closed is not None
                else None
            ),
        }

    def set_state(self, state: dict) -> None:

        live_state = state.get("live")

        self.live = (
            EMAValue(**live_state)
            if live_state is not None
            else None
        )

        history = [
            EMAValue(**value)
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
            EMAValue(**closed_state)
            if closed_state is not None
            else (
                history[-1]
                if history
                else None
            )
        )
