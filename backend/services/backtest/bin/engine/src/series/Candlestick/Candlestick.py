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

from aggregator.bar_aggregator import Bar
from series.series import Series


MAX_HISTORY = 500


@dataclass(frozen=True)
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    start_ts: int
    end_ts: int


class Candlestick(Series):

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
            params
        )

        self._closed: Candle | None = None

        self._live: Candle | None = None

        # Unlike EMA/ATR, `history` does not lag `_closed` by one bar: the
        # candle that `_closed` holds is also the newest one in `history`.
        # `history` is the record the frontend renders, so shifting it to
        # match the indicator series would drop the newest closed candle
        # from every published state to make the two arrangements look
        # alike. ADX keeps the same arrangement.
        self.history: deque[Candle] = deque(
            maxlen=MAX_HISTORY
        )

    @property
    def live(self) -> Candle | None:
        return self._live

    @live.setter
    def live(self, value: Candle | None):
        self._live = value

    @staticmethod
    def _to_candle(bar: Bar) -> Candle:
        return Candle(
            time=bar.time,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.total_volume,
            start_ts=bar.start_ts,
            end_ts=bar.end_ts,
        )

    def _confirm(self, bar: "Bar | None") -> None:
        """
        Confirma la vela cerrada y la publica en `_closed` y en `history`.

        `timeframe.closed` es la autoridad de cuál es la vela cerrada: en un
        rollover es la que acaba de morir y en un `flush()` es la que se
        acaba de finalizar. `_closed` y `history` se Actualizan juntas, así
        que la serie nunca queda una vela por detrás de su Timeframe.

        La guarda por `start_ts` evita que un `flush()` seguido de una vela
        nueva empuje la misma vela dos veces: en el flush ya se confirmó, y
        la vela que nace después volvería a encontrarla en `self.live`.
        """

        if bar is None:
            return

        if (
            self._closed is not None
            and self._closed.start_ts == bar.start_ts
        ):
            return

        candle = self._to_candle(bar)

        self._closed = candle
        self.history.append(candle)

    def update(self) -> None:
        """
        Actualiza la Candle usando la barra actual
        de su Timeframe.

        BarAggregator construye y actualiza la Bar.

        Timeframe.is_new indica si la Bar actual
        acaba de comenzar.

        Timeframe.is_closed indica que hay una Bar
        cerrada pendiente de confirmar.

        Candlestick solamente transforma Bar -> Candle
        y administra closed/live/history.
        """

        timeframe = self._timeframe
        bar = timeframe.live

        if bar is None:
            return

        # --------------------------------------------------
        # Nueva barra del timeframe
        # --------------------------------------------------

        if timeframe.is_new:

            self._confirm(timeframe.closed)

            self.live = self._to_candle(bar)

            return

        # --------------------------------------------------
        # Timeframe.flush(): la barra viva se acaba de
        # finalizar y ninguna la reemplazó, así que también
        # es una vela cerrada. Sin esto la serie se
        # quedaría una vela detrás de `timeframe.closed`.
        # --------------------------------------------------

        if timeframe.is_closed:
            self._confirm(timeframe.closed)

        # --------------------------------------------------
        # Actualización de la barra actual
        # --------------------------------------------------

        self.live = self._to_candle(bar)

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
                asdict(candle)
                for candle in self.history
            ],

            # `live` is the bar that is still open, so it cannot stand in
            # for `_closed`: a restored engine would render the open bar as
            # confirmed. It is published because `history` only receives a
            # bar once the next one starts.
            "closed": (
                asdict(self._closed)
                if self._closed is not None
                else None
            ),
        }

    def set_state(self, state: dict) -> None:
        live_state = state.get("live")

        self.live = (
            Candle(**live_state)
            if live_state is not None
            else None
        )

        self.history = deque(
            (
                Candle(**candle)
                for candle in (state.get("history") or [])
            ),
            maxlen=MAX_HISTORY,
        )

        # `closed` supersedes the fallback kept for states serialized before
        # this field existed. `history[-1]` is the exact alias of `_closed`
        # here, unlike in EMA/ATR where `history` lags one bar behind.
        closed_state = state.get("closed")

        self._closed = (
            Candle(**closed_state)
            if closed_state is not None
            else (
                self.history[-1]
                if self.history
                else None
            )
        )