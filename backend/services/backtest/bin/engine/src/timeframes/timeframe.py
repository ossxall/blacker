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

from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any
from aggregator.bar_aggregator import Bar
from series.series import Series

class Timeframe:
    def __init__(self):
        self._series: dict[str, Series] = {}
        self._levels: list[list[Series]] = []

        self.id: str = ""
        self.timeframe_ms: int = 0
        # Used by BarAggregator, Series
        self.live: Bar | None = None
        self.closed: Bar | None = None
        self.is_new: bool = False
        self.is_closed: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "timeframe_ms": self.timeframe_ms,
            "live": self.live.to_dict() if self.live is not None else None,
            "closed": self.closed.to_dict() if self.closed is not None else None,
            "is_new": self.is_new,
            "is_closed": self.is_closed,
            "series": {
                id: series.to_dict()
                for id, series in self._series.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Timeframe':
        timeframe = cls()

        timeframe.id = data.get("id")
        timeframe.timeframe_ms = data.get("timeframe_ms")
        timeframe.live = (
            Bar.from_dict(data["live"])
            if data.get("live") is not None
            else None
        )
        timeframe.closed = (
            Bar.from_dict(data["closed"])
            if data.get("closed") is not None
            else None
        )        
        timeframe.is_new = data.get("is_new")
        timeframe.is_closed = data.get("is_closed")

        return timeframe

    def add_series(self, series: Series):
        series._timeframe = self
        self._series[series.id] = series

    def get_series(
        self,
        kind: str,
        label: str,
    ):
        for series in self._series.values():
            if (
                series.kind == kind
                and series.params.get("label") == label
            ):
                return series

        raise KeyError(
            f"Series not found: kind={kind!r}, label={label!r}"
        )
    
    def build_levels(self):
        groups = defaultdict(list)

        for series in self._series.values():
            groups[series.level].append(series)

        self._levels = [
            groups[level]
            for level in sorted(groups)
        ]

    def update(self):
        """
        Actualiza todas las Series usando el estado
        actual del Timeframe.

        La barra live es construida por BarAggregator.
        """
        for level in self._levels:
            for series in level:
                series.update()

        self.is_new = False
        self.is_closed = False                

    def flush(self):
        """
        Finaliza la última barra live del Timeframe.

        La última barra pasa a considerarse cerrada para que
        los indicadores puedan procesarla antes de finalizar.
        """

        if self.live is None:
            return

        # --------------------------------------------------
        # La última barra pasa a ser la barra cerrada
        # --------------------------------------------------

        self.closed = self.live

        # --------------------------------------------------
        # La barra acaba de cerrarse.
        # No nació una nueva barra.
        # --------------------------------------------------

        self.is_new = False
        self.is_closed = True

        # --------------------------------------------------
        # Give the final push to the series.
        # --------------------------------------------------

        for level in self._levels:
            for series in level:
                series.update()

        # --------------------------------------------------
        # Ya no existe una barra live.
        # `closed` conserva la última barra confirmada.
        # --------------------------------------------------

        self.live = None

        self.is_new = False
        self.is_closed = False