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

from dataclasses import dataclass, field
from typing import Optional
import warnings

from orders.models import Side


@dataclass
class RiskLevels:
    """
    Protective levels derived for one entry.

    For a long entry the stop sits below the entry price and the
    targets above it; for a short entry the directions are mirrored.
    """

    stop_price: Optional[float] = None
    target_prices: list[float] = field(default_factory=list)
    #: Distance (in price units) used by a trailing stop. 0 disables trailing.
    trailing_distance: float = 0.0


DEFAULT_STOP_CONFIG = {"type": "percent", "value": 0.01}


class RiskManager:
    """
    Converts a risk configuration into concrete stop / target prices.

    The RiskManager is decoupled from the Strategy: strategies emit
    plain entry/exit intents, this object decides how to protect the
    trade, and the OrderManager builds the resulting bracket.

    Configuration schema (all keys optional)::

        {
            "stop":    {"type": "percent",  "value": 0.01}
                     | {"type": "absolute", "value": 1.5}
                     | {"type": "atr",      "multiplier": 2.0},
            "targets": [
                {"type": "percent",  "value": 0.02},
                {"type": "absolute", "value": 3.0},
            ],
            "trailing": {
                "enabled": True,
                "distance": {"type": "percent", "value": 0.005},
            }
        }

    Every order is protected by a stop-loss. If the configuration does
    not provide a ``stop`` key, a default of 1% below/above the entry
    price is used (``DEFAULT_STOP_CONFIG``).

    An ``"atr"`` stop needs an ATR reading, which is only available when a
    ``context`` carrying ``{"atr": <value>}`` is supplied. The engine does
    not build one, so rather than silently resolving the offset to 0.0 --
    which would park the stop exactly on the entry price and take out every
    position on its first tick -- the default percent stop is used instead
    and a warning is raised.
    """

    def __init__(self, config: Optional[dict] = None, context: Optional[dict] = None):
        self.config = config or {}
        self.context = context or {}
        self._warned_atr = False

    def apply(self, side: Side, entry_price: float) -> RiskLevels:
        stop_spec = self.config.get("stop") or DEFAULT_STOP_CONFIG
        target_specs = self.config.get("targets") or []
        trailing_spec = self.config.get("trailing") or {}

        levels = RiskLevels()

        stop_offset = self._offset(stop_spec, entry_price)
        if stop_offset is None:
            stop_offset = self._offset(
                DEFAULT_STOP_CONFIG,
                entry_price,
            )
        if stop_offset is not None:
            levels.stop_price = self._below(stop_offset, entry_price, side)

        for spec in target_specs:
            offset = self._offset(spec, entry_price)
            if offset is None:
                continue
            levels.target_prices.append(self._above(offset, entry_price, side))

        if trailing_spec.get("enabled"):
            distance_spec = trailing_spec.get("distance")
            distance = self._offset(distance_spec, entry_price) if distance_spec else None
            if distance is not None:
                levels.trailing_distance = distance

        return levels

    def _atr_value(self) -> float:
        return abs(float(self.context.get("atr", 0.0) or 0.0))

    def _offset(self, spec: dict, entry_price: float) -> Optional[float]:
        if not spec:
            return None

        kind = spec.get("type")

        if kind == "percent":
            return abs(float(spec.get("value", 0.0))) * entry_price

        if kind == "absolute":
            return abs(float(spec.get("value", 0.0)))

        if kind == "atr":
            atr = self._atr_value()

            if atr <= 0.0:
                if not self._warned_atr:
                    warnings.warn(
                        "Risk configuration asks for an 'atr' stop or target "
                        "but no ATR context was provided, so the offset would "
                        "be 0.0. Falling back to the default percent level. "
                        "Pass context={'atr': <value>} to RiskManager to use "
                        "ATR-based levels.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_atr = True
                return None

            multiplier = abs(float(spec.get("multiplier", 1.0)))
            return multiplier * atr

        return None

    def _below(self, offset: float, entry_price: float, side: Side) -> float:
        # Long stops are placed below the entry, short stops above it.
        if side == Side.BUY:
            return entry_price - offset
        return entry_price + offset

    def _above(self, offset: float, entry_price: float, side: Side) -> float:
        # Long targets sit above the entry, short targets below it.
        if side == Side.BUY:
            return entry_price + offset
        return entry_price - offset