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
from typing import Callable, Optional
import math
import warnings

from orders.models import Side


#: Offset kinds understood by ``_offset``.
OFFSET_KINDS = ("percent", "absolute", "atr")


@dataclass
class RiskLevels:
    """
    Protective levels derived for one entry.

    For a long entry the stop sits below the entry price and the
    targets above it; for a short entry the directions are mirrored.
    """

    stop_price: Optional[float] = None

    #: Profit targets, ordered so the nearest one comes first.
    target_prices: list[float] = field(default_factory=list)


DEFAULT_STOP_CONFIG = {"type": "percent", "value": 0.01}


class RiskManager:
    """
    Converts a risk configuration into concrete stop / target prices.

    The RiskManager is decoupled from the Strategy: strategies emit
    plain entry/exit intents, this object decides how to protect the
    trade, and the OrderManager builds the resulting bracket.

    The bracket is static: a stop-loss plus any number of take-profit
    targets, all priced once against the entry and never moved
    afterwards. There is no trailing stop, so no level here depends on
    where price travels after the fill.

    Configuration schema (all keys optional)::

        {
            "stop":    {"type": "percent",  "value": 0.01}
                     | {"type": "absolute", "value": 1.5}
                     | {"type": "atr",      "multiplier": 2.0},
            "targets": [
                {"type": "percent",  "value": 0.02},
                {"type": "absolute", "value": 3.0},
            ]
        }

    Every order is protected by a stop-loss. If the configuration does
    not provide a ``stop`` key, a default of 1% below/above the entry
    price is used (``DEFAULT_STOP_CONFIG``).

    An offset that cannot produce a usable price never reaches the book:

    * An ``"atr"`` stop needs an ATR reading. It is taken from
      ``atr_provider`` (a callable evaluated on every use) or from a
      static ``context={"atr": <value>}``. When neither is available
      the offset would be 0.0 -- which parks the stop exactly on the
      entry price and takes out every position on its first tick -- so
      the default percent stop is used instead and a warning is raised.
    * A zero, negative or non-finite offset is rejected the same way.
    * An unknown ``type`` is reported instead of being ignored: a typo
      such as ``"pct"`` used to degrade silently to the default stop.

    Every rejected spec warns exactly once, so a broken configuration is
    visible in the engine log without flooding it once per tick.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        context: Optional[dict] = None,
        atr_provider: Optional[Callable[[], Optional[float]]] = None,
    ):
        self.config = config or {}
        self.context = context or {}
        self._atr_provider = atr_provider
        self._warned: set[str] = set()

        if not isinstance(self.config, dict):
            self._warn(
                "config",
                "Risk configuration must be an object, got "
                f"{type(self.config).__name__}. Every entry falls back to the "
                "default stop.",
            )
            self.config = {}

        if not isinstance(self.context, dict):
            self.context = {}

        if "trailing" in self.config:
            self._warn(
                "trailing",
                "The risk configuration sets 'trailing', which is no longer "
                "supported and is ignored. Brackets are static: the stop is "
                "priced once against the entry and never moved afterwards.",
            )

    # ======================================================
    # LEVELS
    # ======================================================

    def apply(self, side: Side, entry_price: float) -> RiskLevels:
        if not _is_price(entry_price):
            raise ValueError(f"entry_price must be a positive number, got {entry_price!r}")

        levels = RiskLevels()

        stop_spec = self.config.get("stop") or DEFAULT_STOP_CONFIG

        candidates = [("stop", stop_spec)]
        if stop_spec != DEFAULT_STOP_CONFIG:
            candidates.append(("stop.default", DEFAULT_STOP_CONFIG))

        for label, spec in candidates:
            offset = self._offset(spec, entry_price, label)

            if not self._usable(offset):
                continue

            price = self._below(offset, entry_price, side)

            if _is_price(price):
                levels.stop_price = price
                break

            self._warn(
                f"{label}.price",
                f"Risk configuration resolves the {label} to {price!r}, which is not "
                "a tradable price.",
            )

        if levels.stop_price is None:
            self._warn(
                "stop.unusable",
                "Risk configuration cannot produce a usable stop. The entry is left "
                "without a stop-loss.",
            )

        for index, spec in enumerate(self._targets()):
            offset = self._offset(spec, entry_price, f"targets[{index}]")

            if not self._usable(offset):
                continue

            price = self._above(offset, entry_price, side)

            if not _is_price(price):
                self._warn(
                    f"targets[{index}].price",
                    f"Risk configuration resolves target {index} to {price!r}, which "
                    "is not a tradable price. The target is dropped.",
                )
                continue

            levels.target_prices.append(price)

        # Nearest target first: a single tick that jumps through several
        # of them must fill the closest one, not whichever happened to be
        # listed first in the configuration.
        levels.target_prices.sort(reverse=(side == Side.SELL))

        return levels

    def _targets(self) -> list:
        targets = self.config.get("targets")

        if targets is None:
            return []

        if not isinstance(targets, (list, tuple)):
            self._warn(
                "targets",
                f"'targets' must be a list, got {type(targets).__name__}. No target "
                "will be placed.",
            )
            return []

        return list(targets)

    # ======================================================
    # OFFSETS
    # ======================================================

    def _atr_value(self) -> float:
        if self._atr_provider is not None:
            try:
                value = self._atr_provider()
            except Exception as exc:  # noqa: BLE001 - user supplied hook
                self._warn(
                    "atr_provider",
                    f"The ATR provider raised {exc!r}. Falling back to a static "
                    "ATR context, if any.",
                )
            else:
                if value is not None:
                    return abs(_number(value))

        return abs(_number(self.context.get("atr")))

    def _offset(self, spec: Optional[dict], price: float, label: str) -> Optional[float]:
        if not spec:
            return None

        if not isinstance(spec, dict):
            self._warn(
                f"{label}.type",
                f"Risk offset for {label} must be an object, got "
                f"{type(spec).__name__}. Falling back to the default percent level.",
            )
            return None

        if not _is_price(price):
            return None

        kind = spec.get("type")

        if kind == "percent":
            return abs(_number(spec.get("value"))) * price

        if kind == "absolute":
            return abs(_number(spec.get("value")))

        if kind == "atr":
            atr = self._atr_value()

            if atr <= 0.0:
                self._warn(
                    f"{label}.atr",
                    f"Risk configuration resolves {label} with type 'atr' but no ATR "
                    "reading is available, so the offset would be 0.0. Falling back to "
                    "the default percent level. Pass context={'atr': <value>} or an "
                    "atr_provider to RiskManager to use ATR-based levels.",
                )
                return None

            return abs(_number(spec.get("multiplier"), default=1.0)) * atr

        self._warn(
            f"{label}.type",
            f"Unknown risk offset type {kind!r} for {label}. Expected one of "
            f"{list(OFFSET_KINDS)}. Falling back to the default percent level.",
        )

        return None

    # ======================================================
    # UTILITIES
    # ======================================================

    @staticmethod
    def _usable(offset: Optional[float]) -> bool:
        return offset is not None and math.isfinite(offset) and offset > 0.0

    def _warn(self, key: str, message: str) -> None:
        if key in self._warned:
            return

        self._warned.add(key)

        warnings.warn(message, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _below(offset: float, entry_price: float, side: Side) -> float:
        # Long stops are placed below the entry, short stops above it.
        if side == Side.BUY:
            return entry_price - offset
        return entry_price + offset

    @staticmethod
    def _above(offset: float, entry_price: float, side: Side) -> float:
        # Long targets sit above the entry, short targets below it.
        if side == Side.BUY:
            return entry_price + offset
        return entry_price - offset


def _number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def _is_price(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0
