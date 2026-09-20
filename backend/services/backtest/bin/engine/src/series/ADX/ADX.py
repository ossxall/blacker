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


MAX_HISTORY_LEN = 500


@dataclass(frozen=True)
class Adx:
    time: int
    start_ts: int
    end_ts: int

    # Public output columns (equivalent to the adx.py DataFrame columns)
    adx: float
    plus_di: float
    minus_di: float
    adx_color: str
    is_reversal: bool
    reversal_level: float | None

    # Internal continuity state, needed to compute the next value
    high: float
    low: float
    close: float
    tr_rma: float
    plus_dm_rma: float
    minus_dm_rma: float


class ADX(Series):
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

        self.dilen = int(params.get("dilen", 14))
        self.adxlen = int(params.get("adxlen", 14))
        self.key_level = float(params.get("key_level", 23))

        # Internal chain state, never suppressed.
        self._internal: Adx | None = None

        # Visible state (None during warm-up).
        self._live: Adx | None = None

        self.history: deque[Adx] = deque(maxlen=MAX_HISTORY_LEN)

    @property
    def live(self) -> Adx | None:
        return self._live

    @live.setter
    def live(self, value: Adx | None):
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
            "history": [asdict(a) for a in self.history],
        }

    def set_state(self, state: dict) -> None:
        self.history = deque(
            (Adx(**a) for a in (state.get("history") or [])),
            maxlen=MAX_HISTORY_LEN,
        )

        live_state = state.get("live")

        self.live = (
            Adx(**live_state) if live_state is not None else None
        )

        # The internal state is rebuilt from live, or from the last history
        # entry when live is suppressed by the warm-up period.
        self._internal = self.live or (self.history[-1] if self.history else None)

    def update(self) -> None:
        candle = self._timeframe.live

        if candle is None:
            return

        # True when updating the current open candle.
        is_same_candle = (
            self._internal is not None
            and self._internal.start_ts == candle.start_ts
        )

        # Select the previous state used to continue the RMA chain.
        if self._internal is None:
            prev_chain = None

        elif is_same_candle:
            # Continue from the last confirmed state (or the current state on the first candle).
            prev_chain = self.history[-1] if self.history else self._internal

        else:
            # Confirm the previous candle before starting a new one.
            self.history.append(self._internal)
            prev_chain = self._internal

        # Last two confirmed values used for ADX color and reversal detection.
        prev1 = self.history[-1] if len(self.history) >= 1 else None
        prev2 = self.history[-2] if len(self.history) >= 2 else None

        # Compute the current ADX state.
        self._internal = self._compute_step(candle, prev_chain, prev1, prev2)

        # Expose values only after the required warm-up period.
        if len(self.history) >= self.dilen + self.adxlen - 1:
            self.live = self._internal
        else:
            self.live = None

    def _compute_step(self, candle, prev_chain: "Adx | None",
                       prev1: "Adx | None", prev2: "Adx | None") -> "Adx":
        # RMA smoothing factors.
        di_alpha = 1 / self.dilen
        adx_alpha = 1 / self.adxlen

        if prev_chain is None:
            # Initialize the first state without a previous candle.
            tr = candle.high - candle.low
            plus_dm = 0.0
            minus_dm = 0.0

            tr_rma = tr
            plus_dm_rma = plus_dm
            minus_dm_rma = minus_dm
        else:
            # Directional movement.
            up = candle.high - prev_chain.high
            down = prev_chain.low - candle.low

            plus_dm = up if (up > down and up > 0) else 0.0
            minus_dm = down if (down > up and down > 0) else 0.0

            # True Range.
            tr = max(
                candle.high - candle.low,
                abs(candle.high - prev_chain.close),
                abs(candle.low - prev_chain.close),
            )

            # Continue the RMA chain.
            tr_rma = tr * di_alpha + prev_chain.tr_rma * (1 - di_alpha)
            plus_dm_rma = plus_dm * di_alpha + prev_chain.plus_dm_rma * (1 - di_alpha)
            minus_dm_rma = minus_dm * di_alpha + prev_chain.minus_dm_rma * (1 - di_alpha)

        # Directional Indicators.
        plus_di = 100 * plus_dm_rma / tr_rma if tr_rma != 0 else 0.0
        minus_di = 100 * minus_dm_rma / tr_rma if tr_rma != 0 else 0.0

        # Directional Index (DX).
        summ = plus_di + minus_di
        divisor = summ if summ != 0 else 1.0
        dx = abs(plus_di - minus_di) / divisor

        # Average Directional Index (ADX).
        if prev1 is None:
            adx_value = 100 * dx
        else:
            adx_value = (100 * dx) * adx_alpha + prev1.adx * (1 - adx_alpha)

        # ADX trend color.
        adx_color = "lime" if (prev1 is not None and adx_value > prev1.adx) else "red"

        # Trend reversal detection.
        if prev1 is not None and prev2 is not None:
            rule1 = adx_value < prev1.adx
            rule2 = prev1.adx > prev2.adx
            rule3 = prev1.adx > self.key_level
            is_reversal = rule1 and rule2 and rule3
        else:
            is_reversal = False

        # Preserve the ADX peak that triggered the reversal.
        reversal_level = prev1.adx if (is_reversal and prev1 is not None) else None

        return Adx(
            time=candle.time,
            start_ts=candle.start_ts,
            end_ts=candle.end_ts,
            adx=adx_value,
            plus_di=plus_di,
            minus_di=minus_di,
            adx_color=adx_color,
            is_reversal=is_reversal,
            reversal_level=reversal_level,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            tr_rma=tr_rma,
            plus_dm_rma=plus_dm_rma,
            minus_dm_rma=minus_dm_rma,
        )