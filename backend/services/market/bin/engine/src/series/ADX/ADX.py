from collections import deque
from dataclasses import asdict, dataclass
from series.series import Series
from ingestion.tick import Tick

MAX_HISTORY_LEN = 500

@dataclass(frozen=True)
class Adx:
    time: int
    start_ts: int
    end_ts: int

    # ── Salida pública (equivalente a las columnas del DataFrame de adx.py) ──
    adx: float
    plus_di: float
    minus_di: float
    adx_color: str
    is_reversal: bool
    reversal_level: float | None

    # ── Estado interno de continuidad (necesario para calcular el siguiente valor) ──
    high: float
    low: float
    close: float
    tr_rma: float
    plus_dm_rma: float
    minus_dm_rma: float

class ADX(Series):
    def __init__(self, level: int, name: str, id: str, source: str,
                 dilen: int = 14, adxlen: int = 14, key_level: float = 23):
        super().__init__(level, name, id)

        self.source = source
        self.dilen = dilen
        self.adxlen = adxlen
        self.key_level = key_level

        self._internal: Adx | None = None  # estado real, nunca suprimido
        self.live: Adx | None = None       # estado visible (None durante warm-up)

        self.history: deque[Adx] = deque(maxlen=MAX_HISTORY_LEN)

    def to_dict(self):
        return {
            "params": {
                "level": self.level,
                "name": self.name,
                "id": self.id,
                "source": self.source,
                "dilen": self.dilen,
                "adxlen": self.adxlen,
                "key_level": self.key_level,
            },
            "live": asdict(self.live) if self.live is not None else None,
            "history": [asdict(a) for a in self.history],
        }

    def set_state(self, state: dict) -> None:
        self.history = deque(
            (Adx(**a) for a in state["history"]),
            maxlen=MAX_HISTORY_LEN,
        )

        self.live = (
            Adx(**state["live"]) if state["live"] is not None else None
        )

        # _internal se reconstruye desde live, o desde el último history
        # si live está suprimido por warm-up
        self._internal = self.live or (self.history[-1] if self.history else None)

    def update(self, tick: Tick) -> None:
        source = self.timeframe.get_series(self.source)

        # Wait until the source has produced its first candle.
        if source.live is None:
            return

        candle = source.live

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