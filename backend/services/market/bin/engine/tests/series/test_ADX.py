from src.series.ADX import ADX
import pytest

class Candle:
    def __init__(self, high, low, close):
        self.time = 1
        self.start_ts = 1
        self.end_ts = 2
        self.high = high
        self.low = low
        self.close = close


def test_compute_step_first_candle_initializes_state():
    adx = ADX(
        level=0,
        name="ADX",
        id="adx",
        source="candles",
    )

    candle = Candle(
        high=110.0,
        low=100.0,
        close=105.0,
    )

    result = adx._compute_step(
        candle=candle,
        prev_chain=None,
        prev1=None,
        prev2=None,
    )

    # True Range initializes from the candle range.
    assert result.tr_rma == pytest.approx(10.0)

    # Directional movement starts at zero.
    assert result.plus_dm_rma == pytest.approx(0.0)
    assert result.minus_dm_rma == pytest.approx(0.0)

    # Directional indicators are zero.
    assert result.plus_di == pytest.approx(0.0)
    assert result.minus_di == pytest.approx(0.0)

    # DX = 0, therefore the first ADX is also zero.
    assert result.adx == pytest.approx(0.0)

    # No trend information exists yet.
    assert result.adx_color == "red"
    assert result.is_reversal is False
    assert result.reversal_level is None

    # Internal state must preserve the candle values.
    assert result.high == 110.0
    assert result.low == 100.0
    assert result.close == 105.0