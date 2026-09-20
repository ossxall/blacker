from src.series import ADXSeries
from src.series.registry import SERIES_REGISTRY
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
    adx = ADXSeries(
        id="adx",
        kind="ADXSeries",
        level=0,
        primary=False,
        overlay=True,
        params={},
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


def test_registry_instantiates_and_restores_like_candlestick():
    # ADXSeries must be registered the same way Candlestick is, so the
    # engine can build and restore an ADXSeries from serialized state.
    assert "ADXSeries" in SERIES_REGISTRY

    adx = SERIES_REGISTRY["ADXSeries"](
        id="adx",
        kind="ADXSeries",
        level=1,
        primary=False,
        overlay=True,
        params={"dilen": 14, "adxlen": 14, "key_level": 23},
    )

    state = {
        "id": "adx",
        "kind": "ADXSeries",
        "level": 1,
        "primary": False,
        "overlay": True,
        "params": {"dilen": 14, "adxlen": 14, "key_level": 23},
        "live": None,
        "history": [],
    }

    adx.set_state(state)

    assert adx.id == "adx"
    assert adx.kind == "ADXSeries"
    assert adx.dilen == 14
    assert adx.adxlen == 14
    assert adx.key_level == pytest.approx(23.0)
    assert adx.live is None
    assert adx._internal is None