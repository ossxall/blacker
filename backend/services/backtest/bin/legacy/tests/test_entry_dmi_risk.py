import numpy as np
import pandas as pd
import pytest

from strategy.config import Config
from strategy.strategy import Backtester, EMACrossStrategy


def make_bars(periods=3):
    index = pd.date_range("2026-01-01", periods=periods, freq="30min", tz="UTC")
    close = np.linspace(100.0, 102.0, periods)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def test_entry_dmi_risk_uses_directional_spread_from_prior_signal_bar():
    bars = make_bars()
    signals = pd.DataFrame(
        {
            "valid": True,
            "target": 0,
            "trend_bias": 0,
            "entry_plus_di": [10.0, 30.0, 90.0],
            "entry_minus_di": [10.0, 10.0, 0.0],
        },
        index=bars.index,
    )
    engine = Backtester(bars, Config(timeframe="30min"), signals=signals)
    engine._bar_index = 2  # Entrada a la apertura de la vela 2, señal en la 1.

    assert engine._entry_dmi_risk_multiplier(1) == pytest.approx(1.1)
    assert engine._entry_dmi_risk_multiplier(-1) == pytest.approx(0.7)


def test_entry_dmi_risk_multiplier_is_clamped_and_can_be_disabled():
    bars = make_bars()
    signals = pd.DataFrame(
        {
            "valid": True,
            "target": 0,
            "trend_bias": 0,
            "entry_plus_di": [10.0, 100.0, 0.0],
            "entry_minus_di": [10.0, 0.0, 100.0],
        },
        index=bars.index,
    )
    engine = Backtester(bars, Config(timeframe="30min"), signals=signals)
    engine._bar_index = 2
    assert engine._entry_dmi_risk_multiplier(1) == pytest.approx(1.25)
    assert engine._entry_dmi_risk_multiplier(-1) == pytest.approx(0.5)

    disabled = Config(timeframe="30min", entry_dmi_risk_scaling_enabled=False)
    no_scale = Backtester(bars, disabled, signals=signals)
    no_scale._bar_index = 2
    assert no_scale._entry_dmi_risk_multiplier(1) == 1.0


def test_generated_signals_expose_di_values_without_filtering_entries():
    bars = make_bars(220)
    enabled = Config(timeframe="30min", streak_trend_only=False)
    disabled = Config(
        timeframe="30min",
        streak_trend_only=False,
        entry_dmi_risk_scaling_enabled=False,
    )

    signals_with_dmi = EMACrossStrategy(enabled).generate_signals(bars)
    signals_without_scaling = EMACrossStrategy(disabled).generate_signals(bars)

    assert {"entry_plus_di", "entry_minus_di"} <= set(signals_with_dmi.columns)
    assert "entry_plus_di" not in signals_without_scaling
    assert np.array_equal(signals_with_dmi.target, signals_without_scaling.target)
