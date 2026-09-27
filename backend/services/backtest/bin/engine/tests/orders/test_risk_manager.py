from src.orders import RiskManager, Side
import pytest


def long_stop(risk: RiskManager, price: float = 100.0):
    return risk.apply(Side.BUY, price).stop_price


def test_default_stop_protects_every_entry():
    assert long_stop(RiskManager()) == 99.0
    assert long_stop(RiskManager({})) == 99.0
    assert long_stop(RiskManager({"targets": [{"type": "percent", "value": 0.02}]})) == 99.0


def test_explicit_stop_overrides_the_default():
    assert long_stop(RiskManager({"stop": {"type": "absolute", "value": 5.0}})) == 95.0
    assert long_stop(RiskManager({"stop": {"type": "percent", "value": 0.02}})) == 98.0


def test_stop_and_targets_are_mirrored_for_a_short_entry():
    levels = RiskManager(
        {
            "stop": {"type": "percent", "value": 0.01},
            "targets": [{"type": "percent", "value": 0.02}],
        }
    ).apply(Side.SELL, 100.0)

    assert levels.stop_price == 101.0
    assert levels.target_prices == [98.0]


def test_targets_are_ordered_nearest_first():
    levels = RiskManager(
        {
            "targets": [
                {"type": "percent", "value": 0.05},
                {"type": "percent", "value": 0.02},
                {"type": "percent", "value": 0.03},
            ]
        }
    ).apply(Side.BUY, 100.0)

    # A tick that jumps through several targets must fill the closest one.
    assert levels.target_prices == [102.0, 103.0, 105.0]

    short = RiskManager(
        {
            "targets": [
                {"type": "percent", "value": 0.05},
                {"type": "percent", "value": 0.02},
                {"type": "percent", "value": 0.03},
            ]
        }
    ).apply(Side.SELL, 100.0)

    assert short.target_prices == [98.0, 97.0, 95.0]


def test_zero_offset_never_parks_the_stop_on_the_entry():
    # A stop at exactly the entry price is filled by the very next tick. The
    # same reasoning that rejects an ATR offset of 0.0 applies to a percent
    # or absolute one: the default is used instead.
    assert long_stop(RiskManager({"stop": {"type": "percent", "value": 0.0}})) == 99.0
    assert long_stop(RiskManager({"stop": {"type": "absolute", "value": 0.0}})) == 99.0
    assert long_stop(RiskManager({"stop": {"type": "percent", "value": -0.01}})) == 99.0


def test_a_stop_beyond_zero_is_rejected_in_favour_of_the_default():
    risk = RiskManager({"stop": {"type": "absolute", "value": 150.0}})

    assert long_stop(risk) == 99.0


def test_targets_that_resolve_to_a_non_positive_price_are_dropped():
    levels = RiskManager(
        {
            "targets": [
                {"type": "absolute", "value": 5.0},
                {"type": "absolute", "value": 500.0},
            ]
        }
    ).apply(Side.SELL, 100.0)

    assert levels.target_prices == [95.0]


def test_atr_offset_uses_a_static_context():
    risk = RiskManager(
        {"stop": {"type": "atr", "multiplier": 2.0}},
        context={"atr": 1.5},
    )

    assert long_stop(risk) == 97.0


def test_atr_offset_uses_a_live_provider():
    readings = [2.0, 4.0]
    risk = RiskManager(
        {"stop": {"type": "atr", "multiplier": 2.0}},
        atr_provider=lambda: readings.pop(0),
    )

    assert long_stop(risk) == 96.0
    assert long_stop(risk) == 92.0


def test_atr_without_a_reading_falls_back_and_warns():
    risk = RiskManager({"stop": {"type": "atr", "multiplier": 2.0}})

    with pytest.warns(RuntimeWarning, match="atr"):
        assert long_stop(risk) == 99.0


def test_a_failing_atr_provider_does_not_break_the_bracket():
    def boom():
        raise RuntimeError("no series")

    risk = RiskManager(
        {"stop": {"type": "atr", "multiplier": 2.0}},
        atr_provider=boom,
    )

    with pytest.warns(RuntimeWarning, match="ATR provider"):
        assert long_stop(risk) == 99.0


def test_unknown_offset_type_is_reported():
    # Regression: a typo used to degrade silently to the default stop, so a
    # misconfigured bracket looked correctly protected.
    risk = RiskManager({"stop": {"type": "pct", "value": 0.02}})

    with pytest.warns(RuntimeWarning, match="Unknown risk offset type"):
        assert long_stop(risk) == 99.0


def test_warnings_are_emitted_once_per_spec():
    risk = RiskManager({"stop": {"type": "pct", "value": 0.02}})

    with pytest.warns(RuntimeWarning):
        long_stop(risk)

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        long_stop(risk)
        long_stop(risk)

    assert caught == []


def test_trailing_keeps_a_relative_spec():
    risk = RiskManager(
        {
            "trailing": {
                "enabled": True,
                "distance": {"type": "percent", "value": 0.01},
            }
        }
    )

    levels = risk.apply(Side.BUY, 100.0)

    assert levels.trailing is not None
    assert levels.trailing.kind == "percent"
    assert risk.trailing_distance(levels.trailing, 100.0) == 1.0
    assert risk.trailing_distance(levels.trailing, 250.0) == 2.5


def test_trailing_spec_round_trips_through_a_dict():
    risk = RiskManager(
        {
            "trailing": {
                "enabled": "true",
                "distance": {"type": "atr", "multiplier": 3.0},
            }
        },
        context={"atr": 2.0},
    )

    levels = risk.apply(Side.BUY, 100.0)

    spec = levels.trailing.to_dict()

    assert spec == {"kind": "atr", "value": 0.0, "multiplier": 3.0}
    assert risk.trailing_distance(spec, 100.0) == 6.0


def test_trailing_is_disabled_when_enabled_is_false():
    risk = RiskManager(
        {
            "trailing": {
                "enabled": False,
                "distance": {"type": "percent", "value": 0.01},
            }
        }
    )

    assert risk.apply(Side.BUY, 100.0).trailing is None


def test_trailing_without_a_distance_is_reported():
    risk = RiskManager({"trailing": {"enabled": True}})

    with pytest.warns(RuntimeWarning, match="no distance"):
        assert risk.apply(Side.BUY, 100.0).trailing is None


def test_trailing_with_an_unknown_distance_type_is_reported():
    risk = RiskManager(
        {
            "trailing": {
                "enabled": True,
                "distance": {"type": "pct", "value": 0.01},
            }
        }
    )

    with pytest.warns(RuntimeWarning, match="Unknown trailing distance type"):
        assert risk.apply(Side.BUY, 100.0).trailing is None


def test_trailing_distance_is_zero_without_a_spec():
    risk = RiskManager()

    assert risk.trailing_distance(None, 100.0) == 0.0
    assert risk.trailing_distance({"kind": "percent", "value": 0.01}, 0.0) == 0.0


def test_entry_price_must_be_a_positive_number():
    for price in (0, -1.0, None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RiskManager().apply(Side.BUY, price)
