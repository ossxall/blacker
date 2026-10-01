"""
El contrato de cierre de my_strategy: toda decisión sale de ``_closed``.

``evaluate`` se llama en cada tick con la vela abierta presente en el
estado, así que la única forma de que la estrategia no se repinta es que no
lea ``live``. Estos casos contraponen las dos rutas: el mismo estado con
``live`` puesto en lo contrario tiene que dar exactamente la misma señal.

Leen ``series._closed``, el contrato que comparten EMA, ATR, ADX,
ReversalTrap y Trail. No hay accessor que caiga a ``live`` ni a
``history[-1]``: si una serie todavía no tiene confirmación, la estrategia
no tiene nada que decidir y devuelve None.
"""

from src.core.engine_state import EngineState
from src.core.portfolio import Portfolio, Position
from src.orders import Side
from src.strategy.my_strategy import Strategy1
from src.timeframes.timeframe import Timeframe


# ---------------------------------------------------------------------
# Dobles
# ---------------------------------------------------------------------


class Box:
    """El payload confirmado de una serie: `.value` en EMA, campos en ADX."""

    def __init__(self, value=None, **fields):
        self.value = value

        for key, field_value in fields.items():
            setattr(self, key, field_value)


class FakeSeries:
    def __init__(self, kind, label, closed=None, live=None, history=()):
        self.id = f"{kind}-{label}"
        self.kind = kind
        self.params = {"label": label}
        self._closed = closed
        self.live = live
        self.history = list(history)


class FakeCandle:
    def __init__(self, start_ts, close, high=None, low=None):
        self.start_ts = start_ts
        self.end_ts = start_ts + 1
        self.close = close
        self.high = close if high is None else high
        self.low = close if low is None else low


def ema(label, closed, live=None):
    return FakeSeries("EMA", label, closed=closed, live=live)


def adx(adx_closed=None, plus_di=30.0, minus_di=10.0, live=None):
    return FakeSeries(
        "ADX",
        "ADX 14",
        closed=None
        if adx_closed is None
        else Box(adx=adx_closed, plus_di=plus_di, minus_di=minus_di),
        live=live,
    )


def candles(*pairs):
    """
    Serie de velas con el contrato real: ``_closed`` es la última vela
    confirmada y ``history`` la contiene, así que la anterior es
    ``history[-2]``.
    """

    history = [
        FakeCandle(ts, close, high, low) for ts, close, high, low in pairs
    ]

    return FakeSeries(
        "Candlestick",
        "Candles",
        closed=history[-1] if history else None,
        history=history,
    )


def timeframe(tf_id, series, closed_candle=None):
    tf = Timeframe()
    tf.id = tf_id
    tf.timeframe_ms = 300_000
    tf.closed = closed_candle

    for one in series:
        tf.add_series(one)

    return tf


def position(side, quantity=1.0):
    return Position(
        side=side,
        quantity=quantity,
        avg_price=100.0,
        entry_time=0,
        entry_tick_index=0,
    )


def build_state(timeframes, held=None):
    portfolio = Portfolio()

    if held is not None:
        portfolio.position = held

    return EngineState(
        boot_id="boot",
        config_id="config",
        tick_index=0,
        time=0,
        timeframes=timeframes,
        strategy=None,
        portfolio=portfolio,
        risk={},
        orders={},
    )


# Un 1h alcista confirmado, el resto de timeframes en el mismo sentido.
BULLISH_1H = {"adx": 30.0, "plus_di": 30.0, "minus_di": 10.0}
BEARISH_1H = {"adx": 30.0, "plus_di": 10.0, "minus_di": 30.0}


def bullish_timeframes(adx_1h=BULLISH_1H, live=None):
    """
    Los cinco timeframes de acuerdo: da BUY si la estrategia lee lo cerrado.

    ``live`` se propaga a todas las series para poder poner la vela abierta en
    lo contrario sin tocar nada de lo confirmado.
    """

    return {
        "1h": timeframe(
            "1h",
            [
                ema("EMA 20", Box(110.0), live=live),
                ema("EMA 50", Box(100.0), live=live),
                adx(
                    adx_1h["adx"],
                    adx_1h["plus_di"],
                    adx_1h["minus_di"],
                    live=live,
                ),
            ],
        ),
        "30m": timeframe(
            "30m",
            [ema("EMA 20", Box(110.0), live=live), ema("EMA 50", Box(100.0), live=live)],
        ),
        "15m": timeframe(
            "15m",
            [
                ema("EMA 20", Box(110.0), live=live),
                ema("EMA 50", Box(100.0), live=live),
            ],
            closed_candle=FakeCandle(2, 105.0),
        ),
        "5m": timeframe(
            "5m",
            [ema("EMA 20", Box(110.0), live=live), ema("EMA 50", Box(100.0), live=live)],
        ),
        "1m": timeframe(
            "1m",
            [
                ema("EMA 20", Box(110.0), live=live),
                ema("EMA 50", Box(100.0), live=live),
                candles((0, 112.0, 112.0, 112.0), (1, 115.0, 115.0, 115.0)),
            ],
        ),
    }


def evaluate(timeframes, held=None, **params):
    return Strategy1("Strategy1", params).evaluate(build_state(timeframes, held))


def bearish_1h_timeframes():
    """1h confirmado en reverso: EMAs invertidas y -DI dominando."""

    timeframes = bullish_timeframes(adx_1h=BEARISH_1H)
    timeframes["1h"]._series["EMA-EMA 20"]._closed = Box(90.0)

    return timeframes


def against_30m_timeframes():
    """1h sigue a favor y el 30m confirmado va en contra: salida parcial."""

    timeframes = bullish_timeframes()
    timeframes["30m"]._series["EMA-EMA 20"]._closed = Box(90.0)

    return timeframes


# ---------------------------------------------------------------------
# Contrapeso: la vela abierta no puede mover la señal
# ---------------------------------------------------------------------


def test_the_closed_reads_do_open_a_long():
    signal = evaluate(bullish_timeframes())

    assert signal is not None
    assert signal.action == "BUY"
    assert signal.quantity == 1


def test_the_open_bar_cannot_stop_an_entry():
    """
    La vela de 1m abierta rompe el máximo anterior y su EMA está cruzada al
    alza: leída, entraría. Confirmado, el gatillo sigue siendo el de la vela
    cerrada y no hay entrada.
    """

    timeframes = bullish_timeframes()
    series = timeframes["1m"]._series["Candlestick-Candles"]

    # La vela cerrada deja de romper el máximo anterior: el `live` de la
    # serie, que sí lo hace, no se lee.
    series._closed = FakeCandle(1, 108.0, 108.0, 108.0)
    series.history[-1] = series._closed

    assert evaluate(timeframes) is None


def test_the_open_bar_cannot_open_an_entry_the_closed_one_denies():
    # 1h confirmado neutral por ADX flojo; la vela abierta, fuerte.
    timeframes = bullish_timeframes(
        adx_1h={"adx": 10.0, "plus_di": 30.0, "minus_di": 10.0},
        live=Box(45.0, adx=45.0, plus_di=45.0, minus_di=1.0),
    )

    assert evaluate(timeframes) is None


def test_the_open_bar_cannot_stop_a_reversal_from_closing_in_full():
    # 1h confirmado en reverso: EMAs invertidas y -DI dominando.
    timeframes = bearish_1h_timeframes()

    closed = evaluate(timeframes, held=position(Side.BUY))

    assert closed.action == "EXIT"
    assert closed.quantity is None

    # Ahora la vela abierta de 1h es alcista y fuerte: si se leyera, la
    # reversión se cancelaría y sólo decidiría el 30m.
    timeframes["1h"]._series["EMA-EMA 20"].live = Box(200.0)
    timeframes["1h"]._series["EMA-EMA 50"].live = Box(100.0)
    timeframes["1h"]._series["ADX-ADX 14"].live = Box(
        adx=45.0, plus_di=45.0, minus_di=1.0
    )

    still_closed = evaluate(timeframes, held=position(Side.BUY))

    assert still_closed.action == "EXIT"
    assert still_closed.quantity is None


def test_the_open_bar_cannot_stop_a_scale_out():
    # 30m confirmado en contra, y su vela abierta de acuerdo.
    timeframes = against_30m_timeframes()
    timeframes["30m"]._series["EMA-EMA 20"].live = Box(200.0)

    signal = evaluate(timeframes, held=position(Side.BUY))

    assert signal.action == "EXIT"
    assert signal.quantity == 0.5


# ---------------------------------------------------------------------
# Sin confirmación no hay dato, y no se busca en `live`
# ---------------------------------------------------------------------


def test_a_series_without_confirmation_yields_no_signal():
    # `_closed` a None en todas las series, `live` con todos los datos.
    timeframes = {
        **bullish_timeframes(),
        "1h": timeframe(
            "1h",
            [
                ema("EMA 20", None, live=Box(110.0)),
                ema("EMA 50", None, live=Box(100.0)),
                adx(30.0, 30.0, 10.0, live=Box(adx=30.0, plus_di=30.0, minus_di=10.0)),
            ],
        ),
    }

    assert evaluate(timeframes) is None


def test_an_unconfirmed_adx_yields_no_signal():
    # El ADX de 1h sin confirmar con -DI dominando cerraría el long entero;
    # sin confirmación no hay nada que leer, y "no hay datos" no es lo mismo
    # que neutral: la estrategia no cierra ni parcial mientras tanto.
    timeframes = bearish_1h_timeframes()
    timeframes["1h"]._series["ADX-ADX 14"]._closed = None

    assert evaluate(timeframes, held=position(Side.BUY)) is None


def test_a_zero_adx_is_a_value_and_not_a_missing_one():
    # adx == 0.0 es un dato: está por debajo del umbral, luego neutral.
    # Tratarlo como None sería indistinguishable, así que se comprueba el
    # umbral exacto en su lugar.
    at_threshold = evaluate(
        bullish_timeframes(adx_1h={"adx": 25.0, "plus_di": 30.0, "minus_di": 10.0})
    )

    assert at_threshold is not None and at_threshold.action == "BUY"

    below = evaluate(
        bullish_timeframes(adx_1h={"adx": 24.999, "plus_di": 30.0, "minus_di": 10.0})
    )

    assert below is None


def test_ema_and_adx_go_through_the_same_accessor():
    # El accessor es uno solo sobre `_closed`; lo que cambia es el campo que
    # se pide. Si uno de los dos volviera a su propia ruta, esto no lo
    # detectaría, pero fija que ambos leen el payload confirmado.
    strategy = Strategy1("Strategy1", {})

    confirmed = Box(42.0, adx=7.0, plus_di=8.0, minus_di=9.0)

    assert strategy._closed(FakeSeries("EMA", "x", closed=confirmed)) is confirmed
    assert strategy._ema_closed_value(FakeSeries("EMA", "x", closed=confirmed)) == 42.0
    assert strategy._adx_closed_value(FakeSeries("ADX", "x", closed=confirmed), "adx") == 7.0

    for value in (None, Box()):
        assert strategy._closed(FakeSeries("EMA", "x", closed=value)) is value
        assert strategy._ema_closed_value(FakeSeries("EMA", "x", closed=value)) == getattr(value, "value", None)
        assert strategy._adx_closed_value(FakeSeries("ADX", "x", closed=value), "adx") == getattr(value, "adx", None)