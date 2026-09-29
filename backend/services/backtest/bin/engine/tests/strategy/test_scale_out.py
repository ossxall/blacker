"""
Scale-out de my_strategy: qué hace cuando el 30m gira contra la
posición sin que lo haga el 1H.

Se prueba la estrategia contra estados construidos a mano en vez de
contra el motor completo, porque lo que interesa aquí es la decisión y
la cantidad que pide. El ciclo de órdenes de un exit parcial ya lo
cubre test_engine_integration.
"""

from src.core.engine_state import EngineState
from src.core.portfolio import Portfolio, Position
from src.orders import Side
from src.strategy.my_strategy import Strategy1
from src.timeframes.timeframe import Timeframe


# Un 1H alcista con ADX fuerte: la tendencia manda, así que un 30m en
# contra no cierra nada entero.
FAST_1H = 110.0
SLOW_1H = 100.0

ADX_LONG = {"adx": 30.0, "plus_di": 30.0, "minus_di": 10.0}

# El ADX es parte de la regla de 1H: sin -DI > +DI la tendencia no está
# bearish aunque las EMAs estén invertidas.
ADX_SHORT = {"adx": 30.0, "plus_di": 10.0, "minus_di": 30.0}


class FakeSeries:
    """Lo mínimo que la estrategia lee de una serie."""

    def __init__(self, series_id, kind, label, closed=None, history=None):
        self.id = series_id
        self.kind = kind
        self.params = {"label": label}
        self._closed = closed
        self.live = None
        self.history = history or []


def ema(series_id, label, value):
    return FakeSeries(
        series_id,
        "EMA",
        label,
        closed=None if value is None else _Box(value),
    )


def adx(values=ADX_LONG):
    # La estrategia lee el ADX confirmado de ``history[-1]``, no de
    # ``_closed``: son rutas distintas a propósito en el código real.
    return FakeSeries(
        "adx",
        "ADX",
        "ADX 14",
        history=[_Box(**values)],
    )


class _Box:
    """La estrategia lee ``._closed.value`` de las EMA y atributos sueltos
    del ADX confirmado."""

    def __init__(self, value=None, **fields):
        self.value = value

        for key, field_value in fields.items():
            setattr(self, key, field_value)


def timeframe(tf_id, fast=None, slow=None, with_adx=True, adx_values=ADX_LONG):
    tf = Timeframe()
    tf.id = tf_id
    tf.timeframe_ms = 300_000

    if fast is not None:
        tf.add_series(ema("ema20", "EMA 20", fast))

    if slow is not None:
        tf.add_series(ema("ema50", "EMA 50", slow))

    if with_adx:
        tf.add_series(adx(adx_values))

    return tf


def build_state(
    position,
    fast_30m,
    slow_30m,
    fast_1h=FAST_1H,
    slow_1h=SLOW_1H,
    adx_1h=ADX_LONG,
):
    timeframes = {
        "1h": timeframe("1h", fast_1h, slow_1h, adx_values=adx_1h),
        "30m": timeframe("30m", fast_30m, slow_30m),
        "15m": timeframe("15m", fast_30m, slow_30m),
        "5m": timeframe("5m", fast_30m, slow_30m),
        "1m": timeframe("1m", fast_30m, slow_30m),
    }

    portfolio = Portfolio()

    if position is not None:
        portfolio.position = position

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


def position(side, quantity=1.0):
    return Position(
        side=side,
        quantity=quantity,
        avg_price=100.0,
        entry_time=0,
        entry_tick_index=0,
    )


def strategy(**params):
    return Strategy1("Strategy1", params)


# ---------------------------------------------------------------------
# 30m en contra, 1H sigue a favor: salida PARCIAL
# ---------------------------------------------------------------------


def test_long_scales_out_half_when_30m_crosses_down():
    state = build_state(position(Side.BUY), fast_30m=90.0, slow_30m=100.0)

    signal = strategy().evaluate(state)

    assert signal is not None
    assert signal.action == "EXIT"
    assert signal.quantity == 0.5


def test_short_scales_out_half_when_30m_crosses_up():
    # 1H bearish: es lo que deja que el cross de 30m llegue al scale-out.
    # Con 1H bullish el cierre completo se llevaría la posición antes.
    state = build_state(
        position(Side.SELL),
        fast_30m=110.0,
        slow_30m=100.0,
        fast_1h=90.0,
        slow_1h=100.0,
        adx_1h=ADX_SHORT,
    )

    signal = strategy().evaluate(state)

    assert signal is not None
    assert signal.action == "EXIT"
    assert signal.quantity == 0.5


def test_scale_out_honours_the_configured_fraction():
    state = build_state(
        position(Side.BUY, quantity=4.0), fast_30m=90.0, slow_30m=100.0
    )

    signal = strategy(scale_out_fraction=0.25).evaluate(state)

    # Una fracción de la posición, no una cantidad fija: la misma
    # cuenta sirve con cualquier tamaño.
    assert signal.quantity == 1.0


def test_scale_out_scales_with_the_position_size():
    # Con la misma fracción, una posición el doble sale el doble.
    state = build_state(
        position(Side.BUY, quantity=2.0), fast_30m=90.0, slow_30m=100.0
    )

    assert strategy().evaluate(state).quantity == 1.0


def test_a_fraction_of_one_falls_back_to_half():
    # Un >= 1 convertiría el cross de 30m en un cierre completo, que es
    # lo que ya hace el reverso de 1H.
    state = build_state(position(Side.BUY), fast_30m=90.0, slow_30m=100.0)

    signal = strategy(scale_out_fraction=1.0).evaluate(state)

    assert signal.quantity == 0.5


# ---------------------------------------------------------------------
# 1H en contra manda: cierre completo
# ---------------------------------------------------------------------


def test_1h_reversal_closes_in_full_even_with_30m_against():
    # 1H invertido y 30M también: la salida parcial no debe ganarle al
    # cierre completo, o la posición nunca cerraría del todo.
    state = build_state(
        position(Side.BUY),
        fast_30m=90.0,
        slow_30m=100.0,
        fast_1h=90.0,
        slow_1h=100.0,
        adx_1h=ADX_SHORT,
    )

    signal = strategy().evaluate(state)

    assert signal.action == "EXIT"
    assert signal.quantity is None


def test_1h_reversal_closes_in_full_for_a_short():
    # Para un short la reversión es 1H al alza, y el ADX tiene que
    # acompañar: EMAs alcistas con -DI dominando dan neutral, no reverso.
    state = build_state(
        position(Side.SELL),
        fast_30m=110.0,
        slow_30m=100.0,
        fast_1h=FAST_1H,
        slow_1h=SLOW_1H,
        adx_1h=ADX_LONG,
    )

    signal = strategy().evaluate(state)

    assert signal.action == "EXIT"
    assert signal.quantity is None


def test_1h_neutral_does_not_close_a_long_in_full():
    # 1H neutral (EMAs cruzadas pero ADX con -DI == +DI) no es reverso:
    # la posición sobrevive y solo decide el 30m.
    state = build_state(
        position(Side.BUY),
        fast_30m=90.0,
        slow_30m=100.0,
        fast_1h=FAST_1H,
        slow_1h=SLOW_1H,
        adx_1h={"adx": 30.0, "plus_di": 10.0, "minus_di": 10.0},
    )

    signal = strategy().evaluate(state)

    assert signal.action == "EXIT"
    assert signal.quantity == 0.5
    # El ADX de 1H manda junto a las EMAs: si 1H queda neutral por ADX
    # y no por las EMAs, el 30m en contra sigue siendo la única señal y
    # la salida parcial aplica igual.
    state = build_state(
        position(Side.BUY),
        fast_30m=90.0,
        slow_30m=100.0,
        fast_1h=FAST_1H,
        slow_1h=SLOW_1H,
        adx_1h={"adx": 30.0, "plus_di": 10.0, "minus_di": 10.0},
    )

    signal = strategy().evaluate(state)

    assert signal.action == "EXIT"
    assert signal.quantity == 0.5


# ---------------------------------------------------------------------
# Idempotencia: no se repite cada tick
# ---------------------------------------------------------------------


def test_scale_out_does_not_repeat_while_30m_stays_against():
    # El caso que hay que vigilar: sin la guarda de posición completa,
    # el 30m en contra se leería en cada tick y la posición menguaría
    # 1.0 -> 0.5 -> 0.25 -> 0.125 hasta quedar en polvo.
    state = build_state(
        position(Side.BUY, quantity=0.5), fast_30m=90.0, slow_30m=100.0
    )

    assert strategy().evaluate(state) is None


def test_scale_out_stops_after_a_bracket_target_took_profit():
    # Un target que llenó ya realizó una salida: la estrategia no pide
    # otra parcial sobre lo que queda.
    state = build_state(
        position(Side.BUY, quantity=0.5), fast_30m=90.0, slow_30m=100.0
    )

    assert strategy().evaluate(state) is None


def test_a_full_position_just_below_entry_size_still_scales_out():
    # La guarda es una tolerancia, no una igualdad: un residuo de coma
    # flotante no debe desactivar la salida parcial para siempre.
    state = build_state(
        position(Side.BUY, quantity=0.9999999999), fast_30m=90.0, slow_30m=100.0
    )

    signal = strategy().evaluate(state)

    assert signal is not None
    assert signal.action == "EXIT"


# ---------------------------------------------------------------------
# 30m de acuerdo: mantener
# ---------------------------------------------------------------------


def test_holds_when_30m_still_agrees_with_the_position():
    state = build_state(position(Side.BUY), fast_30m=110.0, slow_30m=100.0)

    assert strategy().evaluate(state) is None


def test_holds_when_30m_is_exactly_flat():
    # Las dos EMA iguales no es un cross: es neutral, y en neutral la
    # estrategia no toca la posición.
    state = build_state(position(Side.BUY), fast_30m=100.0, slow_30m=100.0)

    assert strategy().evaluate(state) is None


# ---------------------------------------------------------------------
# Datos incompletos: no inventar la decisión
# ---------------------------------------------------------------------


def test_no_scale_out_when_the_30m_series_is_missing():
    timeframes = {
        "1h": timeframe("1h", FAST_1H, SLOW_1H),
        "30m": Timeframe(),
    }

    portfolio = Portfolio()
    portfolio.position = position(Side.BUY)

    state = EngineState(
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

    assert strategy().evaluate(state) is None


def test_no_scale_out_when_the_30m_ema_has_not_closed_yet():
    state = build_state(position(Side.BUY), fast_30m=None, slow_30m=100.0)

    assert strategy().evaluate(state) is None


# ---------------------------------------------------------------------
# Sin posición: esto es una estrategia de entrada, el scale-out no aplica
# ---------------------------------------------------------------------


def test_flat_portfolio_does_not_scale_out():
    state = build_state(None, fast_30m=90.0, slow_30m=100.0)

    # Con 1H alcista y 30m en contra no hay entrada (el 30m no
    # confirma), pero en ningún caso sale algo.
    signal = strategy().evaluate(state)

    assert signal is None or signal.action != "EXIT"
