from series.ADX import ADX
from series.Candlestick import Candlestick
from series.EMA import EMA
from series.ReversalTrap import ReversalTrap

SERIES_REGISTRY = {
    "ADX": ADX,
    "Candlestick": Candlestick,
    "EMA": EMA,
    "ReversalTrap": ReversalTrap,
}