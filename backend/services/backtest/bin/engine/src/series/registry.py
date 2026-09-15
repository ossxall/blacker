from series.Candlestick import Candlestick
from series.EMA import EMA
from series.ReversalTrap import ReversalTrap

SERIES_REGISTRY = {
    "Candlestick": Candlestick,
    "EMA": EMA,
    "ReversalTrap": ReversalTrap,
}