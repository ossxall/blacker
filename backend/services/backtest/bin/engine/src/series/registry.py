from series.ADXSeries import ADXSeries
from series.Candlestick import Candlestick
from series.EMA import EMA
from series.ReversalTrap import ReversalTrap

SERIES_REGISTRY = {
    "ADXSeries": ADXSeries,
    "Candlestick": Candlestick,
    "EMA": EMA,
    "ReversalTrap": ReversalTrap,
}