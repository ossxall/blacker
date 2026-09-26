from series.ADX import ADX
from series.ATR import ATR
from series.Candlestick import Candlestick
from series.EMA import EMA
from series.ReversalTrap import ReversalTrap

SERIES_REGISTRY = {
    "ADX": ADX,
    "ATR": ATR,
    "Candlestick": Candlestick,
    "EMA": EMA,
    "ReversalTrap": ReversalTrap,
}