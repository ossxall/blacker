from series.ADX import ADX
from series.ATR import ATR
from series.BollingerBands import BollingerBands
from series.Candlestick import Candlestick
from series.EMA import EMA
from series.ReversalTrap import ReversalTrap
from series.RSI import RSI
from series.StochRSI import StochRSI

SERIES_REGISTRY = {
    "ADX": ADX,
    "ATR": ATR,
    "BollingerBands": BollingerBands,
    "Candlestick": Candlestick,
    "EMA": EMA,
    "ReversalTrap": ReversalTrap,
    "RSI": RSI,
    "StochRSI": StochRSI,
}