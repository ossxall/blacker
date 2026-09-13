import json
import msgpack
from strategy.base import Strategy
from dataclasses import dataclass
from typing import Optional

@dataclass
class EngineState:
    boot_id: str
    config_id: str
    tick_index: int
    time: int
    timeframes: dict
    strategy: Strategy
    portfolio: Optional[object] = None
    risk: Optional[dict] = None
    orders: Optional[dict] = None

    def to_dict(self):
        return {
            "boot_id": self.boot_id,
            "config_id": self.config_id,
            "tick_index": self.tick_index,
            "time": self.time,
            "timeframes": {
                key: timeframe.to_dict()
                for key, timeframe in self.timeframes.items()
            },
            "strategy": self.strategy.to_dict(),
            "portfolio": (
                self.portfolio.to_dict()
                if self.portfolio is not None
                else None
            ),
            "risk": self.risk or {},
            "orders": self.orders or None,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
        )

    def to_msgpack(self) -> bytes:
        return msgpack.packb(
            self.to_dict(),
            use_bin_type=True,
        )