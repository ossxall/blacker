# BLACKER
# Copyright (C) 2026 Juan José Caballero Rey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import time
import logging
import requests
from ingestion.tick import Tick
from core.engine import TradingEngine
from ingestion.pulsar_consumer import PulsarConsumer
from publication.pulsar_publisher import PulsarPublisher
from enum import StrEnum

#-----------------------------------------------------------------------------------------------------------------------
# IMPLEMENTATION
#-----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

consumer = PulsarConsumer(
        service_url="pulsar://localhost:16650",
        topic="persistent://public/default/master.tick",
        subscription="engine-sub",
)

publisher = PulsarPublisher(
    service_url="pulsar://localhost:16650",
    topic="persistent://public/default/engine.state",
)

engine = TradingEngine()

#-----------------------------------------------------------------------------------------------------------------------
# HANDLE TICKS
#-----------------------------------------------------------------------------------------------------------------------

class TickResponse(StrEnum):
    ACK = "ACK"
    NACK = "NACK"
    ACKALL= "ACKALL"

def handle_tick(cstate: dict, tick: Tick):
    # ----------------------------------------------------------
    # FUNCTIONS
    # ----------------------------------------------------------

    def _fetch_state():
        try:
            response = requests.get(
                "http://localhost:3002/api/backtest/master/get-state",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting backtest state: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing response JSON: {e}")
            return None

    def _sync_master_state():
        res = _fetch_state()
        if res is None:
            print("Master endpoint error.")
            return False

        try:
            engine.set_state(
                boot_id=res["boot_id"],
                config_id=res["master"]["config_id"],
                engine_state=res["master"]["engine_state"],
            )
        except Exception:
            # The engine is unusable until the master sends a state it can
            # actually build, so nothing half built may be left behind for
            # the next tick to pick up.
            logger.exception("master state rejected by set_state, resetting the engine")
            engine.reset()
            raise

        engine.status = "ready"
        return True

    # ----------------------------------------------------------
    # BOOTSTRAP
    # ----------------------------------------------------------

    if engine.status == "init":
        if not _sync_master_state():
            return TickResponse.NACK
       
    if engine.status != "ready":
        print("Engine is not ready.")
        return TickResponse.NACK


    # ----------------------------------------------------------
    # OP VALIDATION
    # ----------------------------------------------------------

    is_strange = (
        engine.boot_id != tick.boot_id
        or engine.config_id != tick.config_id
    )

    if is_strange:
        print(f"Incorrect boot_id / config_id ")

        is_old = (
            tick.boot_id < engine.boot_id 
            or tick.config_id < engine.config_id
        )

        if is_old:
            return TickResponse.ACKALL

        engine.reset()
        return TickResponse.NACK

    # ----------------------------------------------------------
    # TICK SEQUENCE
    # ----------------------------------------------------------

    if engine.state.tick_index > 0:
        if tick.tick_index - 1 != engine.state.tick_index:
            print(
                f"Incorrect engine.state.tick_index={engine.state.tick_index}, "
                f"tick.tick_index={tick.tick_index}, expected={engine.state.tick_index + 1}"
            )

            return TickResponse.NACK

    # ----------------------------------------------------------
    # PROCESS
    # ----------------------------------------------------------
    # A failing tick leaves the engine half applied: the bars, the
    # indicators and any fill it already booked are permanent, and the
    # signal it was about to produce is lost. None of that can be rolled
    # back, so the only sound recovery is to drop every bit of engine
    # state and resync from the master checkpoint.
    #
    # NACKing redelivers the whole batch. The master publishes the engine
    # state only on the last tick of a batch, so its checkpoint always
    # sits on a message boundary and the redelivered batch lines up with
    # the state we resync to. Without this reset the sequence guard below
    # would reject that redelivery forever, because a stale tick_index no
    # longer matches the start of the batch.

    try:
        new_state, signal = engine.on_tick(tick)
    except Exception:
        logger.exception(
            "tick %s failed, resetting the engine and nacking the batch",
            tick.tick_index,
        )
        engine.reset()
        return TickResponse.NACK

    if signal:
        print("SIGNAL:", signal)

    if tick.tick_index % 100_000 == 0:
        print(f"Processed: {tick.tick_index}")


    # ----------------------------------------------------------
    # LAST
    # ----------------------------------------------------------

    if cstate.get("is_last"):
        publisher.publish(new_state)

#-----------------------------------------------------------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------------------------------------------------------

def main():
    consumer.listen(handle_tick)

main()