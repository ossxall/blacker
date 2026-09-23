from pulsar import Client, ConsumerType, InitialPosition
from ingestion.tick import Tick
import msgpack

SCALE = 1e8

class PulsarConsumer:
    
    def __init__(
        self,
        service_url="pulsar://localhost:16650",
        topic="persistent://public/default/master.tick",
        subscription="engine-sub",
    ):
        self.client = Client(service_url)

        self.consumer = self.client.subscribe(
            topic,
            subscription_name=subscription,
            consumer_type=ConsumerType.Exclusive,
            initial_position=InitialPosition.Latest
        )

    def _decode_batch(self, msg):
        boot_id, config_id, first_tick_index, raw_ticks = msgpack.unpackb(
            msg.data(),
            raw=False,
        )

        ticks = [
            Tick(
                boot_id=boot_id,
                config_id=config_id,
                tick_index=first_tick_index + offset,
                trade_id=trade_id,
                time=time,
                price=price / SCALE,
                qty=qty / SCALE,
                is_buyer_maker=is_buyer_maker,
            )
            for offset, (trade_id, time, price, qty, is_buyer_maker)
            in enumerate(raw_ticks)
        ]

        return boot_id, first_tick_index, ticks

    def ack(self, msg):
        self.consumer.acknowledge(msg)

    def nack(self, msg):
        self.consumer.negative_acknowledge(msg)

    def listen(self, callback):
        print("Consumer listening")

        state = {
            "is_last": False
        }

        while True:
            msg = self.consumer.receive()

            try:
                _, _, ticks = self._decode_batch(msg)

                last = len(ticks) - 1

                for i, tick in enumerate(ticks):
                    state["is_last"] = i == last

                    sig = callback(state, tick)

                    if sig == "NACK":
                        self.nack(msg)
                        break 

                    if sig == "ACK":
                        continue
                    
                    if sig == "ACKALL":
                        self.ack(msg)
                        break
                else:
                    self.ack(msg)

            except Exception as e:
                self.nack(msg)
                print(f"Error listening message: {e}")

    def close(self):
        self.consumer.close()
        self.client.close()