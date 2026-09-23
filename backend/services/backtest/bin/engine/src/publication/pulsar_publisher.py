from core.engine import EngineState
from pulsar import Client, Producer

class PulsarPublisher:
    def __init__(
        self,
        service_url="pulsar://localhost:16650",
        topic="persistent://public/default/engine.state",
    ):
        self.client = Client(service_url)
        self.producer: Producer = self.client.create_producer(topic, batching_enabled=False)


    def publish(self, engine_state: EngineState):
        payload = engine_state.to_msgpack()

        size_mb = len(payload) / (1024 * 1024)

        print(f"EngineState message size: {size_mb:.3f} MB")

        self.producer.send(payload)


    def close(self):
        self.producer.flush()
        self.producer.close()
        self.client.close()