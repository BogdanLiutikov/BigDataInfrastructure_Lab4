import json
from time import sleep
from kafka import KafkaProducer
from kafka.errors import KafkaError

from .logger import Logger

logger = Logger(True).get_logger(__name__)


class KafkaProducerImpl:
    def __init__(self) -> None:
        self.producer = KafkaProducer(bootstrap_servers=['kafka:9092'])
        self.topic = 'main'

    def send(self, message, key=None, topic=None):

        message = message.encode()

        if topic is None:
            topic = self.topic

        future = self.producer.send(topic, key=key, value=message)
        try:
            record_metadata = future.get(timeout=10)
        except KafkaError as e:
            logger.error('KafkaError', exc_info=e)
            return

        logger.info(f'Kafka Producer send topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}')


if __name__ == '__main__':
    producer = KafkaProducerImpl()
    for i in range(10):
        producer.send(f'i - {i}'.encode())
        sleep(1)