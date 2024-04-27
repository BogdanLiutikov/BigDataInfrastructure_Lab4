import json
from time import sleep
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from .database import Database
from .schemas import PredictedModel

from .logger import Logger

logger = Logger(True).get_logger(__name__)


class KafkaConsumerImpl:
    def __init__(self, topic=None) -> None:
        self.topic = 'main'
        if topic is None:
            topic = self.topic
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=['kafka:9092'],
                                      group_id='Lab4',
                                      auto_offset_reset='earliest',
                                    #   consumer_timeout_ms=1000
                                      )
        # for message in self.consumer:
        #     print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
        #                                          message.offset, message.key,
        #                                          message.value))


class KafkaConsumerDataBase:
    def __init__(self, database: Database, topic=None) -> None:
        # super().__init__(topic)
        self.database = database
        self.topic = 'main'
        if topic is None:
            topic = self.topic
        self.consumer = KafkaConsumer(topic,
                                      bootstrap_servers=['kafka:9092'],
                                      group_id='Lab4',
                                      auto_offset_reset='earliest',
                                    #   consumer_timeout_ms=1000
                                      )

    def listen(self):
        logger.info('Start listen')
        try:
            while True:
                print(f'{self.consumer.bootstrap_connected()=}')
                messages = self.consumer.poll(timeout_ms=1000)
                print('consumer_message:', messages, flush=True)
                if messages is None or messages == {}:
                    sleep(3)
                    continue

                for topic_partition, records in messages.items():
                    session = next(self.database.get_session())
                    for record in records:
                        value = json.loads(record.value.decode())
                        predict = PredictedModel(x=value.get('x'), y_pred=value.get('y_pred'), y_true=value.get('y_true'))
                        self.database.create_record(session, predict)
        except Exception as e:
            logger.error(e, exc_info=e)
        finally:
            logger.info('Consumer close')
            self.consumer.close()


if __name__ == '__main__':
    consumer = KafkaConsumerDataBase()
    consumer.listen()
