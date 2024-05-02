import os

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from .logger import Logger

logger = Logger(True).get_logger(__name__)


class KafkaService:
    def __init__(self) -> None:
        topic_name = os.environ.get('TOPIC_NAME')
        admin_client = KafkaAdminClient(bootstrap_servers="kafka:9092")
        topics = [NewTopic(topic_name, num_partitions=3, replication_factor=1)]

        try:
            admin_client.create_topics(topics)
        except TopicAlreadyExistsError as e:
            logger.error(f"Topic '{topic_name}' already exists.")
