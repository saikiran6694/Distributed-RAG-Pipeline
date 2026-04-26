"""
Stage 3: Kafka producer.
Publishes validated DocumentIngestionMessage objects to the raw-documents topic.

Design decisions:
  - acks='all'  → waits for all in-sync replicas before confirming
  - Manual delivery confirmation via callback
  - Partition key = source_type (ordering within a source)
  - Messages are JSON-serialized Pydantic models
"""

from __future__ import annotations

import logging
from threading import Event

from confluent_kafka import KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from shared.config import get_settings
from shared.models import DocumentIngestionMessage

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionProducer:
    """
    Thread-safe Kafka producer for the ingestion pipeline.
    One instance per service — reuse across requests.
    """

    def __init__(self):
        self._producer = Producer({
            "bootstrap.servers":   settings.KAFKA_BOOTSTRAP_SERVERS,
            "acks":                "1",             # single broker — leader ack sufficient
            "retries":             5,
            "retry.backoff.ms":    200,
            "enable.idempotence":  False,           # disabled — single broker, not needed
            "compression.type":    "snappy",        # reduce wire size
            "linger.ms":           10,              # small batching window
            "batch.size":          65536,
        })

    def publish(self, message: DocumentIngestionMessage) -> None:
        """
        Synchronously publish a message and wait for delivery confirmation.
        Raises KafkaException if delivery fails after retries.
        """
        payload = message.model_dump_json().encode("utf-8")
        partition_key = message.source_type.encode("utf-8")
        delivery_event = Event()
        delivery_error: list[Exception] = []   # mutable container for callback closure

        def on_delivery(err, kafka_msg):
            if err:
                delivery_error.append(KafkaException(err))
                logger.error(
                    "Kafka delivery failed",
                    extra={"doc_id": str(message.doc_id), "error": str(err)},
                )
            else:
                logger.debug(
                    "Kafka delivery confirmed",
                    extra={
                        "doc_id":    str(message.doc_id),
                        "topic":     kafka_msg.topic(),
                        "partition": kafka_msg.partition(),
                        "offset":    kafka_msg.offset(),
                    },
                )
            delivery_event.set()

        self._producer.produce(
            topic=settings.KAFKA_TOPIC_RAW_DOCUMENTS,
            key=partition_key,
            value=payload,
            on_delivery=on_delivery,
        )

        # Poll continuously until delivery callback fires (max 30s)
        # poll(0) only checks once — must loop to actually trigger callbacks
        import time
        deadline = time.monotonic() + 30
        while not delivery_event.is_set():
            self._producer.poll(0.5)   # poll for 500ms, triggers callbacks
            if time.monotonic() > deadline:
                break

        if not delivery_event.is_set():
            raise TimeoutError(f"Kafka delivery timed out for doc {message.doc_id}")
        if delivery_error:
            raise delivery_error[0]

    def flush(self, timeout: float = 10.0) -> None:
        """Flush any buffered messages. Call before shutdown."""
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning("Kafka flush: %d messages still in queue after timeout", remaining)

    def close(self) -> None:
        self.flush()


def ensure_topics_exist() -> None:
    """
    Idempotently create all required Kafka topics.
    Safe to call on every startup — skips existing topics.
    """
    admin = AdminClient({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})

    topics = [
        NewTopic(
            settings.KAFKA_TOPIC_RAW_DOCUMENTS,
            num_partitions=4,
            replication_factor=1,
            config={"retention.ms": str(7 * 24 * 3600 * 1000)},   # 7 days
        ),
        NewTopic(
            settings.KAFKA_TOPIC_DLQ,
            num_partitions=2,
            replication_factor=1,
            config={"retention.ms": str(30 * 24 * 3600 * 1000)},  # 30 days
        ),
        NewTopic(
            settings.KAFKA_TOPIC_EVENTS,
            num_partitions=2,
            replication_factor=1,
            config={"retention.ms": str(24 * 3600 * 1000)},        # 24 hours
        ),
    ]

    results = admin.create_topics(topics)
    for topic, future in results.items():
        try:
            future.result()
            logger.info("Kafka topic created: %s", topic)
        except Exception as e:
            if "TOPIC_ALREADY_EXISTS" in str(e):
                logger.debug("Kafka topic already exists: %s", topic)
            else:
                logger.error("Failed to create Kafka topic %s: %s", topic, e)
                raise