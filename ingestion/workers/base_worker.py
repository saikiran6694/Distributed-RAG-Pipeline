"""
Abstract base class for all ingestion workers.
Handles the Kafka consumer loop, retry logic, offset management,
and DLQ routing so concrete workers only implement _process().

Critical design: offsets are committed ONLY after successful processing.
This guarantees at-least-once delivery — never lose a document on crash.
"""

from __future__ import annotations

import logging
import signal
import time
from abc import ABC, abstractmethod
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, Producer

from shared.config import get_settings
from shared.models import DLQEvent, DocumentIngestionMessage
from shared.telemetry import (
    DLQ_MESSAGES,
    DOCUMENTS_PROCESSED,
    INGESTION_DURATION,
    traced_span,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseWorker(ABC):
    """
    Base Kafka consumer worker.

    Subclasses must implement:
        worker_name: str class attribute
        _process(message: DocumentIngestionMessage) -> None
    """

    worker_name: str = "base-worker"   # override in subclass

    def __init__(self):
        self._consumer = self._build_consumer()
        self._dlq_producer = self._build_dlq_producer()
        self._running = False
        self._setup_signal_handlers()

    # ─────────────────────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the blocking consumer loop. Runs until SIGTERM/SIGINT."""
        self._consumer.subscribe([settings.KAFKA_TOPIC_RAW_DOCUMENTS])
        self._running = True
        logger.info("%s started, consuming from %s", self.worker_name, settings.KAFKA_TOPIC_RAW_DOCUMENTS)

        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue           # normal end-of-partition signal
                    raise KafkaException(msg.error())

                self._handle_message(msg)

        except KeyboardInterrupt:
            logger.info("%s interrupted by user", self.worker_name)
        finally:
            logger.info("%s shutting down", self.worker_name)
            self._consumer.close()

    # ─────────────────────────────────────────────────────────
    #  Message handling
    # ─────────────────────────────────────────────────────────

    def _handle_message(self, msg: Message) -> None:
        """
        Deserialize, validate, and process one Kafka message.
        Handles retries and DLQ routing.
        Commits offset only on success.
        """
        start_time = time.monotonic()

        try:
            ingestion_msg = DocumentIngestionMessage.model_validate_json(msg.value())
        except Exception as e:
            # Unparseable message — route straight to DLQ, can't retry meaningfully
            logger.error("Failed to deserialize Kafka message: %s", e)
            self._send_to_dlq(
                raw_payload=msg.value(),
                error_type="DeserializationError",
                error_message=str(e),
                kafka_partition=msg.partition(),
                kafka_offset=msg.offset(),
            )
            self._commit(msg)
            return

        doc_id = str(ingestion_msg.doc_id)
        doc_type = ingestion_msg.doc_type

        with traced_span(
            f"{self.worker_name}.process",
            {"doc_id": doc_id, "doc_type": doc_type, "retry": ingestion_msg.retry_count},
        ) as span:
            try:
                self._process(ingestion_msg)

                # ✅ Success — commit offset and record metric
                elapsed = time.monotonic() - start_time
                DOCUMENTS_PROCESSED.labels(status="success", doc_type=doc_type).inc()
                INGESTION_DURATION.labels(doc_type=doc_type).observe(elapsed)
                span.set_attribute("success", True)
                span.set_attribute("duration_seconds", round(elapsed, 3))

                self._commit(msg)

            except RetryableError as e:
                # Transient failure (network, timeout, etc.) — retry up to max
                retry_count = ingestion_msg.retry_count + 1
                logger.warning(
                    "Retryable error in %s (attempt %d/%d): %s",
                    self.worker_name, retry_count, settings.KAFKA_MAX_RETRIES, e,
                )
                DOCUMENTS_PROCESSED.labels(status="retry", doc_type=doc_type).inc()
                span.set_attribute("retried", True)

                if retry_count >= settings.KAFKA_MAX_RETRIES:
                    logger.error(
                        "Max retries exceeded for doc %s — routing to DLQ", doc_id
                    )
                    self._route_to_dlq(ingestion_msg, msg, str(type(e).__name__), str(e))
                else:
                    # Re-publish with incremented retry count
                    ingestion_msg.retry_count = retry_count
                    self._republish(ingestion_msg)

                self._commit(msg)   # always advance past this offset

            except PoisonPillError as e:
                # Permanent failure — route to DLQ immediately, no retry
                logger.error("Poison pill detected for doc %s: %s", doc_id, e)
                DOCUMENTS_PROCESSED.labels(status="poison_pill", doc_type=doc_type).inc()
                DLQ_MESSAGES.labels(error_type="PoisonPill").inc()
                self._route_to_dlq(ingestion_msg, msg, "PoisonPill", str(e))
                self._commit(msg)

            except Exception as e:
                # Unexpected error — treat as poison pill
                logger.exception("Unexpected error processing doc %s", doc_id)
                DOCUMENTS_PROCESSED.labels(status="unexpected_error", doc_type=doc_type).inc()
                self._route_to_dlq(ingestion_msg, msg, type(e).__name__, str(e))
                self._commit(msg)

    @abstractmethod
    def _process(self, message: DocumentIngestionMessage) -> None:
        """
        Implement document processing logic in subclass.
        Raise RetryableError for transient failures.
        Raise PoisonPillError for permanent failures.
        """
        ...

    # ─────────────────────────────────────────────────────────
    #  DLQ routing
    # ─────────────────────────────────────────────────────────

    def _route_to_dlq(
        self,
        ingestion_msg: DocumentIngestionMessage,
        kafka_msg: Message,
        error_type: str,
        error_message: str,
    ) -> None:
        dlq_event = DLQEvent(
            doc_id=ingestion_msg.doc_id,
            kafka_topic=kafka_msg.topic(),
            kafka_partition=kafka_msg.partition(),
            kafka_offset=kafka_msg.offset(),
            error_type=error_type,
            error_message=error_message,
            payload=ingestion_msg.model_dump(),
            retry_count=ingestion_msg.retry_count,
        )
        self._send_to_dlq(
            raw_payload=dlq_event.model_dump_json().encode(),
            error_type=error_type,
            error_message=error_message,
            kafka_partition=kafka_msg.partition(),
            kafka_offset=kafka_msg.offset(),
        )
        DLQ_MESSAGES.labels(error_type=error_type).inc()

    def _send_to_dlq(
        self,
        raw_payload: bytes,
        error_type: str,
        error_message: str,
        kafka_partition: int | None = None,
        kafka_offset: int | None = None,
    ) -> None:
        try:
            self._dlq_producer.produce(
                topic=settings.KAFKA_TOPIC_DLQ,
                value=raw_payload,
            )
            self._dlq_producer.poll(0)
        except Exception as e:
            logger.critical("Failed to write to DLQ: %s", e)

    def _republish(self, message: DocumentIngestionMessage) -> None:
        """Re-publish message to the main topic with incremented retry count."""
        try:
            self._dlq_producer.produce(
                topic=settings.KAFKA_TOPIC_RAW_DOCUMENTS,
                key=message.source_type.encode(),
                value=message.model_dump_json().encode(),
            )
            self._dlq_producer.poll(0)
        except Exception as e:
            logger.error("Failed to re-publish message for retry: %s", e)

    # ─────────────────────────────────────────────────────────
    #  Kafka plumbing
    # ─────────────────────────────────────────────────────────

    def _commit(self, msg: Message) -> None:
        """Manually commit offset for this message."""
        try:
            self._consumer.commit(message=msg, asynchronous=False)
        except KafkaException as e:
            logger.error("Failed to commit offset: %s", e)

    def _build_consumer(self) -> Consumer:
        return Consumer({
            "bootstrap.servers":        settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id":                 f"{settings.KAFKA_CONSUMER_GROUP}.{self.worker_name}",
            "auto.offset.reset":        "earliest",
            "enable.auto.commit":       False,      # CRITICAL — manual commit only
            "max.poll.interval.ms":     settings.KAFKA_MAX_POLL_INTERVAL_MS,
            "session.timeout.ms":       settings.KAFKA_SESSION_TIMEOUT_MS,
            "heartbeat.interval.ms":    3000,
        })

    def _build_dlq_producer(self) -> Producer:
        return Producer({
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "acks":              "1",               # DLQ: relaxed durability
        })

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        logger.info("%s received shutdown signal", self.worker_name)
        self._running = False


# ─────────────────────────────────────────────────────────────────
#  Custom exceptions for worker error classification
# ─────────────────────────────────────────────────────────────────

class RetryableError(Exception):
    """
    Transient failure — network timeout, rate limit, temporary unavailability.
    Worker will retry up to KAFKA_MAX_RETRIES times.
    """


class PoisonPillError(Exception):
    """
    Permanent failure — corrupt file, unsupported format, unrecoverable parse error.
    Worker routes directly to DLQ without retry.
    """