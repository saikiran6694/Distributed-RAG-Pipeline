import logging
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from shared.config import get_settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
#  Prometheus metrics
#  All metrics are module-level singletons — safe to import anywhere
# ─────────────────────────────────────────────────────────────────

DOCUMENTS_PROCESSED = Counter(
    "ingestion_documents_total",
    "Total documents processed",
    ["status", "doc_type"],
)

CHUNKS_PRODUCED = Counter(
    "ingestion_chunks_total",
    "Total chunks produced",
    ["strategy"],
)

INGESTION_DURATION = Histogram(
    "ingestion_duration_seconds",
    "End-to-end ingestion latency per document",
    ["doc_type"],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600],
)

EMBEDDING_BATCH_DURATION = Histogram(
    "embedding_batch_duration_seconds",
    "Time per embedding batch API call",
    ["backend"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

KAFKA_CONSUMER_LAG = Gauge(
    "kafka_consumer_lag",
    "Current consumer lag per topic partition",
    ["topic", "partition"],
)

DLQ_MESSAGES = Counter(
    "dlq_messages_total",
    "Messages routed to dead letter queue",
    ["error_type"],
)

EMBEDDING_COST_USD = Counter(
    "ingestion_cost_usd_total",
    "Cumulative embedding API spend in USD",
    ["model"],
)

PARSE_ERRORS = Counter(
    "parse_errors_total",
    "Non-fatal parsing errors",
    ["doc_type", "error_type"],
)


# ─────────────────────────────────────────────────────────────────
#  OpenTelemetry tracer
# ─────────────────────────────────────────────────────────────────

_tracer: trace.Tracer | None = None


def configure_telemetry(service_name: str | None = None) -> None:
    """
    Call once at startup before any tracing.
    Configures OTLP exporter → Jaeger and Prometheus metrics server.
    """
    global _tracer
    settings = get_settings()
    name = service_name or settings.SERVICE_NAME

    if settings.ENABLE_TRACING:
        resource = Resource.create({"service.name": name, "environment": settings.ENVIRONMENT})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing configured → %s", settings.OTEL_EXPORTER_OTLP_ENDPOINT)

    _tracer = trace.get_tracer(name)

    if settings.ENABLE_METRICS:
        try:
            start_http_server(settings.PROMETHEUS_PORT)
            logger.info("Prometheus metrics server started on :%d", settings.PROMETHEUS_PORT)
        except OSError:
            # Port already in use (e.g. in tests) — ignore
            pass


def get_tracer() -> trace.Tracer:
    """Return the configured tracer. configure_telemetry() must be called first."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(get_settings().SERVICE_NAME)
    return _tracer


@contextmanager
def traced_span(
    name: str,
    attributes: dict | None = None,
) -> Generator[trace.Span, None, None]:
    """
    Context manager for creating a named span with optional attributes.

    Usage:
        with traced_span("parse_pdf", {"doc_id": str(doc_id)}) as span:
            ...
            span.set_attribute("page_count", 42)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        yield span


def configure_logging() -> None:
    """Structured JSON logging setup. Call at service startup."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='{"time": "%(asctime)s", "level": "%(levelname)s", '
               '"service": "%(name)s", "message": "%(message)s"}',
    )