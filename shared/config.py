"""
All configuration read from environment variables.
Pydantic Settings validates types and provides defaults.
"""

from functools import lru_cache
from typing import Literal
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_PATH = Path(__file__).parent.parent / ".env"

if ENV_PATH.exists():
    print(f"Loading environment variables from {ENV_PATH}")
else:    
    print(f"Warning: {ENV_PATH} not found. Make sure to create a .env file with the required environment variables.")
    

class Settings(BaseSettings):
    # ── Service identity ──────────────────────────────────────
    SERVICE_NAME: str = "rag-ingestion"
    ENVIRONMENT:  Literal["local", "staging", "production"] = "local"
    LOG_LEVEL:    Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Kafka ─────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS:    str = "localhost:9092"
    KAFKA_TOPIC_RAW_DOCUMENTS:  str = "raw-documents"
    KAFKA_TOPIC_DLQ:            str = "ingestion-dlq"
    KAFKA_TOPIC_EVENTS:         str = "ingestion-events"
    KAFKA_CONSUMER_GROUP:       str = "rag-ingestion-workers"
    KAFKA_MAX_POLL_INTERVAL_MS: int = 300_000   # 5 min — parsing can be slow
    KAFKA_SESSION_TIMEOUT_MS:   int = 45_000
    KAFKA_MAX_RETRIES:          int = 3         # before routing to DLQ

    # ── PostgreSQL ────────────────────────────────────────────
    POSTGRES_HOST:     str = "localhost"
    POSTGRES_PORT:     int = 5432
    POSTGRES_DB:       str = "rag_registry"
    POSTGRES_USER:     str = "rag"
    POSTGRES_PASSWORD: str = "rag_secret"
    POSTGRES_POOL_MIN: int = 2
    POSTGRES_POOL_MAX: int = 10

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Qdrant ────────────────────────────────────────────────
    QDRANT_HOST:            str = "localhost"
    QDRANT_PORT:            int = 6333
    QDRANT_GRPC_PORT:       int = 6334
    QDRANT_COLLECTION_NAME: str = "rag_chunks"
    QDRANT_USE_GRPC:        bool = True     # gRPC is faster for bulk upserts

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    # ── Redis ─────────────────────────────────────────────────
    REDIS_HOST:     str = "localhost"
    REDIS_PORT:     int = 6379
    REDIS_DB:       int = 0

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── Ingestion limits ──────────────────────────────────────
    MAX_DOCUMENT_SIZE_MB:   int   = 200         # reject files larger than this
    MAX_OCR_RETRY_PAGES:    int   = 3
    OCR_CONFIDENCE_THRESHOLD: float = 0.70      # skip OCR chunks below this score

    @field_validator("OCR_CONFIDENCE_THRESHOLD")
    @classmethod
    def validate_ocr_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("OCR_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        return v

    # ── Chunking ──────────────────────────────────────────────
    CHUNK_FIXED_SIZE_TOKENS:     int   = 512
    CHUNK_FIXED_OVERLAP_TOKENS:  int   = 50
    CHUNK_SEMANTIC_MIN_TOKENS:   int   = 100
    CHUNK_SEMANTIC_MAX_TOKENS:   int   = 800
    CHUNK_SEMANTIC_THRESHOLD:    float = 0.75   # cosine similarity split threshold
    CHUNK_HIER_PARENT_TOKENS:    int   = 1500
    CHUNK_HIER_CHILD_TOKENS:     int   = 300

    # ── Embedding ─────────────────────────────────────────────
    # Backend: 'openai' | 'huggingface' | 'ollama'
    EMBED_BACKEND:          Literal["openai", "huggingface", "ollama"] = "huggingface"
    EMBED_BATCH_SIZE:       int = 32
    EMBED_MAX_RETRIES:      int = 5
    EMBED_RETRY_BASE_DELAY: float = 2.0         # seconds (exponential backoff base)
    EMBED_RETRY_MAX_DELAY:  float = 60.0

    # OpenAI
    OPENAI_API_KEY:         str | None = None
    OPENAI_EMBED_MODEL:     str = "text-embedding-3-small"
    OPENAI_EMBED_MODEL_VERSION: str = "1.0.0"
    OPENAI_EMBED_DIM:       int = 1536

    # HuggingFace local
    HF_EMBED_MODEL:         str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_EMBED_MODEL_VERSION: str = "1.0.0"
    HF_EMBED_DIM:           int = 384
    HF_DEVICE:              str = "cpu"         # 'cpu' | 'cuda' | 'mps'

    GROQ_API_KEY:           str | None = None
    GROQ_LLAMA_MODEL_NAME:  str = "llama-3.3-70b-versatile"

    # Ollama
    OLLAMA_BASE_URL:        str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL:     str = "nomic-embed-text"
    OLLAMA_EMBED_DIM:       int = 768
    OLLAMA_MODEL_NAME:      str = "llama3.2:1b"
    

    # ── Observability ─────────────────────────────────────────
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    PROMETHEUS_PORT:             int = 9090
    ENABLE_TRACING:              bool = True
    ENABLE_METRICS:              bool = True

    # ── Reconciliation job ────────────────────────────────────
    RECONCILIATION_INTERVAL_SECONDS: int = 300       # run every 5 min
    RECONCILIATION_STALE_MINUTES:    int = 10        # re-queue pending_vector older than this

    model_config = SettingsConfigDict(
        env_file=ENV_PATH if ENV_PATH.exists else None,
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance. Call this everywhere instead of Settings()."""
    return Settings()
