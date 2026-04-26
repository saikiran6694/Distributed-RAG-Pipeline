# ─────────────────────────────────────────────────────────────────
#  Distributed RAG Pipeline — Makefile
# ─────────────────────────────────────────────────────────────────

.PHONY: up down reset logs setup-collection setup-kafka setup \
         api api-prod \
         workers worker-pdf worker-html \
         ingest-file ingest-dir ingest-url \
         unit integration e2e test-all smoke \
         lint typecheck \
         frontend-install frontend frontend-build

# ── Infrastructure ────────────────────────────────────────────────

up:
	cd infrastructure && docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@cd infrastructure && docker compose ps

down:
	cd infrastructure && docker compose down

reset:
	cd infrastructure && docker compose down -v
	@echo "Volumes wiped. Run 'make up' to start fresh."

logs:
	cd infrastructure && docker compose logs -f --tail=50

# ── Collection setup (run once after 'make up') ───────────────────

setup-collection:
	python -m infrastructure.qdrant.collection_setup

setup-kafka:
	docker exec -it rag-kafka kafka-topics \
		--bootstrap-server localhost:9092 \
		--create --if-not-exists --topic raw-documents \
		--partitions 4 --replication-factor 1
	@echo "Kafka topic raw-documents ready"

setup: setup-collection setup-kafka
	@echo "All setup complete"

# ── API ───────────────────────────────────────────────────────────

api:
	uvicorn api.main:app --reload --port 8000 --log-level info

api-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Tests ─────────────────────────────────────────────────────────

unit:
	pytest tests/unit/ -v --tb=short

integration:
	pytest tests/integration/ -v --tb=short --timeout=120

e2e:
	python scripts/test_e2e.py

test-all: unit integration e2e

# ── Code quality ──────────────────────────────────────────────────

lint:
	ruff check ingestion/ query/ api/ shared/ tests/

typecheck:
	mypy ingestion/ query/ api/ shared/ --ignore-missing-imports

# ── Workers ───────────────────────────────────────────────────────

workers:
	python -m ingestion.workers.worker_manager

worker-pdf:
	python -m ingestion.workers.pdf_worker

worker-html:
	python -m ingestion.workers.html_worker

# ── Ingestion ─────────────────────────────────────────────────────

ingest-file:
	@test -n "$(FILE)" || (echo "Usage: make ingest-file FILE=path/to/doc.pdf" && exit 1)
	python scripts/ingest.py --file $(FILE)

ingest-dir:
	@test -n "$(DIR)" || (echo "Usage: make ingest-dir DIR=./docs/" && exit 1)
	python scripts/ingest.py --dir $(DIR)

ingest-url:
	@test -n "$(URL)" || (echo "Usage: make ingest-url URL=https://..." && exit 1)
	python scripts/ingest.py --url $(URL)

# ── Quick smoke test (no Docker required) ────────────────────────

smoke:
	@echo "Running unit tests only (no Docker needed)..."
	pytest tests/unit/ -v --tb=short -q