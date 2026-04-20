"""
Starts all ingestion workers as separate OS processes.

Usage:
    python -m ingestion.workers.worker_manager

    # Or via Makefile:
    make workers

Each worker type runs in its own process so a crash in the PDF worker
doesn't affect the HTML worker. Workers restart automatically on crash
(up to MAX_RESTARTS times before giving up).

Process layout:
    worker_manager (supervisor)
    ├── pdf-worker  (process 1)
    ├── html-worker (process 2)
    └── [future: docx-worker, markdown-worker]
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_RESTARTS   = 5
RESTART_DELAY  = 3   # seconds before restarting a crashed worker


@dataclass
class WorkerSpec:
    name:      str
    target_fn: str       # dotted import path to the start_*_worker function
    restarts:  int = 0
    process:   multiprocessing.Process | None = None


def _run_worker(target_fn: str) -> None:
    """Subprocess entry: import and call the worker start function."""
    module_path, fn_name = target_fn.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    fn  = getattr(mod, fn_name)
    fn()


class WorkerManager:
    """
    Supervisor that starts and monitors all ingestion worker processes.
    Restarts crashed workers up to MAX_RESTARTS times.
    """

    def __init__(self):
        self._specs: list[WorkerSpec] = [
            WorkerSpec("pdf-worker",  "ingestion.workers.pdf_worker.start_pdf_worker"),
            WorkerSpec("html-worker", "ingestion.workers.html_worker.start_html_worker"),
        ]
        self._running = True
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        )
        logger.info("WorkerManager starting %d workers", len(self._specs))

        # Start all workers
        for spec in self._specs:
            self._start(spec)

        # Monitor loop
        while self._running:
            time.sleep(2)
            for spec in self._specs:
                if not self._running:
                    break
                if spec.process and not spec.process.is_alive():
                    exit_code = spec.process.exitcode
                    if spec.restarts >= MAX_RESTARTS:
                        logger.error(
                            "%s exceeded max restarts (%d) — giving up",
                            spec.name, MAX_RESTARTS,
                        )
                        continue
                    logger.warning(
                        "%s died (exit=%d), restarting in %ds (attempt %d/%d)",
                        spec.name, exit_code, RESTART_DELAY,
                        spec.restarts + 1, MAX_RESTARTS,
                    )
                    time.sleep(RESTART_DELAY)
                    spec.restarts += 1
                    self._start(spec)

        self._stop_all()

    def _start(self, spec: WorkerSpec) -> None:
        p = multiprocessing.Process(
            target=_run_worker,
            args=(spec.target_fn,),
            name=spec.name,
            daemon=True,
        )
        p.start()
        spec.process = p
        logger.info("Started %s (pid=%d)", spec.name, p.pid)

    def _stop_all(self) -> None:
        logger.info("Stopping all workers...")
        for spec in self._specs:
            if spec.process and spec.process.is_alive():
                spec.process.terminate()
                spec.process.join(timeout=5)
                if spec.process.is_alive():
                    spec.process.kill()
                logger.info("Stopped %s", spec.name)

    def _shutdown(self, signum, frame) -> None:
        logger.info("WorkerManager received shutdown signal")
        self._running = False


if __name__ == "__main__":
    WorkerManager().run()