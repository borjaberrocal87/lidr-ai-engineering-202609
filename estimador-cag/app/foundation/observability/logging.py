"""Configuración de logging basada en structlog.

Integra structlog con la librería estándar (`logging`) para que las llamadas al
LLM, FastAPI, uvicorn y Streamlit compartan el mismo formato y el mismo nivel
(`LOG_LEVEL`). Es idempotente: puede invocarse varias veces (por ejemplo en los
reruns de Streamlit) sin duplicar handlers.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.types import Processor

from app.config import settings

_configured = False

_NOISY_LOGGERS = (
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "openai",
    "anthropic",
    "urllib3",
    "asyncio",
)


def _renderer() -> Processor:
    """Devuelve el renderer según `LOG_FORMAT` (texto legible o JSON)."""
    if settings.log_format == "json":
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())


def configure_logging() -> None:
    """Configura structlog + logging estándar una sola vez por proceso."""
    global _configured
    if _configured:
        return

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _renderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
