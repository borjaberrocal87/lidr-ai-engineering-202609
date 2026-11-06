"""Logging estructurado por etapa del pipeline RAG (Sesión 9).

``log_pipeline_stage`` envuelve cada paso para que emita el trío consistente
``stage.started`` / ``stage.completed`` (con ``duration_ms``) / ``stage.failed``,
todo correlacionado por el mismo ``request_id``. Así el viaje de una petición
por brief → retrieval → augmentación → generación es trivial de seguir en los
logs JSON.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import structlog

log = structlog.get_logger()


def current_request_id() -> str:
    """Reutiliza el ``request_id`` del middleware HTTP o genera uno nuevo."""
    bound = structlog.contextvars.get_contextvars().get("request_id")
    return str(bound) if bound else uuid4().hex


@contextmanager
def log_pipeline_stage(stage: str, request_id: str, **context: Any) -> Iterator[None]:
    """Registra el ciclo de vida de una etapa ligada a ``request_id``.

    ``**context`` se adjunta a los tres eventos. Cualquier excepción se registra
    como ``stage.failed`` (con duración y tipo de error) y se re-lanza.
    """
    log.info("stage.started", stage=stage, request_id=request_id, **context)
    started_at = time.perf_counter()
    try:
        yield
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        log.error(
            "stage.failed",
            stage=stage,
            request_id=request_id,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            **context,
        )
        raise
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(
        "stage.completed",
        stage=stage,
        request_id=request_id,
        duration_ms=duration_ms,
        **context,
    )
