"""Pydantic Logfire wiring for the estimation graph (Session 13, Level 2).

``configure_logfire`` is called once at startup:

* the per-node ``span("node: …")`` calls in ``nodes.py`` become one span per node
  inside the request trace;
* ``instrument_fastapi`` gives the HTTP request its root span;
* ``instrument_httpx`` captures the outbound LLM calls (the provider SDKs use
  httpx under the hood).

It is a **no-op without a token**: ``send_to_logfire="if-token-present"`` means a
run with no ``LOGFIRE_TOKEN`` still executes every span locally but exports
nothing, so the service never depends on Logfire being configured.

If you prefer LangSmith, this is the seam to swap: keep the ``span`` calls (they
are cheap) and configure the LangGraph tracer here instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import structlog

from app.config import get_settings, reveal_secret

log = structlog.get_logger(__name__)

_configured = False


def configure_logfire(app: Any | None = None) -> bool:
    """Configure Logfire once. Returns ``True`` if configuration succeeded.

    A missing package / token / network never breaks startup: failures are logged
    and swallowed. ``app`` is the FastAPI instance to instrument; pass ``None`` to
    skip FastAPI instrumentation (e.g. from a CLI run).
    """
    global _configured
    if _configured:
        return True
    try:
        import logfire

        settings = get_settings()
        kwargs: dict[str, Any] = {
            "service_name": settings.logfire_service_name,
            # Export only when a token is configured; otherwise spans run locally.
            "send_to_logfire": "if-token-present",
            "console": False,
        }
        token = reveal_secret(settings.logfire_token)
        if token:
            kwargs["token"] = token
        logfire.configure(**kwargs)
    except Exception as exc:
        log.warning("logfire_configure_failed", error=str(exc)[:200])
        return False

    # Mark configured before the optional auto-instrumentations: a missing extra
    # (httpx/fastapi) must not make every later span try to reconfigure.
    _configured = True
    if app is not None:
        with suppress(Exception):
            logfire.instrument_fastapi(app)
    with suppress(Exception):
        logfire.instrument_httpx()
    log.info("logfire_configured", service_name=get_settings().logfire_service_name)
    return True


@contextmanager
def span(name: str) -> Iterator[None]:
    """Open a Logfire span, degrading to a no-op if Logfire is unavailable.

    Errors raised by the wrapped block propagate untouched: only failures while
    *opening* the span are swallowed, so observability can never hide a node
    failure.
    """
    if not _configured:
        configure_logfire()
    opened: Any = None
    try:
        import logfire

        opened = logfire.span(name)
        opened.__enter__()
    except Exception:
        opened = None
    try:
        yield
    finally:
        if opened is not None:
            with suppress(Exception):
                opened.__exit__(None, None, None)
