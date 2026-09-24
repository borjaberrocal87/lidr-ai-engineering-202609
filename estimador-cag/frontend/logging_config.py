"""Logging mínimo del frontend, independiente de structlog/`app.config`."""

import logging
import sys

from frontend.config import get_log_level

_configured = False


def configure_logging() -> None:
    """Configura el logging estándar una sola vez por proceso.

    Idempotente: Streamlit reejecuta el script en cada rerun.
    """
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=get_log_level(),
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _configured = True
