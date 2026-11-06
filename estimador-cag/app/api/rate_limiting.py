"""Rate limiting por API key para los endpoints de la Sesión 9.

El retrieval es barato pero puede martillearse; la estimación completa dispara
varias llamadas LLM y es la superficie más cara. Limitamos por ``X-API-Key``
(cuando existe) y caemos a la IP para no crear un bypass trivial.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def _rate_limit_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


_RETRY_AFTER_SECONDS = 60


def rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    """Devuelve un 429 JSON con ``Retry-After``, coherente con la API."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Demasiadas peticiones. Inténtalo de nuevo más tarde."},
        headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
    )


__all__ = ["RateLimitExceeded", "limiter", "rate_limit_handler"]
