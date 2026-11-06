"""Autenticación por API key de los endpoints de la Sesión 9.

Dos secretos independientes para separar superficies: una clave para retrieval
(barato, alta frecuencia) y otra para estimación (caro, baja frecuencia). Si la
clave esperada no está configurada, el endpoint responde 401: no hay modo
"abierto" accidental en producción.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status
from pydantic import SecretStr

from app.config import reveal_secret, settings


def _check_key(provided: str | None, configured: SecretStr | None, surface: str) -> None:
    expected = reveal_secret(configured)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"La API key de {surface} no está configurada.",
        )
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"API key de {surface} inválida o ausente.",
        )


async def require_retrieval_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Exige una API key válida para los endpoints de retrieval."""
    _check_key(x_api_key, settings.retrieval_api_key, "retrieval")


async def require_estimate_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Exige una API key válida para los endpoints de estimación."""
    _check_key(x_api_key, settings.estimate_api_key, "estimate")
