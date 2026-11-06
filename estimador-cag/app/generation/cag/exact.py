"""CAG exact-match: política de caché exacta del estimador.

Los backends genéricos (memoria, Redis y null) viven en `foundation/cache` para
que `LLMWrapper` pueda usarlos sin depender de la capa CAG. Este módulo es el
punto de entrada de CAG para la caché exacta: la clave sigue siendo el SHA-256
del material de dominio + los knobs de generación.
"""

from __future__ import annotations

from app.foundation.cache.backends import (
    InMemoryCache,
    NullCache,
    RedisCache,
    ResponseCache,
    make_cache_key,
)

__all__ = [
    "InMemoryCache",
    "NullCache",
    "RedisCache",
    "ResponseCache",
    "make_cache_key",
]
