"""Embedder de OpenAI para el pipeline de embeddings.

Envuelve ``text-embedding-3-small`` (1536 dims por defecto). ``embed_many``
agrupa los chunks con ``itertools.batched`` en una llamada por batch (el
endpoint de embeddings acepta listas), mucho más barato en latencia que una
petición por chunk.
"""

from __future__ import annotations

import time
from itertools import batched
from typing import Any

import structlog
from openai import OpenAI, RateLimitError

from app.config import Settings, reveal_secret
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

# Modelo por defecto del ejercicio. Mantenemos las 1536 dims completas (sin
# recorte Matryoshka: eso es discusión del directo).
MODEL = "text-embedding-3-small"

# Chunks por llamada a la API. El endpoint acepta listas; batchear evita
# martillear la API con peticiones serializadas de un solo item.
BATCH_SIZE = 100

# Constante de precio — CAMBIA CON EL TIEMPO. A fecha de este ejercicio,
# text-embedding-3-small cuesta $0.02 por 1M de tokens de entrada.
PRICE_PER_MILLION_TOKENS_USD = 0.02

# Esperas (segundos) antes de cada reintento por rate limit: 3 reintentos.
_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


def estimate_cost_usd(total_tokens: int) -> float:
    """Coste en USD de embeber ``total_tokens`` tokens de entrada."""
    return total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


class OpenAIEmbedder:
    """Wrapper fino sobre ``client.embeddings.create`` con batching y reintentos."""

    def __init__(self, client: OpenAI, model: str = MODEL, dimensions: int | None = None) -> None:
        self._client = client
        self._model = model
        # Recorte Matryoshka opcional. ``None`` = dimensión nativa del modelo.
        self._dimensions = dimensions

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIEmbedder | None:
        """Construye el embedder desde la configuración del servicio.

        Devuelve ``None`` si no hay embeddings configurados (ni ``OPEN_AI_KEY``
        ni ``EMBEDDING_BASE_URL`` + ``EMBEDDING_API_KEY``). Única fábrica del
        cliente: la usan tanto las dependencias de FastAPI como el CLI.
        """
        if not settings.has_embeddings:
            return None
        api_key = reveal_secret(settings.embedding_api_key) or reveal_secret(settings.open_ai_key)
        client = OpenAI(api_key=api_key, base_url=settings.embedding_base_url or None)
        return cls(client=client, model=settings.embedding_model)

    def embed_one(self, text: str) -> list[float]:
        """Embebe un único texto. Lo usa el script CLI `compare.py`."""
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embebe todos los chunks manteniendo el orden, en batches."""
        embedded: list[EmbeddedChunk] = []
        for batch in batched(chunks, BATCH_SIZE):
            texts = [chunk.text for chunk in batch]
            batch_tokens = sum(chunk.token_count for chunk in batch)

            started_at = time.perf_counter()
            vectors = self._create(texts)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

            log.info(
                "embedding.batch.end",
                chunks=len(batch),
                tokens=batch_tokens,
                latency_ms=latency_ms,
                model=self._model,
            )

            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    def _create(self, texts: list[str]) -> list[list[float]]:
        """Llama a la API con reintentos ante rate limit.

        Se hacen 3 reintentos con esperas de 1s, 2s y 4s. El resto de errores
        (red, autenticación, respuestas no recuperables) se propagan.
        """
        retries = 0
        while True:
            try:
                kwargs: dict[str, Any] = {"model": self._model, "input": texts}
                if self._dimensions is not None:
                    kwargs["dimensions"] = self._dimensions
                response = self._client.embeddings.create(**kwargs)
                return [item.embedding for item in response.data]
            except RateLimitError:
                if retries >= len(_RETRY_DELAYS_SECONDS):
                    raise
                delay = _RETRY_DELAYS_SECONDS[retries]
                retries += 1
                log.warning("embedding.batch.rate_limited", retry=retries, retry_in_s=delay)
                time.sleep(delay)
