"""Caché semántica del estimador (sesión 4).

Dos peticiones se consideran la misma cuando:

1. Su **bucket** coincide exactamente. El bucket es una etiqueta determinista
   ``prompt_version:project_type:detail_level:output_format``. Dos peticiones con
   opciones de formulario distintas nunca comparten entrada aunque sus
   descripciones sean parecidas: el prompt renderizado es distinto.

2. La similitud coseno de los embeddings de sus descripciones es al menos
   ``threshold`` (por defecto 0.85).

Con ``log_only=True`` se hace la búsqueda y se registra la puntuación, pero nunca
se devuelve un acierto: útil para calibrar el umbral con tráfico real.

Requiere ``redis/redis-stack``: la imagen vanila ``redis:7-alpine`` no trae
RediSearch y ``SearchIndex.create()`` fallaría al arrancar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from app.schemas.estimations import EstimationRequest, EstimationResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _to_bytes(vector: list[float]) -> bytes:
    """RediSearch guarda los vectores como bytes float32."""
    return np.array(vector, dtype=np.float32).tobytes()


_INDEX_SCHEMA: dict[str, Any] = {
    "index": {
        "name": "estimations",
        "prefix": "estimation:semantic",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "bucket", "type": "tag"},
        {"name": "result_json", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 1536,  # text-embedding-3-small
                "distance_metric": "cosine",
                "algorithm": "flat",
            },
        },
    ],
}


class EstimationSemanticCache:
    """Caché por similitud vectorial sobre redisvl + Redis Stack."""

    def __init__(
        self,
        *,
        redis_client: Any,
        vectorizer: Any,
        threshold: float = 0.85,
        ttl: int = 86_400,
        log_only: bool = False,
        index_name: str = "estimations",
    ) -> None:
        from redisvl.index import SearchIndex

        self.redis_client = redis_client
        self.vectorizer = vectorizer
        self.threshold = threshold
        self.ttl = ttl
        self.log_only = log_only

        schema = {**_INDEX_SCHEMA, "index": {**_INDEX_SCHEMA["index"], "name": index_name}}
        self.index = SearchIndex.from_dict(schema)
        self.index.set_client(redis_client)
        try:
            self.index.create(overwrite=False)
        except Exception as exc:
            logger.debug("semantic_index_create_skipped", error=str(exc)[:120])

    @staticmethod
    def bucket_for(request: EstimationRequest, prompt_version: str) -> str:
        return (
            f"{prompt_version}"
            f":{request.project_type.value}"
            f":{request.detail_level.value}"
            f":{request.output_format.value}"
        )

    def lookup(self, request: EstimationRequest, prompt_version: str) -> EstimationResult | None:
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        bucket = self.bucket_for(request, prompt_version)
        embedding = self.vectorizer.embed(request.description)
        query = VectorQuery(
            vector=_to_bytes(embedding),
            vector_field_name="embedding",
            return_fields=["result_json", "bucket"],
            num_results=1,
            return_score=True,
            filter_expression=Tag("bucket") == bucket,
        )
        results = self.index.query(query)
        if not results:
            logger.info("semantic_cache_miss", bucket=bucket, reason="empty_index")
            return None

        hit = results[0]
        similarity = _similarity(hit)
        logger.info(
            "semantic_cache_lookup",
            bucket=bucket,
            similarity=round(similarity, 4),
            threshold=self.threshold,
        )

        if similarity < self.threshold:
            logger.info("semantic_cache_miss", bucket=bucket, reason="below_threshold")
            return None

        if self.log_only:
            logger.info(
                "semantic_cache_hit_log_only",
                bucket=bucket,
                similarity=round(similarity, 4),
            )
            return None

        logger.info("semantic_cache_hit", bucket=bucket, similarity=round(similarity, 4))
        return EstimationResult.model_validate_json(hit["result_json"])

    def store(
        self,
        request: EstimationRequest,
        result: EstimationResult,
        prompt_version: str,
    ) -> None:
        bucket = self.bucket_for(request, prompt_version)
        embedding = self.vectorizer.embed(request.description)
        payload = [
            {
                "bucket": bucket,
                "result_json": result.model_dump_json(),
                "embedding": _to_bytes(embedding),
            }
        ]
        try:
            self.index.load(payload, ttl=self.ttl)
            logger.info("semantic_cache_stored", bucket=bucket, ttl=self.ttl)
        except Exception as exc:
            logger.warning(
                "semantic_cache_store_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )


def _similarity(hit: dict[str, Any]) -> float:
    """Similitud coseno a partir del resultado de redisvl.

    `redisvl` devuelve distancia coseno (0 = idéntico). Algunas versiones
    normalizan y exponen directamente `similarity`.
    """
    if "similarity" in hit:
        return float(hit["similarity"])
    distance = float(hit.get("vector_distance", 1.0))
    return 1.0 - distance
