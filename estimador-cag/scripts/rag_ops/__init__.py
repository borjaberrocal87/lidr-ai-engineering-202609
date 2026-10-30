"""RAG Ops: utilidades de operación del store vectorial (Sesión 08).

El paquete agrupa los benchmarks y la auditoría de índices de pgvector. Cada
script se ejecuta como módulo para compartir importaciones limpias:

    docker compose exec api python -m scripts.rag_ops.baseline_latency
"""

from __future__ import annotations

__all__ = ["common"]
