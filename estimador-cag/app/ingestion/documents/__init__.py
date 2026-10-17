"""El contrato canónico ``Document``.

Todos los parsers producen instancias de ``Document``. Todos los consumidores
aguas abajo (cleaning, validación, PII, futuro RAG) aceptan instancias de
``Document``. Mantener este modelo deliberadamente *plano* (sin tipos de negocio
anidados) hace que sea trivial de simular en tests y homogéneo entre formatos.
"""

from app.ingestion.documents.models import Document, DocumentMetadata

__all__ = ["Document", "DocumentMetadata"]
