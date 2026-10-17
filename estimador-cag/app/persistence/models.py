"""Modelos ORM de SQLAlchemy para la Sesión 6.

Dos tablas, ambas estrechas y de escritura poco frecuente:

* ``pseudonym_mappings`` — la correspondencia reversible con nivel RGPD para
  PII. Tiene como clave ``(entity_type, original_hash)``; el hash es HMAC-SHA256
  del valor original con una sal del servidor. Almacenar el hash (no el texto
  en claro) implica que una lectura de la base de datos por sí sola no puede
  reconstruir el original — esa es la propiedad que hace auditable el "derecho
  al olvido" del art. 17.

* ``ingestion_jobs`` — contabilidad del endpoint asíncrono ``POST
  /ingestion/runs``. Se crea una fila cuando llega la petición y una
  BackgroundTask la actualiza a ``running``/``completed``/``failed``. El
  endpoint ``GET /ingestion/jobs/{id}`` lee de aquí.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Index, Integer, String, UniqueConstraint, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import DateTime


class Base(DeclarativeBase):
    """Base declarativa única — la recoge el env.py de Alembic."""


class PseudonymMappingRow(Base):
    __tablename__ = "pseudonym_mappings"
    __table_args__ = (
        UniqueConstraint("entity_type", "original_hash", name="uq_mappings_entity_hash"),
        Index("idx_mappings_lookup", "entity_type", "original_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    original_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    pseudonym: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class IngestionJobRow(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("idx_jobs_status", "status"),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    documents_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
