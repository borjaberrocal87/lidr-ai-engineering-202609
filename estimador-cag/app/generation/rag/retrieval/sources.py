"""Catálogo de fuentes recuperables (Sesión 10).

La recuperación multifuente necesita una única pieza que sepa, por cada fuente:
qué modelo ORM la respalda, cómo leer su fecha, cómo traducir una fila al
contrato uniforme :class:`RetrievedChunk` y qué vocabulario la identifica. Ese
conocimiento vive en ``data/catalog/collections.yaml``, no en el código: aquí
solo lo cargamos, lo validamos y lo exponemos.

La ventaja de tenerlo en datos es doble: añadir una fuente no obliga a tocar el
pipeline, y las reglas de enrutado se revisan como se revisa un catálogo (con
diff, con validación y con tests), no como un diccionario escondido en Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Date, Integer, cast

from app.config import BASE_DIR
from app.generation.rag.schemas import RetrievedChunk
from app.generation.rag.store.models import (
    BudgetChunkRow,
    TechnicalDocChunkRow,
    TranscriptChunkRow,
)
from app.generation.rag.store.repository import ChunkMatch

COLLECTIONS_CATALOG_PATH = BASE_DIR / "data" / "catalog" / "collections.yaml"

# Ejes de filtro que una fuente puede declarar en el catálogo. El pipeline solo
# emite cláusulas para los ejes que la fuente entiende; el resto se ignora.
FilterAxis = Literal["sectors", "technologies", "min_date", "version"]


class Collection(StrEnum):
    """Conjunto cerrado de fuentes recuperables.

    Es el enum que puede emitir el clasificador LLM y la clave del catálogo: un
    valor fuera de esta lista no existe para el pipeline.
    """

    BUDGET = "budget"
    TRANSCRIPT = "transcript"
    TECHNICAL_DOC = "technical_doc"


_MODEL_BY_TABLE: dict[str, type] = {
    "budget_chunks": BudgetChunkRow,
    "transcript_chunks": TranscriptChunkRow,
    "technical_doc_chunks": TechnicalDocChunkRow,
}


class DateSpec(BaseModel):
    """Cómo se lee la fecha de una fuente desde su metadata JSONB."""

    model_config = ConfigDict(extra="forbid")

    key: str
    kind: Literal["year", "iso"]


class CollectionDefinition(BaseModel):
    """Una fuente tal y como se declara en el YAML."""

    model_config = ConfigDict(extra="forbid")

    name: Collection
    label: str = Field(min_length=1)
    description: str = ""
    table: str
    chunk_type: str = Field(min_length=1)
    source_id_key: str = Field(min_length=1)
    date: DateSpec | None = None
    filters: list[FilterAxis] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)

    @field_validator("table")
    @classmethod
    def _table_is_known(cls, value: str) -> str:
        if value not in _MODEL_BY_TABLE:
            raise ValueError(f"Tabla desconocida en el catálogo: {value}")
        return value

    @field_validator("rules")
    @classmethod
    def _rules_compile(cls, values: list[str]) -> list[str]:
        for pattern in values:
            try:
                re.compile(pattern)
            except re.error as exc:  # pragma: no cover - el mensaje guía el fix
                raise ValueError(f"Patrón inválido en el catálogo: {pattern!r} ({exc})") from exc
        return values


class CollectionsCatalog(BaseModel):
    """El catálogo completo, versionado en git junto al código."""

    model_config = ConfigDict(extra="forbid")

    version: str
    description: str = ""
    collections: list[CollectionDefinition] = Field(min_length=1)

    @field_validator("collections")
    @classmethod
    def _names_are_unique(cls, values: list[CollectionDefinition]) -> list[CollectionDefinition]:
        seen: set[Collection] = set()
        for definition in values:
            if definition.name in seen:
                raise ValueError(f"Fuente duplicada en el catálogo: {definition.name.value}")
            seen.add(definition.name)
        return values


@dataclass(frozen=True)
class HardFilters:
    """Restricciones de metadata aplicadas ANTES del ranking vectorial.

    Cada fuente consume el subconjunto que entiende y descarta el resto. Un
    campo vacío o ``None`` significa "no restringir este eje".
    """

    technologies: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()
    min_date: date | None = None
    version: str | None = None


@dataclass(frozen=True)
class CollectionSpec:
    """Vista operativa de una fuente: modelo, fecha, filtros y reglas."""

    collection: Collection
    label: str
    description: str
    model: type
    chunk_type: str
    source_id_key: str
    date_spec: DateSpec | None = None
    filters: tuple[FilterAxis, ...] = field(default_factory=tuple)
    rule_patterns: tuple[str, ...] = field(default_factory=tuple)

    def date_of(self, metadata: dict) -> date | None:
        """Lee una fecha de esta fuente desde su metadata, o ``None``."""
        if self.date_spec is None:
            return None
        raw = (metadata or {}).get(self.date_spec.key)
        if raw in (None, ""):
            return None
        try:
            if self.date_spec.kind == "year":
                return date(int(raw), 1, 1)
            if self.date_spec.kind == "iso":
                return date.fromisoformat(str(raw))
        except (TypeError, ValueError):
            return None
        return None

    def to_chunk(
        self,
        match: ChunkMatch,
        *,
        distance: float,
        relevance_score: float | None = None,
    ) -> RetrievedChunk:
        """Traduce una coincidencia del store al contrato uniforme de retrieval."""
        metadata = match.metadata or {}
        raw_year = metadata.get("year")
        try:
            project_year = int(raw_year) if raw_year is not None else None
        except (TypeError, ValueError):
            project_year = None
        sector = metadata.get("client_sector")
        return RetrievedChunk(
            id=match.chunk_id,
            document_id=match.document_id,
            content=match.content,
            sector=str(sector) if sector is not None else None,
            project_year=project_year,
            chunk_type=match.chunk_type,
            distance=distance,
            budget_id=metadata.get("budget_id"),
            collection=self.collection.value,
            source_id=metadata.get(self.source_id_key),
            relevance_score=relevance_score,
            document_date=self.date_of(metadata),
            estimated_hours=_as_int(metadata.get("estimated_hours")),
        )

    def hard_filter_clauses(self, filters: HardFilters) -> list:
        """Cláusulas SQLAlchemy que embeben los filtros duros de la petición.

        Solo se emiten los ejes declarados por la fuente. Una lista vacía de
        cláusulas deja la búsqueda exactamente como si no hubiera filtros.
        """
        md = self.model.metadata_
        clauses: list = []

        if "sectors" in self.filters and filters.sectors:
            clauses.append(md["client_sector"].astext.in_(list(filters.sectors)))
        if "technologies" in self.filters and filters.technologies:
            clauses.append(md["main_technology"].astext.in_(list(filters.technologies)))
        if "min_date" in self.filters and filters.min_date is not None:
            if self.date_spec is not None and self.date_spec.kind == "year":
                min_year = filters.min_date.year
                clauses.append(cast(md[self.date_spec.key].astext, Integer) >= min_year)
            elif self.date_spec is not None and self.date_spec.kind == "iso":
                clauses.append(cast(md[self.date_spec.key].astext, Date) >= filters.min_date)
        if "version" in self.filters and filters.version is not None:
            clauses.append(md["version"].astext == filters.version)

        return clauses


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_spec(definition: CollectionDefinition) -> CollectionSpec:
    return CollectionSpec(
        collection=definition.name,
        label=definition.label,
        description=definition.description,
        model=_MODEL_BY_TABLE[definition.table],
        chunk_type=definition.chunk_type,
        source_id_key=definition.source_id_key,
        date_spec=definition.date,
        filters=tuple(definition.filters),
        rule_patterns=tuple(definition.rules),
    )


@lru_cache
def load_collections_catalog(
    path: Path | str = COLLECTIONS_CATALOG_PATH,
) -> dict[Collection, CollectionSpec]:
    """Carga y valida el catálogo una sola vez por proceso."""
    catalog_path = Path(path)
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog = CollectionsCatalog.model_validate(raw or {})
    return {definition.name: _to_spec(definition) for definition in catalog.collections}


def collections() -> dict[Collection, CollectionSpec]:
    """Catálogo operativo (fuente → spec)."""
    return load_collections_catalog()


def spec_for(collection: Collection) -> CollectionSpec:
    return collections()[collection]


def all_collections() -> tuple[Collection, ...]:
    """Orden canónico de búsqueda completa (fallback del enrutado)."""
    return tuple(collections())


def match_rules(query_text: str) -> list[Collection]:
    """Nivel 1 del enrutado: vocabulario determinista del catálogo.

    Devuelve las fuentes cuyo vocabulario aparece en la consulta, en el orden del
    catálogo. Una lista vacía significa que ninguna regla disparó y la cascada
    debe escalar al clasificador.
    """
    text = query_text.lower()
    hits: list[Collection] = []
    for collection, spec in collections().items():
        if any(re.search(pattern, text) for pattern in spec.rule_patterns):
            hits.append(collection)
    return hits
