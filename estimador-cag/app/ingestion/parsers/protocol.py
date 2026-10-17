"""El protocolo :class:`Parser` y el :class:`ParseContext` que se le pasa.

¿Por qué un Protocol en lugar de una clase base abstracta? Dos razones:

* Los parsers no comparten implementación: no hay ningún método útil que heredar.
* El tipado estructural hace trivial colocar un fake durante los tests sin
  heredar de nada; cualquier clase con ``supported_formats`` y ``parse``
  satisface el contrato.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Protocol, runtime_checkable

from app.ingestion.catalog.models import CatalogSource
from app.ingestion.documents.models import Document
from app.ingestion.loaders.filesystem import LoadedBlob


@dataclass(frozen=True)
class ParseContext:
    """Todo lo que un parser necesita y que no vive en el propio blob.

    El orquestador lo construye una vez por corrida de ingesta y lo reutiliza
    entre blobs, de modo que ``ingested_at`` y ``source_version`` son constantes
    durante toda la corrida: esa constancia es lo que permite que aguas abajo se
    deduplique por hash.
    """

    source: CatalogSource
    source_version: str
    ingested_at: datetime


@runtime_checkable
class Parser(Protocol):
    supported_formats: ClassVar[set[str]]

    def parse(
        self, blob: LoadedBlob, context: ParseContext
    ) -> Iterable[Document]:  # pragma: no cover - protocol stub
        ...
