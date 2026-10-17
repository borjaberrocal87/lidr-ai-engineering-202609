"""Parser CSV propio (formato ``csv``).

Tercer parser de la rama, añadido para demostrar que el contrato `Parser` +
`ParserRegistry` es extensible sin tocar el orquestador. Emite **un `Document`
por fila** (granularidad fina, útil para filtros por columna) renderizando la
fila como markdown `- clave: valor`.

Diferencia con `budget_json`: aquí no hay esquema de dominio, cada fila es un
registro genérico cuyas columnas van a `metadata.extra`.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from typing import ClassVar

from app.ingestion.documents.models import Document, DocumentMetadata
from app.ingestion.loaders.filesystem import LoadedBlob
from app.ingestion.parsers.protocol import ParseContext


class CsvRecordsParser:
    supported_formats: ClassVar[set[str]] = {"csv"}

    def parse(self, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        reader = csv.DictReader(io.StringIO(blob.bytes_.decode("utf-8")))
        for index, raw_row in enumerate(reader, start=1):
            record = {key: (value or "") for key, value in raw_row.items() if key}
            if not record:
                continue
            yield Document(
                id=f"{context.source.name}:{index}:{blob.relative_path}",
                text=_render_row_markdown(record),
                metadata=DocumentMetadata(
                    source_name=context.source.name,
                    source_version=context.source_version,
                    ingested_at=context.ingested_at,
                    lineage=list(context.source.lineage),
                    sensitivity_pii_flags=list(context.source.sensitivity.pii_flags),
                    sensitivity_access_level=context.source.sensitivity.access_level,
                    location=blob.relative_path,
                    extra=dict(record),
                ),
            )


def _render_row_markdown(record: dict[str, str]) -> str:
    lines = ["# Registro CSV", ""]
    for key, value in record.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip() + "\n"
