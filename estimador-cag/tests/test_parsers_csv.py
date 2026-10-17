"""Tests del parser CSV propio (formato ``csv``)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.ingestion.catalog.models import CatalogSource, QualityScore, Sensitivity
from app.ingestion.loaders.filesystem import LoadedBlob
from app.ingestion.parsers.csv_records import CsvRecordsParser
from app.ingestion.parsers.protocol import ParseContext
from app.ingestion.parsers.registry import default_registry


def _context() -> ParseContext:
    source = CatalogSource(
        name="presupuestos_csv",
        location="presupuestos_extra.csv",
        format="csv",
        quality=QualityScore(completeness=4, consistency=4, actuality=5, reliability=4),
        sensitivity=Sensitivity(has_pii=True, pii_flags=["PERSON"]),
        decision="include",
    )
    return ParseContext(
        source=source,
        source_version="1.1.0",
        ingested_at=datetime(2026, 5, 22, 10, 0, tzinfo=UTC),
    )


def test_csv_parser_emits_one_document_per_row():
    csv_bytes = b"budget_id,client_name\nBUDGET-2024-0001,Acme\nBUDGET-2024-0002,Beta\n"
    blob = LoadedBlob(relative_path="presupuestos_extra.csv", bytes_=csv_bytes)

    documents = list(CsvRecordsParser().parse(blob, _context()))

    assert len(documents) == 2
    assert documents[0].metadata.extra["budget_id"] == "BUDGET-2024-0001"
    assert "Acme" in documents[0].text
    assert documents[1].id.endswith(":2:presupuestos_extra.csv")


def test_csv_parser_is_registered_in_default_registry():
    assert "csv" in default_registry().formats()
