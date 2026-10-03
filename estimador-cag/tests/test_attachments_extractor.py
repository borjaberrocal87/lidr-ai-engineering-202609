"""Tests unitarios del extractor local de adjuntos (Camino B)."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from app.attachments.extractor import (
    AttachmentExtractionError,
    UnsupportedAttachmentError,
    enrich_transcript,
    extract_text,
)


def _docx_bytes(paragraphs: list[str]) -> bytes:
    from docx import Document

    document = Document()
    for line in paragraphs:
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_docx_extraction_returns_joined_paragraphs() -> None:
    content = _docx_bytes(["Proyecto Nimbus", "React + Postgres", "Fase 1: descubrimiento"])

    out = extract_text(filename="spec.docx", content=content, max_chars=10_000)

    assert "Proyecto Nimbus" in out
    assert "React + Postgres" in out
    assert "Fase 1: descubrimiento" in out


def test_docx_extraction_truncates_at_max_chars() -> None:
    content = _docx_bytes(["a" * 5_000, "b" * 5_000])

    out = extract_text(filename="long.docx", content=content, max_chars=1_000)

    assert len(out) == 1_000


def test_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedAttachmentError):
        extract_text(filename="foto.png", content=b"", max_chars=100)


def test_pdf_extraction_calls_pypdf(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pages = [
        SimpleNamespace(extract_text=lambda: "Página uno"),
        SimpleNamespace(extract_text=lambda: "Página dos"),
    ]
    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", lambda _stream: SimpleNamespace(pages=fake_pages))

    out = extract_text(filename="propuesta.pdf", content=b"%PDF-fake", max_chars=10_000)

    assert "Página uno" in out
    assert "Página dos" in out


def test_pdf_extraction_swallows_per_page_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> str:
        raise RuntimeError("página corrupta")

    fake_pages = [SimpleNamespace(extract_text=fail), SimpleNamespace(extract_text=lambda: "ok")]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", lambda _stream: SimpleNamespace(pages=fake_pages))

    out = extract_text(filename="mixed.pdf", content=b"%PDF-fake", max_chars=10_000)

    assert "ok" in out


def test_pdf_extraction_wraps_global_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_stream: Any) -> SimpleNamespace:
        raise ValueError("no es un pdf")

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", boom)

    with pytest.raises(AttachmentExtractionError):
        extract_text(filename="malo.pdf", content=b"xxxx", max_chars=100)


def test_enrich_transcript_wraps_attachments_with_fences() -> None:
    out = enrich_transcript(
        transcript="Petición: construir un CRM.",
        attachments=[("spec.pdf", "Cuerpo de la spec."), ("notas.docx", "Notas internas.")],
    )

    assert "Petición: construir un CRM." in out
    assert "--- attachment: spec.pdf ---" in out
    assert "Cuerpo de la spec." in out
    assert "--- end attachment ---" in out
    assert "--- attachment: notas.docx ---" in out


def test_enrich_transcript_returns_original_when_no_attachments() -> None:
    assert enrich_transcript(transcript="hola", attachments=[]) == "hola"


def test_enrich_transcript_skips_empty_attachments() -> None:
    out = enrich_transcript(
        transcript="x", attachments=[("vacio.pdf", ""), ("real.docx", "contenido")]
    )

    assert "vacio.pdf" not in out
    assert "real.docx" in out
