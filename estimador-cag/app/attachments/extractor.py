"""Extracción local de texto de adjuntos PDF/DOCX (Camino B).

El router es quien tiene los `UploadFile`; aquí solo entra ``(filename, bytes)``.
El texto se trunca a ``max_chars`` para proteger el presupuesto del prompt. La
extracción local desacopla el servicio del proveedor y deja el texto preparado
para el chunking de RAG.
"""

from __future__ import annotations

import io
from pathlib import PurePosixPath

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_SUPPORTED_EXTS = {".pdf", ".docx"}


class AttachmentExtractionError(Exception):
    """Fallo genérico de extracción (PDF corrupto, DOCX ilegible…)."""

    def __init__(self, filename: str, message: str) -> None:
        super().__init__(message)
        self.filename = filename
        self.message = message


class UnsupportedAttachmentError(AttachmentExtractionError):
    """Extensión no soportada."""


def _extension(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower()


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("pdf_page_extract_failed", error=str(exc)[:200])
            text = ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx(content: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(content))
    return "\n".join(p.text for p in document.paragraphs if p.text and p.text.strip())


def extract_text(*, filename: str, content: bytes, max_chars: int) -> str:
    """Devuelve el texto extraído del adjunto, truncado a ``max_chars``.

    Lanza ``UnsupportedAttachmentError`` para extensiones desconocidas y
    ``AttachmentExtractionError`` para fallos del parser.
    """
    ext = _extension(filename)
    if ext not in _SUPPORTED_EXTS:
        raise UnsupportedAttachmentError(
            filename,
            f"Extensión no soportada: {ext!r}; soportadas: {sorted(_SUPPORTED_EXTS)}",
        )

    try:
        text = _extract_pdf(content) if ext == ".pdf" else _extract_docx(content)
    except AttachmentExtractionError:
        raise
    except Exception as exc:
        raise AttachmentExtractionError(filename, f"No se pudo extraer {filename}: {exc}") from exc

    text = text.strip()
    if not text:
        logger.info("attachment_extracted_empty", filename=filename)
        return ""

    if len(text) > max_chars:
        logger.info(
            "attachment_truncated",
            filename=filename,
            original_chars=len(text),
            kept_chars=max_chars,
        )
        text = text[:max_chars]
    return text


def enrich_transcript(*, transcript: str, attachments: list[tuple[str, str]]) -> str:
    """Concatena la transcripción con cada adjunto entre fences explícitos.

    ``attachments`` es una lista de ``(filename, extracted_text)`` ya filtrada
    de contenidos vacíos.
    """
    if not attachments:
        return transcript

    parts = [transcript.strip()]
    for filename, text in attachments:
        if text:
            parts.append(f"--- attachment: {filename} ---\n{text}\n--- end attachment ---")
    return "\n\n".join(parts)
