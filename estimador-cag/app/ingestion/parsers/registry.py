"""Registro mínimo de parsers — mapea una cadena de formato a una instancia de
:class:`Parser`.

El registro es un simple dict disfrazado. Lo mantenemos como clase para poder
hacer exigible la invariante "un solo parser por formato" y para dar al
orquestador un único objeto del que depender.
"""

from __future__ import annotations

from app.ingestion.parsers.budget_json import BudgetJsonParser
from app.ingestion.parsers.csv_records import CsvRecordsParser
from app.ingestion.parsers.protocol import Parser
from app.ingestion.parsers.transcript_txt import TranscriptTxtParser


class ParserRegistry:
    def __init__(self) -> None:
        self._by_format: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        for fmt in parser.supported_formats:
            if fmt in self._by_format:
                raise ValueError(
                    f"Format {fmt!r} already has a registered parser "
                    f"({type(self._by_format[fmt]).__name__})"
                )
            self._by_format[fmt] = parser

    def get(self, fmt: str) -> Parser:
        try:
            return self._by_format[fmt]
        except KeyError as exc:
            raise KeyError(f"No parser registered for format {fmt!r}") from exc

    def formats(self) -> set[str]:
        return set(self._by_format)


def default_registry() -> ParserRegistry:
    """Devuelve un registro precargado con los parsers disponibles en esta rama.

    Los parsers de XLSX/DOCX/PDF NO se registran aquí a propósito en la Sesión 6;
    se entregan en ``guides/session-06-reference/``. Añadirlos más adelante es una
    llamada de una sola línea ``registry.register(XlsxParser())``.
    """
    registry = ParserRegistry()
    registry.register(BudgetJsonParser())
    registry.register(TranscriptTxtParser())
    registry.register(CsvRecordsParser())
    return registry
