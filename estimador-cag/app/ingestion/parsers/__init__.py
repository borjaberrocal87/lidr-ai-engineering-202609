"""Parsers — bytes → ``Document``.

El :class:`Parser` es un :class:`typing.Protocol`, no una clase base. El tipado
estructural es la herramienta adecuada aquí: un parser es cualquier cosa que
exponga ``supported_formats`` y ``parse``. En ningún caso queremos que un parser
herede estado compartido de un padre: son funciones puras sobre bytes.
"""

from app.ingestion.parsers.budget_json import BudgetJsonParser
from app.ingestion.parsers.csv_records import CsvRecordsParser
from app.ingestion.parsers.protocol import ParseContext, Parser
from app.ingestion.parsers.registry import ParserRegistry, default_registry
from app.ingestion.parsers.transcript_txt import TranscriptTxtParser

__all__ = [
    "BudgetJsonParser",
    "CsvRecordsParser",
    "ParseContext",
    "Parser",
    "ParserRegistry",
    "TranscriptTxtParser",
    "default_registry",
]
