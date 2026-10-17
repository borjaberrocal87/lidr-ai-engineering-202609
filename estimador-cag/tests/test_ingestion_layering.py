"""Regla de dependencia del subsistema de ingesta (Clean Architecture).

El núcleo (caso de uso + modelos de dominio + puertos) no debe importar
infraestructura: ni SQLAlchemy, ni Presidio, ni pandas/pandera, ni Faker, ni
``app.persistence``. Sólo los adaptadores (``cleaning/``, ``pii/``,
``persistence/``) pueden tocar esas librerías. Este test convierte esa
convención en un contrato ejecutable.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"

# Núcleo: caso de uso, errores, puertos y modelos de dominio.
CORE_FILES = [
    APP / "ingestion" / "orchestrator.py",
    APP / "ingestion" / "errors.py",
    APP / "ingestion" / "ports.py",
    APP / "ingestion" / "documents" / "models.py",
    APP / "ingestion" / "catalog" / "models.py",
    APP / "ingestion" / "parsers" / "protocol.py",
]

# Prefijos que el núcleo tiene prohibido importar.
FORBIDDEN_PREFIXES = (
    "sqlalchemy",
    "presidio_analyzer",
    "presidio_anonymizer",
    "pandas",
    "pandera",
    "faker",
    "spacy",
    "app.persistence",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_core_does_not_import_infrastructure():
    violations: dict[str, list[str]] = {}
    for path in CORE_FILES:
        bad = sorted(
            module
            for module in _imported_modules(path)
            if any(
                module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES
            )
        )
        if bad:
            violations[path.name] = bad
    assert not violations, f"Frontera de capas violada: {violations}"
