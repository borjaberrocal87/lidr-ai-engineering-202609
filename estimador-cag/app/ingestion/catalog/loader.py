"""Carga y valida el catálogo YAML.

Lee un fichero YAML con ``yaml.safe_load``, lo valida contra
:class:`DataCatalog`. El ``load_catalog`` de dos líneas es intencionadamente
trivial — todo el trabajo pesado (reglas de decisión, detección de duplicados)
vive en el modelo, no aquí.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from app.ingestion.catalog.models import DataCatalog


def load_catalog(path: str | Path) -> DataCatalog:
    """Interpreta ``path`` como YAML y lo valida como un :class:`DataCatalog`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DataCatalog.model_validate(raw)


def _main(argv: list[str]) -> int:
    """CLI: ``python -m app.ingestion.catalog.loader data/catalog/catalog.yaml``."""
    if len(argv) != 2:
        print("usage: python -m app.ingestion.catalog.loader <path.yaml>")
        return 1
    catalog = load_catalog(argv[1])
    included = catalog.included_sources()
    print(f"Catalog version: {catalog.version}")
    print(f"Sources total:  {len(catalog.sources)}")
    print(f"Sources included: {len(included)}")
    for src in included:
        print(f"  - {src.name} ({src.format}) — {src.description!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
