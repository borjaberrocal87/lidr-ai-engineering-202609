"""Loader de sistema de ficheros.

Resuelve el ``CatalogSource.location`` contra el ``INGESTION_DATA_ROOT`` del
proyecto, recorre la ruta resultante y produce un ``LoadedBlob`` por cada
fichero cuya extensión coincide con el formato declarado de la fuente. Una
ubicación inexistente es un error duro: un catálogo que apunta a ninguna parte
está silenciosamente mal, el peor tipo de roto.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedBlob:
    """Los bytes de un único fichero más la ruta relativa dentro de la raíz de datos."""

    relative_path: str
    bytes_: bytes


class FileSystemLoader:
    def __init__(self, data_root: Path | str) -> None:
        self._data_root = Path(data_root)

    def iter_blobs(self, location: str, formats: set[str]) -> Iterator[LoadedBlob]:
        path = self._data_root / location
        if not path.exists():
            raise FileNotFoundError(
                f"Catalog location {location!r} does not resolve to a real path "
                f"({path!s} not found)"
            )
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            ext = candidate.suffix.lower().lstrip(".")
            if ext not in formats:
                continue
            yield LoadedBlob(
                relative_path=str(candidate.relative_to(self._data_root)),
                bytes_=candidate.read_bytes(),
            )
