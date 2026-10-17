"""Loaders — productores de bytes en crudo.

Un loader sabe enumerar ficheros (o URLs, o filas de BD) y entregar al parser un
flujo de pares ``(name, bytes)``. Los loaders nunca miran DENTRO de los bytes
— eso es trabajo del parser. La separación mantiene ambas mitades lo bastante
pequeñas como para testearlas.
"""

from app.ingestion.loaders.filesystem import FileSystemLoader, LoadedBlob

__all__ = ["FileSystemLoader", "LoadedBlob"]
