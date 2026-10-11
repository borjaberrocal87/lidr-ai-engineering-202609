"""Generador determinista de adjuntos PDF calibrados (sesión 6, bloque 3).

Produce PDFs de texto plano de tamaños ``5/20/50/100 KB`` para medir cómo escala
el CAG con el tamaño del adjunto (latencia, coste y recall del contenido). El
``0 KB`` del escenario es ausencia de fichero, así que no se genera.

Determinismo: fpdf2 estampa fecha de creación por defecto, lo que cambiaría los
bytes en cada run. Fijamos ``creation_date`` y calibramos el número de caracteres
con una búsqueda binaria monótona, de modo que dos ejecuciones producen bytes
idénticos. Los PDFs **no se commitean**: se regeneran desde el runner.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

SIZES_KB: tuple[int, ...] = (5, 20, 50, 100)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent

_FIXED_CREATION_DATE = datetime(2026, 1, 1, 0, 0, 0)

_LOREM = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
    "eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt "
    "in culpa qui officia deserunt mollit anim id est laborum "
)


def marker_for(target_kb: int) -> str:
    """Marcador estable y localizable que el runner busca en el PDF extraído."""
    return f"NIMBUS-ATTACH-MARKER-{target_kb}KB"


def _render(target_kb: int, body_chars: int) -> bytes:
    """Renderiza un PDF de ``body_chars`` caracteres de relleno (en memoria)."""
    pdf = FPDF(format="A4")
    pdf.set_creation_date(_FIXED_CREATION_DATE)
    pdf.set_title(f"Nimbus synthetic attachment {target_kb}KB")
    pdf.set_author("evals.stress")
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 6, marker_for(target_kb), new_x="LMARGIN", new_y="NEXT")
    body = (_LOREM * (body_chars // len(_LOREM) + 1))[:body_chars]
    pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def build_pdf(*, path: Path, target_kb: int) -> Path:
    """Genera ``path`` como un PDF de ~``target_kb`` KB y devuelve la ruta.

    La búsqueda binaria asume que el tamaño del PDF crece de forma monótona con
    el relleno, cierto para texto comprimido con zlib en un único flujo.
    """
    target_bytes = target_kb * 1024
    low, high = 1, 4_000_000
    best = _render(target_kb, low)
    for _ in range(25):
        mid = (low + high) // 2
        candidate = _render(target_kb, mid)
        if len(candidate) <= target_bytes:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
        if low > high:
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(best)
    return path


def build_all(out_dir: Path | None = None, sizes: tuple[int, ...] = SIZES_KB) -> dict[int, Path]:
    """Genera todos los tamaños y devuelve ``{kb: ruta}`` (excluye el 0 KB)."""
    directory = out_dir or DEFAULT_OUT_DIR
    return {kb: build_pdf(path=directory / f"attach_{kb}kb.pdf", target_kb=kb) for kb in sizes}


def main() -> int:
    paths = build_all()
    for kb, path in paths.items():
        print(f"{kb:>4} KB -> {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
