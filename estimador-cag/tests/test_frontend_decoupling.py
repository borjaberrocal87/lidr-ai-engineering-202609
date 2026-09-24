"""Blinda la arquitectura por capas: el frontend no importa el backend."""

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _imports_app(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "app" or alias.name.startswith("app.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "app" or module.startswith("app."):
                return True
    return False


def test_frontend_does_not_import_backend() -> None:
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in FRONTEND_DIR.rglob("*.py")
        if _imports_app(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "El frontend no debe importar `app.*` (arquitectura por capas). "
        f"Archivos infractores: {offenders}"
    )


def test_streamlit_entrypoint_resolves_frontend_package() -> None:
    """Regresión: `streamlit run frontend/streamlit_app.py` debe poder importar `frontend`.

    Al ejecutar el script, Python pone en `sys.path` el directorio del script
    (`frontend/`), no la raíz, así que sin el bootstrap del entrypoint los
    imports fallan con `ModuleNotFoundError`. Lo ejecutamos desde otro cwd para
    reproducirlo.
    """
    result = subprocess.run(
        [sys.executable, str(FRONTEND_DIR / "streamlit_app.py")],
        cwd=str(PROJECT_ROOT.parent),
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "API_BASE_URL": "http://127.0.0.1:1"},
    )

    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert "No module named 'frontend'" not in combined, combined
