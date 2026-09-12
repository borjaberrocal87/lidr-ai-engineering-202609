"""Valida que la estructura de carpetas del proyecto es la definida en el spec.

Forma parte del entregable: comprobar de manera automática que el scaffolding
existe y que el servicio expone lo esperado. Se ejecuta en local y en CI.
"""

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/routers/__init__.py",
    "app/routers/estimations.py",
    "app/services/__init__.py",
    "app/services/llm_service.py",
    "app/context/__init__.py",
    "app/context/examples.py",
    "tests/__init__.py",
    "pyproject.toml",
    "README.md",
    ".env.example",
    ".gitignore",
)


def test_required_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).exists()]

    assert not missing, f"Faltan rutas requeridas en la estructura: {missing}"


def test_app_exposes_expected_entrypoints() -> None:
    from app.config import settings
    from app.main import app
    from app.routers import estimations
    from app.services.llm_service import LLMConfigurationError, generate_estimation

    assert app.title
    assert settings.llm_provider in {"openai", "anthropic"}
    assert callable(generate_estimation)
    assert issubclass(LLMConfigurationError, RuntimeError)
    assert estimations.router is not None

    paths = set(app.openapi()["paths"])
    assert "/health" in paths
    assert "/api/v1/estimate" in paths


def test_env_file_is_gitignored() -> None:
    entries = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert ".env" in entries


def _tracked_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "."],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git no está disponible en este entorno")
    return result.stdout.splitlines()


def test_secrets_are_not_tracked() -> None:
    tracked = set(_tracked_files())

    assert ".env.example" in tracked
    assert ".env" not in tracked, "El archivo .env con secretos no debe estar en git"
