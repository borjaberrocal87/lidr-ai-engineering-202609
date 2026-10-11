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
    "app/dependencies.py",
    "app/routers/__init__.py",
    "app/routers/estimations.py",
    "app/routers/sessions.py",
    "app/sessions/__init__.py",
    "app/sessions/models.py",
    "app/sessions/store.py",
    "app/sessions/metadata_extractor.py",
    "app/attachments/__init__.py",
    "app/attachments/extractor.py",
    "app/services/__init__.py",
    "app/services/estimation.py",
    "app/services/streaming.py",
    "app/services/boss.py",
    "app/services/critic.py",
    "app/services/llm_wrapper.py",
    "app/services/cache.py",
    "app/services/errors.py",
    "app/sessions/tier_resolver.py",
    "app/sessions/compression/__init__.py",
    "app/sessions/compression/anchors.py",
    "app/sessions/compression/policy.py",
    "app/sessions/compression/summarizer.py",
    "app/schemas/acb.py",
    "app/schemas/critic.py",
    "app/prompts/critic/v1/system.j2",
    "app/prompts/critic/v1/user.j2",
    "app/prompts/conversation_summary/v1/system.j2",
    "app/prompts/conversation_summary/v1/user.j2",
    "app/prompts/__init__.py",
    "app/prompts/loader.py",
    "app/prompts/estimation/v1/system.j2",
    "app/prompts/estimation/v1/user.j2",
    "app/prompts/estimation/v1/examples.j2",
    "app/prompts/estimation/v2/system.j2",
    "app/prompts/estimation/v2/user.j2",
    "app/prompts/estimation/v2/examples.j2",
    "app/prompts/estimation/v3/system.j2",
    "app/prompts/estimation/v3/user.j2",
    "app/prompts/estimation/v3/examples.j2",
    "app/prompts/metadata_extraction/v1/system.j2",
    "app/prompts/metadata_extraction/v1/user.j2",
    "app/schemas/__init__.py",
    "app/schemas/estimations.py",
    "frontend/__init__.py",
    "frontend/config.py",
    "frontend/models.py",
    "frontend/client.py",
    "frontend/streamlit_app.py",
    "tests/__init__.py",
    "tests/test_cache.py",
    "tests/test_llm_wrapper.py",
    "tests/test_schemas.py",
    "tests/prompts/__init__.py",
    "tests/prompts/test_estimation_v1.py",
    "tests/prompts/test_estimation_versions.py",
    "tests/prompts/test_estimation_v3.py",
    "tests/prompts/test_prompt_logging.py",
    "tests/test_sessions_models.py",
    "tests/test_attachments_extractor.py",
    "tests/test_sessions_endpoints.py",
    "tests/test_critic.py",
    "tests/test_acb_boss.py",
    "tests/test_sessions_anchors.py",
    "tests/test_sessions_compression.py",
    "tests/test_tier_resolver.py",
    "tests/test_evals_dataset.py",
    "evals/__init__.py",
    "evals/dataset.py",
    "evals/metrics.py",
    "evals/run.py",
    "evals/golden_dataset.json",
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
    from app.routers import estimations, sessions
    from app.services.errors import LLMConfigurationError
    from app.services.estimation import EstimationService
    from app.services.streaming import generate_estimation

    assert app.title
    assert settings.llm_provider in {"openai", "anthropic", "custom"}
    assert callable(generate_estimation)
    assert callable(EstimationService.estimate_conversational)
    assert issubclass(LLMConfigurationError, RuntimeError)
    assert estimations.router is not None
    assert sessions.router is not None

    paths = set(app.openapi()["paths"])
    assert "/health" in paths
    assert "/api/v1/estimate" in paths
    assert "/api/v1/estimate/stream" in paths
    assert "/api/v1/context" in paths
    assert "/api/v1/sessions" in paths
    assert "/api/v1/sessions/{session_id}" in paths
    assert "/api/v1/sessions/{session_id}/estimate" in paths


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
