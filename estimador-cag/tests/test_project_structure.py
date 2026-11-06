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
    # foundation
    "app/foundation/__init__.py",
    "app/foundation/attachments/__init__.py",
    "app/foundation/attachments/extractor.py",
    "app/foundation/cache/__init__.py",
    "app/foundation/cache/backends.py",
    "app/foundation/guardrails/__init__.py",
    "app/foundation/guardrails/input.py",
    "app/foundation/guardrails/output.py",
    "app/foundation/llm/__init__.py",
    "app/foundation/llm/errors.py",
    "app/foundation/llm/runtime_config.py",
    "app/foundation/llm/wrapper.py",
    "app/foundation/observability/__init__.py",
    "app/foundation/observability/logging.py",
    "app/foundation/persistence/__init__.py",
    "app/foundation/persistence/database.py",
    "app/foundation/persistence/models.py",
    "app/foundation/persistence/repositories/__init__.py",
    "app/foundation/persistence/repositories/jobs.py",
    "app/foundation/persistence/repositories/mappings.py",
    "app/foundation/persistence/unit_of_work.py",
    "app/foundation/prompts/__init__.py",
    "app/foundation/prompts/loader.py",
    "app/foundation/prompts/critic/v1/system.j2",
    "app/foundation/prompts/critic/v1/user.j2",
    "app/foundation/prompts/conversation_summary/v1/system.j2",
    "app/foundation/prompts/conversation_summary/v1/user.j2",
    "app/foundation/prompts/estimation/v1/system.j2",
    "app/foundation/prompts/estimation/v1/user.j2",
    "app/foundation/prompts/estimation/v1/examples.j2",
    "app/foundation/prompts/estimation/v2/system.j2",
    "app/foundation/prompts/estimation/v2/user.j2",
    "app/foundation/prompts/estimation/v2/examples.j2",
    "app/foundation/prompts/estimation/v3/system.j2",
    "app/foundation/prompts/estimation/v3/user.j2",
    "app/foundation/prompts/estimation/v3/examples.j2",
    "app/foundation/prompts/metadata_extraction/v1/system.j2",
    "app/foundation/prompts/metadata_extraction/v1/user.j2",
    # domain
    "app/domain/__init__.py",
    "app/domain/estimation_service.py",
    "app/domain/streaming_service.py",
    "app/domain/ports/__init__.py",
    "app/domain/ports/ingestion.py",
    "app/domain/schemas/__init__.py",
    "app/domain/schemas/acb.py",
    "app/domain/schemas/conversation.py",
    "app/domain/schemas/critic.py",
    "app/domain/schemas/estimations.py",
    "app/domain/schemas/ingestion.py",
    "app/domain/schemas/observability.py",
    # generation
    "app/generation/__init__.py",
    "app/generation/cag/__init__.py",
    "app/generation/cag/exact.py",
    "app/generation/cag/semantic.py",
    "app/generation/agentic/__init__.py",
    "app/generation/agentic/boss.py",
    "app/generation/agentic/critic.py",
    "app/generation/conversation/__init__.py",
    "app/generation/conversation/metadata_extractor.py",
    "app/generation/conversation/models.py",
    "app/generation/conversation/store.py",
    "app/generation/conversation/tier_resolver.py",
    "app/generation/conversation/compression/__init__.py",
    "app/generation/conversation/compression/anchors.py",
    "app/generation/conversation/compression/policy.py",
    "app/generation/conversation/compression/summarizer.py",
    "app/generation/rag/__init__.py",
    "app/generation/rag/ingest_service.py",
    "app/generation/rag/retriever.py",
    "app/generation/rag/schemas.py",
    "app/generation/rag/analysis/__init__.py",
    "app/generation/rag/analysis/comparison.py",
    "app/generation/rag/analysis/similarity.py",
    "app/generation/rag/chunking/__init__.py",
    "app/generation/rag/chunking/base.py",
    "app/generation/rag/chunking/chunker.py",
    "app/generation/rag/chunking/structural.py",
    "app/generation/rag/chunking/strategies/__init__.py",
    "app/generation/rag/embedding/__init__.py",
    "app/generation/rag/embedding/embedder.py",
    "app/generation/rag/store/__init__.py",
    "app/generation/rag/store/models.py",
    "app/generation/rag/store/repository.py",
    # ingestion
    "app/ingestion/__init__.py",
    "app/ingestion/architecture.py",
    "app/ingestion/orchestrator.py",
    # api
    "app/api/__init__.py",
    "app/api/config.py",
    "app/api/embeddings.py",
    "app/api/estimations.py",
    "app/api/ingestion.py",
    "app/api/search.py",
    "app/api/sessions.py",
    # project files
    "alembic/versions/0002_session8_pgvector.py",
    "scripts/query_examples.py",
    "data/output/session08/output_examples.txt",
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
    "docs/patron-actor-critic-boss.md",
    "docs/evals-golden-dataset.md",
    "docs/persistencia-pgvector-busqueda-semantica-sesion-08.md",
    "ARQUITECTURA.md",
    "tests/test_layering.py",
    "pyproject.toml",
    "README.md",
    ".env.example",
    ".gitignore",
)


def test_required_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).exists()]

    assert not missing, f"Faltan rutas requeridas en la estructura: {missing}"


def test_app_exposes_expected_entrypoints() -> None:
    from app.api import estimations, sessions
    from app.config import settings
    from app.domain.estimation_service import EstimationService
    from app.domain.streaming_service import generate_estimation
    from app.foundation.llm.errors import LLMConfigurationError
    from app.main import app

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
    assert "/api/v1/embeddings/ingest" in paths
    assert "/api/v1/embeddings/compare" in paths
    assert "/api/v1/search" in paths
    assert "/api/v1/config/models" in paths


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
