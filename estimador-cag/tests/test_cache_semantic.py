"""Tests de la caché semántica (índice y vectorizer mockeados, sin Redis)."""

from typing import Any

from app.cache.semantic import EstimationSemanticCache, _similarity, _to_bytes
from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
)


def _result() -> EstimationResult:
    return EstimationResult(
        summary="Una landing page con CRM, blog y formulario de contacto.",
        confidence_pct=70,
        phases=[
            Phase(
                name="Discovery",
                duration_weeks=1,
                cost_eur=2000,
                summary="Alcance y entrevistas con marketing.",
            )
        ],
        total_duration_weeks=1,
        total_cost_eur=2000,
    )


def _request() -> EstimationRequest:
    return EstimationRequest(
        description="Plataforma de inventario para cinco tiendas con alertas y dashboard.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )


class _FakeVectorizer:
    def embed(self, content: Any = None, text: Any = None, **kwargs: Any) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeIndex:
    def __init__(self, results: list[dict] | None = None, *, boom: bool = False) -> None:
        self.results = results or []
        self.boom = boom
        self.loaded: list[tuple[list[dict], int | None]] = []
        self.queries: list[Any] = []

    def query(self, query: Any) -> list[dict]:
        self.queries.append(query)
        return list(self.results)

    def load(self, data: list[dict], ttl: int | None = None) -> list[str]:
        if self.boom:
            raise RuntimeError("redis down")
        self.loaded.append((data, ttl))
        return ["key"]


def _cache(
    *, threshold: float = 0.85, log_only: bool = False, index: _FakeIndex | None = None
) -> EstimationSemanticCache:
    cache = EstimationSemanticCache.__new__(EstimationSemanticCache)
    cache.redis_client = None
    cache.vectorizer = _FakeVectorizer()
    cache.index = index if index is not None else _FakeIndex()
    cache.threshold = threshold
    cache.ttl = 3600
    cache.log_only = log_only
    return cache


def _row(distance: float, result: EstimationResult | None = None) -> dict[str, Any]:
    return {
        "result_json": (result or _result()).model_dump_json(),
        "vector_distance": distance,
        "bucket": "v1:web_saas:medium:phases_table",
    }


def test_bucket_includes_version_and_enums() -> None:
    bucket = EstimationSemanticCache.bucket_for(_request(), "v2")

    assert bucket == "v2:web_saas:medium:phases_table"


def test_lookup_returns_hit_above_threshold() -> None:
    cache = _cache(index=_FakeIndex([_row(0.05)]))

    hit = cache.lookup(_request(), "v1")

    assert hit is not None
    assert hit.total_cost_eur == 2000
    # Un solo filtro por bucket en la consulta.
    assert len(cache.index.queries) == 1


def test_lookup_misses_below_threshold() -> None:
    cache = _cache(index=_FakeIndex([_row(0.5)]))

    assert cache.lookup(_request(), "v1") is None


def test_lookup_misses_on_empty_index() -> None:
    cache = _cache(index=_FakeIndex([]))

    assert cache.lookup(_request(), "v1") is None


def test_lookup_log_only_never_returns_hit() -> None:
    index = _FakeIndex([_row(0.01)])
    cache = _cache(log_only=True, index=index)

    assert cache.lookup(_request(), "v1") is None
    assert len(index.queries) == 1


def test_store_loads_payload_with_ttl() -> None:
    index = _FakeIndex()
    cache = _cache(index=index)

    cache.store(_request(), _result(), "v1")

    [(payload, ttl)] = index.loaded
    assert ttl == 3600
    assert payload[0]["bucket"] == "v1:web_saas:medium:phases_table"
    assert isinstance(payload[0]["embedding"], bytes)
    assert "summary" in payload[0]["result_json"]


def test_store_swallows_errors() -> None:
    cache = _cache(index=_FakeIndex(boom=True))

    cache.store(_request(), _result(), "v1")  # no debe lanzar


def test_similarity_accepts_both_shapes() -> None:
    assert _similarity({"vector_distance": 0.1}) == 0.9
    assert _similarity({"similarity": 0.95}) == 0.95


def test_to_bytes_returns_float32_bytes() -> None:
    raw = _to_bytes([1.0, 2.0])

    assert isinstance(raw, bytes)
    assert len(raw) == 8  # 2 floats de 4 bytes


def test_service_uses_semantic_cache(monkeypatch) -> None:
    from app.services import llm_service
    from tests.conftest import FakeWrapper

    fake_wrapper = FakeWrapper()
    monkeypatch.setattr(llm_service, "get_llm_wrapper", lambda: fake_wrapper)

    expected = _result()

    class _FakeSemantic:
        def lookup(self, request: Any, version: str) -> EstimationResult:
            return expected

        def store(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("no debe almacenar en un acierto de caché")

    monkeypatch.setattr(llm_service, "get_semantic_cache", lambda: _FakeSemantic())

    outcome = llm_service.generate_structured_estimation(_request())

    assert outcome.cache_hit is True
    assert outcome.result == expected
    assert fake_wrapper.structured_calls == []
