"""Fábricas de dependencias compartidas (caché y wrapper LLM).

Se cachean con `lru_cache` para reutilizar el `Router` de LiteLLM y la conexión
de Redis durante toda la vida del proceso. Los tests pueden sustituir
`get_llm_wrapper` con un doble.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
import redis
import structlog
from openai import AsyncOpenAI, OpenAI

from app.config import BASE_DIR, get_settings, reveal_secret
from app.foundation.cache.backends import InMemoryCache, NullCache, RedisCache, ResponseCache
from app.foundation.llm.runtime_config import RuntimeModelConfig, RuntimeRetrievalConfig
from app.foundation.llm.wrapper import LLMWrapper, provider_from_model
from app.foundation.persistence.database import get_async_session_factory
from app.foundation.persistence.unit_of_work import UnitOfWork
from app.generation.agentic.estimation_agent.schemas import SearchBudgetsArgs
from app.generation.agentic.estimation_agent.tools import BudgetSearchBackend
from app.generation.cag.semantic import EstimationSemanticCache
from app.generation.conversation.store import SessionStore
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.chunking.strategies import (
    ContextualRetrievalChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
)
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.ingest_service import RagIngestService
from app.generation.rag.quality.augmentation import AugmentationPolicy
from app.generation.rag.quality.hallucination import HallucinationGate
from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.retrieval.sources import Collection
from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.store.repository import ChunkStore
from app.ingestion.catalog import DataCatalog, load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import ParserRegistry, default_registry

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from tiktoken import Encoding

    from app.domain.estimation_service import EstimationService
    from app.domain.ports.ingestion import TextPseudonymizer
    from app.generation.rag.idempotency import IdempotencyStore
    from app.ingestion.pii import ConsistentPseudonymizer

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@lru_cache
def get_cache() -> ResponseCache:
    settings = get_settings()
    if settings.cache_backend == "none":
        return NullCache()
    if settings.cache_backend == "redis":
        return RedisCache.from_url(settings.redis_url, ttl=settings.cache_ttl)
    return InMemoryCache(ttl=settings.cache_ttl)


@lru_cache
def get_runtime_config() -> RuntimeModelConfig:
    """Store Redis de overrides de modelo (Ajustes).

    El singleton solo contiene el handle de Redis: la frescura viene de leer
    Redis en cada llamada, no de reconstruir el objeto.
    """
    settings = get_settings()
    return RuntimeModelConfig.from_url(settings.redis_url, settings)


@lru_cache
def get_runtime_retrieval_config() -> RuntimeRetrievalConfig:
    """Redis-backed override store for the Session 10 retrieval toggles.

    Read per call so a flip in the config API takes effect on the next
    retrieval without a restart.
    """
    settings = get_settings()
    return RuntimeRetrievalConfig.from_url(settings.redis_url, settings)


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Cross-encoder reranker singleton.

    Building the wrapper is cheap; the model loads lazily on the first rerank,
    so import time and app startup stay free of torch.
    """
    return CrossEncoderReranker.from_settings()


# --- Sesión 11 (directo) — palancas de calidad de la generación ---------------


def build_augmentation_policy() -> AugmentationPolicy | None:
    """Política de augmentación vigente, o ``None`` si la palanca está apagada.

    Se construye por llamada (no se cachea): el toggle y sus sub-palancas se leen
    en caliente para que Ajustes aplique en la siguiente petición.
    """
    if not get_runtime_retrieval_config().effective_augmentation():
        return None
    settings = get_settings()
    return AugmentationPolicy(
        compress=settings.augmentation_compress,
        reorder=settings.augmentation_reorder,
    )


def build_hallucination_gate() -> HallucinationGate | None:
    """Gate semántico configurado, o ``None`` si está apagado en runtime.

    El juez reutiliza el wrapper LLM (nunca una API cruda) y toma el modelo
    efectivo del store de modelos: cambiarlo en Ajustes surte efecto sin reinicio.
    """
    if not get_runtime_retrieval_config().effective_hallucination_gate():
        return None
    return HallucinationGate(
        judge=get_llm_wrapper(),
        model=get_runtime_config().effective("HALLUCINATION_JUDGE_MODEL"),
        tolerance=get_settings().hallucination_numeric_tolerance,
    )


def get_corpus_index_service() -> CorpusIndexService | None:
    """Expansión incremental del corpus. ``None`` sin embeddings configurados."""
    ingest = get_rag_ingest_service()
    if ingest is None:
        return None
    return CorpusIndexService(ingest)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    fallback_model = settings.llm_fallback_model
    return LLMWrapper(
        primary_model=settings.llm_model,
        primary_provider=settings.llm_provider,
        fallback_model=fallback_model,
        fallback_provider=provider_from_model(fallback_model) if fallback_model else None,
        routing_mode=settings.llm_routing_mode,
        timeout=settings.llm_timeout_seconds,
        num_retries=settings.llm_max_retries,
        cache=get_cache(),
        openai_api_key=reveal_secret(settings.open_ai_key),
        anthropic_api_key=reveal_secret(settings.anthropic_api_key),
        custom_base_url=settings.custom_llm_base_url,
        custom_api_key=reveal_secret(settings.custom_llm_api_key),
        runtime_config=get_runtime_config(),
    )


@lru_cache
def get_session_store() -> SessionStore:
    """Store de sesiones en memoria, uno por worker.

    `lru_cache` lo hace singleton durante la vida del proceso; los tests pueden
    sustituirlo con `app.dependency_overrides[get_session_store]`.
    """
    settings = get_settings()
    return SessionStore(max_turns=settings.max_conversation_turns)


def get_estimation_service() -> EstimationService:
    """Orquestador de estimación (structured + conversacional + ACB).

    No se cachea: ``EstimationService`` no tiene estado y lee su configuración
    de ``settings`` en cada llamada, así que devolver una instancia nueva por
    petición es barato y mantiene los tests libres de estado compartido.

    El import es diferido para evitar el ciclo ``dependencies ↔ estimation``
    (``estimation`` importa los getters de infraestructura de este módulo).
    """
    from app.domain.estimation_service import EstimationService

    return EstimationService()


@lru_cache
def get_openai_client() -> OpenAI | None:
    """Cliente OpenAI perezoso para la moderación y los embeddings.

    Devuelve ``None`` si no hay clave o si la moderación está desactivada, de
    modo que el pipeline sigue funcionando solo con las capas regex.
    """
    settings = get_settings()
    if not settings.has_moderation:
        return None
    key = reveal_secret(settings.open_ai_key)
    if not key:
        return None
    return OpenAI(api_key=key)


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Construye la caché semántica, degradando si falta algo.

    Devuelve ``None`` (con warning) si está desactivada, si no hay embeddings
    configurados o si Redis Stack / RediSearch no está disponible. Así la
    generación sigue funcionando sin la capa semántica.
    """
    settings = get_settings()
    if not settings.semantic_cache_enabled or not settings.has_embeddings:
        return None

    try:
        from redisvl.utils.vectorize import OpenAITextVectorizer

        api_key = reveal_secret(settings.embedding_api_key) or reveal_secret(settings.open_ai_key)
        api_config: dict[str, str] = {"api_key": api_key or ""}
        if settings.embedding_base_url:
            api_config["base_url"] = settings.embedding_base_url
        vectorizer = OpenAITextVectorizer(model=settings.embedding_model, api_config=api_config)

        redis_client = redis.from_url(settings.redis_url, decode_responses=False)
        return EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            dims=vectorizer.dims,
            model=settings.embedding_model,
            threshold=settings.semantic_cache_threshold,
            ttl=settings.semantic_cache_ttl,
            log_only=settings.semantic_cache_log_only,
        )
    except Exception as exc:
        logger.warning(
            "semantic_cache_disabled",
            reason="setup_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return None


# --- Sesión 6 — ingesta de datos -------------------------------------------------


def _resolve_under_base_dir(path: Path) -> Path:
    """Resuelve rutas relativas de configuración contra la raíz del proyecto."""
    return path if path.is_absolute() else BASE_DIR / path


@lru_cache
def get_catalog() -> DataCatalog:
    """Carga y cachea el catálogo de fuentes (una vez por proceso).

    Releerlo invalidaría las decisiones ya horneadas en el pipeline en marcha;
    desplegar una versión nueva del catálogo exige reiniciar el proceso.
    """
    settings = get_settings()
    return load_catalog(_resolve_under_base_dir(settings.catalog_path))


@lru_cache
def get_filesystem_loader() -> FileSystemLoader:
    settings = get_settings()
    return FileSystemLoader(data_root=_resolve_under_base_dir(settings.ingestion_data_root))


@lru_cache
def get_parser_registry() -> ParserRegistry:
    """Registry con los parsers disponibles en esta rama (json, txt)."""
    return default_registry()


def get_unit_of_work() -> Iterator[UnitOfWork]:
    """Dependencia FastAPI: cede un UoW y lo cierra al terminar la request.

    No hace commit automático: la frontera decide cuándo cerrar la transacción
    (`uow.commit()`), tal como exige el patrón UoW ligero.
    """
    uow = UnitOfWork()
    try:
        yield uow
    finally:
        uow.close()


def build_pseudonymizer(session: Session) -> ConsistentPseudonymizer:
    """Construye un ``ConsistentPseudonymizer`` respaldado por Postgres.

    No es singleton: el mapping store envuelve una ``Session``, así que quien lo
    llame (scripts, BackgroundTasks, tests) debe pasar la suya. El analyzer (sí
    singleton vía ``build_analyzer``) y la salt salen de settings. El import es
    diferido para no cargar Presidio/spaCy al arrancar el proceso.
    """
    from app.ingestion.pii import (
        ConsistentPseudonymizer,
        PostgresMappingStore,
        build_analyzer,
    )

    settings = get_settings()
    return ConsistentPseudonymizer(
        analyzer=build_analyzer(),
        mapping_store=PostgresMappingStore(session),
        salt=settings.pseudonym_hash_salt,
        faker_locale=settings.pseudonym_faker_locale,
        language="es",
    )


def get_text_pseudonymizer(session: Session) -> TextPseudonymizer:
    """Adaptador ``str → str`` sobre :func:`build_pseudonymizer`.

    El núcleo de ingesta sólo conoce el puerto ``TextPseudonymizer``; aquí se
    encapsula el tipo de resultado de Presidio. Import diferido de Presidio para
    no cargarlo salvo que una fuente con PII vaya a ingerirse.
    """
    pseudonymizer = build_pseudonymizer(session)

    def _pseudonymize(text: str) -> str:
        return pseudonymizer.pseudonymize(text).pseudonymized_text

    return _pseudonymize


# --- Sesión 7 — pipeline de embeddings ----------------------------------------


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    """Chunker estructural (sin estado) del pipeline de embeddings."""
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder | None:
    """Embedder de OpenAI. ``None`` si no hay embeddings configurados.

    La construcción del cliente (clave, ``base_url`` y modelo) vive en
    ``OpenAIEmbedder.from_settings``: API y CLI comparten una única fuente de
    verdad. Sigue el patrón de degradación de ``get_semantic_cache`` y el router
    traduce el ``None`` a un 500 con mensaje genérico.

    Ojo: NO reutilizamos ``get_openai_client()``, porque ese cliente está
    condicionado a la moderación (``has_moderation``) y aquí lo necesitamos
    aunque la moderación esté apagada.
    """
    embedder = OpenAIEmbedder.from_settings(get_settings())
    if embedder is None:
        logger.warning("embedding.disabled", reason="no_embeddings_configured")
    return embedder


# --- Sesión 8 — persistencia pgvector + búsqueda semántica --------------------


@lru_cache
def get_chunk_store() -> ChunkStore:
    """Capa de acceso async, sin estado, sobre documents/chunks."""
    return ChunkStore(use_halfvec=get_settings().search_use_halfvec)


@lru_cache
def get_rag_ingest_service() -> RagIngestService | None:
    """Orquestación chunk → embed → persist.

    ``None`` sin embeddings configurados (mismo contrato que ``get_embedder``);
    el router lo traduce a un 500 con mensaje genérico.
    """
    embedder = get_embedder()
    if embedder is None:
        return None
    return RagIngestService(
        chunker=get_chunker(),
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


@lru_cache
def get_semantic_retriever() -> SemanticRetriever | None:
    """Contraparte de consulta del servicio de ingesta. Mismo contrato ``None``."""
    embedder = get_embedder()
    if embedder is None:
        return None
    return SemanticRetriever(
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
        reranker=get_reranker(),
    )


# --- Sesión 12 — capa agéntica de estimación -----------------------------------


@lru_cache
def get_async_openai_client() -> AsyncOpenAI | None:
    """Cliente async para la Responses API del agente. ``None`` sin clave.

    Excepción deliberada a ``LLMWrapper``: el ejercicio conduce a mano el bucle de
    function calling (``responses.create`` + ``previous_response_id``), y eso
    exige el cliente crudo. Mismo contrato de degradación que ``get_embedder``:
    el router traduce el ``None`` a un 500 con mensaje genérico.
    """
    settings = get_settings()
    key = reveal_secret(settings.open_ai_key)
    if not key:
        logger.warning("agent.client.disabled", reason="no_openai_key_configured")
        return None
    return AsyncOpenAI(api_key=key)


def get_budget_search_backend() -> BudgetSearchBackend | None:
    """Backend de ``search_budgets`` con los valores por defecto de ``Settings``.

    Envoltura de :func:`make_budget_search_backend` sin overrides: la usa el
    agente autónomo. El wizard agéntico prefiere la factoría para ajustar
    ``top_k`` / ``distance_threshold`` por ejecución (perfil de agente).
    """
    return make_budget_search_backend()


def make_budget_search_backend(
    *,
    top_k: int | None = None,
    distance_threshold: float | None = None,
) -> BudgetSearchBackend | None:
    """Factoría del backend de ``search_budgets``: envuelve el retrieval de S9-S10.

    Es el único punto del agente que toca el pipeline RAG. Se construye aquí
    (composition root) porque ``app/generation/agentic`` no puede importar a su
    hermano ``generation.rag``; el contrato de capas lo verifica
    ``tests/test_layering.py``.

    ``top_k`` / ``distance_threshold`` permiten afinarlo por ejecución (los knobs
    del perfil de agente). Cuando son ``None`` caen a ``AGENT_SEARCH_TOP_K`` /
    ``AGENT_SEARCH_DISTANCE_THRESHOLD``. ``None`` sin embeddings configurados; los
    toggles de recuperación se leen en cada llamada, así que un cambio en Ajustes
    afecta a la siguiente búsqueda sin reiniciar el proceso.
    """
    embedder = get_embedder()
    if embedder is None:
        return None

    settings = get_settings()
    store = get_chunk_store()
    session_factory = get_async_session_factory()
    reranker = get_reranker()

    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        runtime = get_runtime_retrieval_config()
        query_embedding = await asyncio.to_thread(embedder.embed_one, args.query)
        outcome = await retrieve(
            store=store,
            session_factory=session_factory,
            query_embedding=query_embedding,
            query_text=args.query,
            search_mode=runtime.effective_search_mode(),
            rerank=runtime.effective_rerank(),
            top_k=top_k if top_k is not None else settings.agent_search_top_k,
            distance_threshold=(
                distance_threshold
                if distance_threshold is not None
                else settings.agent_search_distance_threshold
            ),
            collection=Collection.BUDGET,
            sectors=args.filters.sectors if args.filters else None,
            chunk_types=["budget_component", "historical_task"],
            reranker=reranker,
        )
        return [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "budget_id": chunk.budget_id,
                "sector": chunk.sector,
                "chunk_type": chunk.chunk_type,
                "content_preview": " ".join(chunk.content.split())[:200],
                "estimated_hours": chunk.estimated_hours,
                "distance": round(chunk.distance, 4),
            }
            for chunk in outcome.chunks
        ]

    return backend


# --- Sesión 7 (directo) — una factoría por estrategia de chunking --------------


@lru_cache
def get_raw_openai_client() -> OpenAI | None:
    """Cliente OpenAI sin la condición de moderación.

    ``get_openai_client`` devuelve ``None`` si la moderación está desactivada;
    las estrategias de chunking y el comparador necesitan el cliente aunque sea
    así.
    """
    settings = get_settings()
    key = reveal_secret(settings.open_ai_key)
    if not key:
        return None
    return OpenAI(api_key=key)


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic | None:
    """Cliente Anthropic perezoso. ``None`` si no hay API key configurada."""
    settings = get_settings()
    key = reveal_secret(settings.anthropic_api_key)
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


@lru_cache
def get_fixed_size_chunker() -> FixedSizeChunker:
    return FixedSizeChunker()


@lru_cache
def get_recursive_chunker() -> RecursiveChunker:
    return RecursiveChunker()


@lru_cache
def get_sentence_window_chunker() -> SentenceWindowChunker:
    return SentenceWindowChunker()


@lru_cache
def get_hierarchical_chunker() -> HierarchicalChunker:
    return HierarchicalChunker()


@lru_cache
def get_semantic_chunker() -> SemanticChunker:
    settings = get_settings()
    key = reveal_secret(settings.embedding_api_key) or reveal_secret(settings.open_ai_key)
    return SemanticChunker(api_key=key, model=settings.embedding_model)


# Sin ``lru_cache``: se reconstruyen por request de comparación para que un
# override de modelo en runtime aplique en la siguiente llamada.
def get_propositional_chunker() -> PropositionalChunker:
    client = get_raw_openai_client()
    if client is None:
        raise RuntimeError("PropositionalChunker requiere OPEN_AI_KEY.")
    settings = get_settings()
    return PropositionalChunker(client=client, model=settings.propositional_chunker_model)


def get_contextual_retrieval_chunker() -> ContextualRetrievalChunker:
    client = get_anthropic_client()
    if client is None:
        raise RuntimeError("ContextualRetrievalChunker requiere ANTHROPIC_API_KEY.")
    settings = get_settings()
    return ContextualRetrievalChunker(client=client, model=settings.contextual_chunker_model)


CHUNKER_FACTORIES: dict[str, Callable[[], Chunker]] = {
    "structural": get_chunker,
    "fixed_size": get_fixed_size_chunker,
    "recursive": get_recursive_chunker,
    "sentence_window": get_sentence_window_chunker,
    "semantic": get_semantic_chunker,
    "propositional": get_propositional_chunker,
    "contextual_retrieval": get_contextual_retrieval_chunker,
    "hierarchical": get_hierarchical_chunker,
}
ALL_STRATEGIES = list(CHUNKER_FACTORIES)


def build_chunkers(names: list[str]) -> list[Chunker]:
    """Instancia las estrategias pedidas por nombre.

    ``KeyError`` para una estrategia desconocida y ``RuntimeError`` si falta la
    API key de una estrategia de pago; el router los traduce a 400/500.
    """
    chunkers: list[Chunker] = []
    for name in names:
        factory = CHUNKER_FACTORIES.get(name)
        if factory is None:
            raise KeyError(name)
        chunkers.append(factory())
    return chunkers


# --- Sesión 9 — RAG end-to-end ------------------------------------------------


@lru_cache
def get_idempotency_store() -> IdempotencyStore:
    """Caché de idempotencia de estimaciones (Redis con fallback en memoria)."""
    from app.generation.rag.idempotency import IdempotencyStore

    return IdempotencyStore.from_settings(get_settings())


@lru_cache
def get_token_encoder() -> Encoding:
    """Encoder de tokens usado por el presupuesto de contexto.

    ``cl100k_base`` es el tokenizador de la familia GPT-4/GPT-5, alineado con
    ``text-embedding-3-small`` y los modelos de generación configurados.
    """
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")
