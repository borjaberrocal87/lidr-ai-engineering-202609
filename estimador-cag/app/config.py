from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

APP_NAME = "Estimador de Software CAG"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}
DEFAULT_TEMPERATURE = 0.2


def reveal_secret(value: SecretStr | str | None) -> str | None:
    """Devuelve el valor en claro de un secreto, o `None` si está vacío.

    Acepta tanto `SecretStr` (producción) como `str` (tests) para no acoplar
    los dobles a la representación del secreto.
    """
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "DEBUG"
    log_format: Literal["console", "json"] = "console"

    llm_provider: Literal["openai", "anthropic", "custom"] = "openai"
    llm_model: str = ""
    llm_fallback_model: str = ""
    llm_routing_mode: Literal["fallback", "balanced"] = "fallback"
    temperature: float = DEFAULT_TEMPERATURE

    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 2048
    description_min_length: int = 20
    description_max_length: int = 50_000

    cache_backend: Literal["memory", "redis", "none"] = "memory"
    cache_ttl: int = 86_400
    redis_url: str = "redis://localhost:6379"

    # Sesión 4 — salida estructurada (JSON schema + validación + reintento).
    structured_max_retries: int = 2

    # Sesión 4 — guardrails de entrada/salida.
    guardrails_enabled: bool = True
    guardrail_moderation_enabled: bool = True

    # Sesión 4 — caché semántica (embeddings + Redis Stack / RediSearch).
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.85
    semantic_cache_ttl: int = 86_400
    # Si true, se registran los aciertos potenciales pero no se sirven.
    semantic_cache_log_only: bool = False
    embedding_model: str = "text-embedding-3-small"
    # Endpoint OpenAI-compatible para embeddings (vacío = usar OpenAI estándar).
    embedding_base_url: str = ""
    embedding_api_key: SecretStr | None = None

    # Sesión 5 — memoria conversacional y adjuntos.
    # MAX_CONVERSATION_TURNS cuenta pares user+assistant. El system prompt es
    # invariante y se regenera cada turno, fuera de la ventana.
    max_conversation_turns: int = 6
    # Tope por adjunto extraído (caracteres), para proteger el prompt.
    max_attachment_chars: int = 60_000
    # Vacío => usa el modelo principal del servicio para extraer metadata.
    metadata_extractor_model: str = ""
    # Versión del prompt conversacional. Vacío => default del loader ("v3" tras
    # la sesión 5 del directo).
    conversational_prompt_version: str = ""

    # Sesión 5 (directo) — patrón Actor-Critic-Boss.
    # Vacío => usa el modelo principal del servicio como Critic.
    critic_model: str = ""
    boss_max_iterations: int = 2

    # Sesión 5 (directo) — compresión de memoria con anclas.
    # Vacío => usa el modelo principal del servicio para resumir/decidir anclas.
    compression_model: str = ""
    anchor_detection_mode: Literal["heuristic", "llm"] = "heuristic"

    # Sesión 7 (directo) — modelos de las estrategias de chunking con LLM.
    propositional_chunker_model: str = "gpt-4o-mini"
    contextual_chunker_model: str = "claude-sonnet-4-5"

    # Catálogo de modelos seleccionables en caliente desde Ajustes. Se filtra
    # por las API keys realmente configuradas.
    available_models: list[str] = [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "deepseek-v4-flash",
    ]

    # Sesión 6 — ingesta de datos, persistencia y PII.
    # Postgres (imagen pgvector) para el tracking de jobs y el mapping de PII.
    # En docker-compose se sobreescribe a estimator-postgres:5432.
    database_url: str = "postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
    # Catálogo YAML de fuentes (se carga una vez al arrancar). Relativo a BASE_DIR
    # si no es absoluto.
    catalog_path: Path = Path("data/catalog/catalog.yaml")
    # Raíz donde se resuelven los `location` del catálogo.
    ingestion_data_root: Path = Path("data/seed")
    # Modelo spaCy que usa Presidio para detectar PERSON/LOC en español.
    presidio_spacy_model: str = "es_core_news_md"
    # Locale de Faker para generar pseudónimos consistentes por tipo de entidad.
    pseudonym_faker_locale: str = "es_ES"
    # Sal manual del HMAC que indexa el mapping de PII. Rotarla invalida los
    # mappings previos (palanca GDPR art. 17). Cambiar en producción.
    pseudonym_hash_salt: str = "change-me-in-prod"
    # Si está activo, la ingesta pseudonimiza el texto de las fuentes con PII
    # (``sensitivity.has_pii``) antes de marcarlas como completadas. Requiere el
    # modelo spaCy de Presidio; sólo se carga cuando hace falta.
    ingestion_pseudonymize_pii: bool = True

    open_ai_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    custom_llm_base_url: str = ""
    custom_llm_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _resolve_model(self) -> "Settings":
        """Deriva el modelo por defecto del proveedor activo.

        Así no se puede arrancar con una pareja proveedor/modelo incoherente
        (p. ej. `openai` con un modelo de Anthropic) por olvidar `LLM_MODEL`.
        El proveedor `custom` no tiene default: exige modelo explícito.
        """
        if self.llm_model:
            return self
        default = DEFAULT_MODELS.get(self.llm_provider)
        if default is None:
            raise ValueError(f"LLM_MODEL es obligatorio para el proveedor {self.llm_provider!r}.")
        self.llm_model = default
        return self

    @property
    def is_configured(self) -> bool:
        """Indica si el proveedor activo tiene todo lo necesario para llamar."""
        if self.llm_provider == "openai":
            return bool(reveal_secret(self.open_ai_key))
        if self.llm_provider == "anthropic":
            return bool(reveal_secret(self.anthropic_api_key))
        if self.llm_provider == "custom":
            return bool(reveal_secret(self.custom_llm_api_key)) and bool(self.custom_llm_base_url)
        return False

    @property
    def has_moderation(self) -> bool:
        """La moderación usa la API de OpenAI; sin clave propia se desactiva."""
        return self.guardrail_moderation_enabled and bool(reveal_secret(self.open_ai_key))

    @property
    def has_embeddings(self) -> bool:
        """Hay embeddings si se configura un endpoint propio o si hay clave OpenAI."""
        if self.embedding_base_url and reveal_secret(self.embedding_api_key):
            return True
        return bool(reveal_secret(self.open_ai_key))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
