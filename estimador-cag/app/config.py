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
    temperature: float = DEFAULT_TEMPERATURE

    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 2048
    transcription_min_length: int = 10
    transcription_max_length: int = 50_000

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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
