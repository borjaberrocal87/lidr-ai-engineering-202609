from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

APP_NAME = "Estimador de Software CAG"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


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
    llm_model: str = DEFAULT_MODELS["openai"]
    temperature: float = 0.2

    open_ai_key: str = ""
    anthropic_api_key: str = ""

    custom_llm_base_url: str = ""
    custom_llm_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
