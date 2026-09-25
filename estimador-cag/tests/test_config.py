import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_arranca_sin_credenciales() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_model="",
        open_ai_key=None,
        anthropic_api_key=None,
        custom_llm_api_key=None,
    )

    assert settings.llm_model == "gpt-4o-mini"
    assert settings.is_configured is False


def test_modelo_se_deriva_del_proveedor() -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", llm_model="")

    assert settings.llm_model == "claude-haiku-4-5"


def test_modelo_explicito_se_respeta() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", llm_model="gpt-4o")

    assert settings.llm_model == "gpt-4o"


def test_routing_mode_por_defecto_es_fallback() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_routing_mode == "fallback"


def test_custom_sin_modelo_es_error() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="custom", llm_model="")


def test_is_configured_custom_requiere_url_y_key() -> None:
    sin_url = Settings(
        _env_file=None,
        llm_provider="custom",
        llm_model="llama3.1:8b",
        custom_llm_base_url="",
        custom_llm_api_key="k",
    )
    assert sin_url.is_configured is False

    completo = Settings(
        _env_file=None,
        llm_provider="custom",
        llm_model="llama3.1:8b",
        custom_llm_base_url="http://localhost:11434/v1",
        custom_llm_api_key="k",
    )
    assert completo.is_configured is True
