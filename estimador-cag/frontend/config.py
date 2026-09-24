"""Configuración del frontend.

Independiente de `app.config`: el frontend no lee claves LLM ni el `.env` del
backend, solo la URL donde está la API. Se lee del entorno para poder fijarla
en Docker/Compose sin tocar código.
"""

import os

import httpx

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_LOG_LEVEL = "INFO"


def get_api_base_url() -> str:
    """URL base de la API (sin barra final)."""
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()


def build_http_timeout() -> httpx.Timeout:
    """Timeout para las llamadas a la API.

    Sin `read_timeout`: las generaciones en streaming pueden tardar minutos y
    cortar por lectura abortaría estimaciones válidas. El resto queda acotado.
    """
    connect = float(os.getenv("API_CONNECT_TIMEOUT_SECONDS", str(DEFAULT_CONNECT_TIMEOUT_SECONDS)))
    return httpx.Timeout(connect=connect, read=None, write=30.0, pool=connect)
