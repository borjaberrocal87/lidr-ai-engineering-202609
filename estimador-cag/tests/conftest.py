from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Evita que la caché de `get_settings` arrastre estado entre tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    # `with` ejecuta el lifespan de la app: si algún día se añade uno, los tests
    # lo respetan en vez de saltárselo silenciosamente.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def transcription() -> str:
    return (
        "En la reunión con el equipo de marketing, el cliente explicó que "
        "necesita una landing page con formulario de contacto, integración con "
        "su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG."
    )
