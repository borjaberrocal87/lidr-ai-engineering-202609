import pytest
from starlette.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def transcription() -> str:
    return (
        "En la reunión con el equipo de marketing, el cliente explicó que "
        "necesita una landing page con formulario de contacto, integración con "
        "su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG."
    )
