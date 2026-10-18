# Spec — Proyecto 2: Pipeline de embeddings (chunking estructural + `text-embedding-3-small`)

## Objetivo

La Sesión 06 cerró con los presupuestos históricos limpios, normalizados y guardados en formato JSON estructurado. En esta sesión damos el primer paso hacia la búsqueda semántica: **convertir esos JSON en vectores**.

El objetivo es construir dentro del servicio IA un pipeline funcional mínimo que reciba presupuestos históricos en formato JSON, los parta en chunks respetando la estructura del documento, genere embeddings con OpenAI y devuelva los vectores listos para su uso posterior. El ejercicio se cierra con un pequeño script de línea de comandos que verifica que los embeddings funcionan razonablemente, comparando la similitud entre tres parejas de textos.

Al terminar este ejercicio tendrás:

- Un **chunker estructural** específico para presupuestos JSON: un componente del presupuesto = un chunk, con cabecera de contexto del presupuesto padre y conteo de tokens con `tiktoken`.
- Un **embedder** que invoca `text-embedding-3-small` en batches, reintenta ante `RateLimitError` y calcula el coste estimado.
- El endpoint **`POST /api/v1/embeddings/ingest`**, que recibe presupuestos y devuelve sus chunks vectorizados más las estadísticas agregadas.
- El script CLI **`scripts/compare.py`**, que toma dos textos y devuelve su similitud coseno (calculada a mano, solo biblioteca estándar).
- El deliverable cualitativo: **`app/embedding_pipeline/SANITY_CHECK.md`**, con las tres parejas de validación y el comentario de resultados.

La sesión en vivo partirá de este pipeline para introducir otras estrategias de chunking y compararlas.

> **Reconciliación con el repo.** El punto de partida es `main` con las sesiones 01–06 cerradas: wrapper LLM con fallback, cachés (exacta y semántica), sesiones conversacionales con memoria, baseline de estrés del CAG e ingesta con catálogo, PII y Postgres. **Nada se reescribe**: el módulo `app/embedding_pipeline/` es aditivo y la interfaz visual y el cliente de negocio no se tocan. La caché semántica (sesión 4) ya usa embeddings a través de `redisvl.OpenAITextVectorizer`, pero como vectorizador opaco dentro de Redis; aquí damos el **primer embedder explícito**, reutilizable y con métricas propias. Los vectores **no se persisten todavía**: se generan en memoria y se devuelven por HTTP. La persistencia entra en la Sesión 08, cuando introduzcamos PostgreSQL + pgvector.

## Contexto del proyecto

Piezas relevantes que se reutilizan o extienden (todas ya en `main`):

- `app/config.py` — ya define `embedding_model` (default `text-embedding-3-small`), `embedding_base_url` y `embedding_api_key` (endpoint OpenAI-compatible alternativo), y la propiedad `has_embeddings`, que decide si hay embeddings configurados. El pipeline los **reutiliza**: no hay variables de entorno nuevas.
- `app/dependencies.py` — `get_semantic_cache()` es el patrón de dependencia opcional que **degrada a `None`** en vez de romper el arranque. Las fábricas `get_chunker()` y `get_embedder()` del pipeline se añaden con ese mismo estilo (`lru_cache`, degradación silenciosa y aviso en logs).
- `app/cache/semantic.py` — primer uso de embeddings del repo (redisvl, `text-embedding-3-small`, 1536 dims). No se toca, pero fija la referencia de dimensión esperada.
- `app/logging_config.py` + structlog — convención de eventos dotted con kwargs de la sesión 03 (`llm.call.end`, `embedding.batch.end`, …).
- `app/main.py` — los routers se registran con prefijo `/api/v1` (`estimations`, `sessions`, `ingestion`). El router de embeddings se suma ahí: la ruta completa queda `/api/v1/embeddings/ingest`.
- `Dockerfile` / `docker-compose.yml` — desde la sesión 06 la imagen copia `data/` y `scripts/`, y el servicio se llama `api`. Ni el endpoint ni `compare.py` requieren tocar Docker.
- `pyproject.toml` — `openai` ya está (`>=3.13.0`); solo falta `tiktoken`. `ruff` y `mypy` en modo estricto marcan el estilo (línea 100, docstrings en español, identificadores en inglés).
- `tests/test_project_structure.py` — valida rutas y los paths de OpenAPI; el endpoint nuevo no rompe nada, pero conviene añadirlo a la lista si escribes tests.

## Requisitos para el ejercicio

- Proyecto de la sesión 06 en verde (`uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy app frontend`).
- Dependencia nueva: **`tiktoken`** (para contar tokens). `openai` ya está desde las primeras sesiones:

```bash
uv add tiktoken
```

> El `Dockerfile` construye con `uv sync --frozen`, así que el commit debe incluir el `uv.lock` actualizado.

- API key de OpenAI configurada en `.env` (`OPEN_AI_KEY`), o un endpoint compatible con embeddings (`EMBEDDING_BASE_URL` + `EMBEDDING_API_KEY`). Si no hay ninguno de los dos, el endpoint responde 500 (degradación explícita, no silenciosa).
- Docker y Docker Compose funcionando.
- Python 3.12 disponible localmente (opcional; solo necesario para ejecutar `compare.py` fuera del contenedor con `uv run`).

## Lo que entra y lo que no entra

**Entra:**

- Un chunker estructural específico para presupuestos JSON (un componente del presupuesto = un chunk).
- Un embedder que invoca `text-embedding-3-small` de OpenAI.
- Un endpoint FastAPI `POST /api/v1/embeddings/ingest` que recibe presupuestos y devuelve sus chunks vectorizados.
- Un script CLI `scripts/compare.py` que toma dos textos y devuelve su similitud coseno.
- Validación con tres parejas de textos de prueba.

**No entra** (lo tocaremos en la sesión en vivo o en la Sesión 08):

- Cualquier otra estrategia de chunking: recursive, semantic, hierarchical, late chunking, contextual retrieval, etc. Si te ves implementando alguna de estas, has salido del scope.
- Comparativa entre varios modelos de embeddings.
- Cualquier técnica de enriquecimiento del chunk con contexto generado por LLM.
- Persistencia en base de datos vectorial (pgvector es la Sesión 08).
- Búsqueda semántica (retrieval).
- Métricas formales de calidad de retrieval (recall@k, NDCG, etc.).
- Cambios en la interfaz visual o en el cliente de negocio.

## Datos para el ejercicio

Se proporciona en el repositorio de datos del programa un fichero `data/budgets_sample.json` con **15 presupuestos históricos ya normalizados**, cubriendo sectores variados (fintech, e-commerce, healthcare, industrial) y stacks tecnológicos diferentes. Cópialo a la carpeta `data/` del proyecto. Cada presupuesto sigue este esquema simplificado:

```json
{
  "budget_id": "BUD-2024-014",
  "client_metadata": {
    "name": "FintechCorp",
    "sector": "finance",
    "country": "ES"
  },
  "project_summary": "Mobile banking API with OAuth 2.0 authentication and PSD2 compliance",
  "main_technology": "ruby_on_rails",
  "year": 2024,
  "total_estimated_hours": 480,
  "components": [
    {
      "component_id": "AUTH-001",
      "name": "OAuth 2.0 authentication backend",
      "description": "Implementation of OAuth 2.0 flows (authorization code, refresh token) with JWT-based session management, multi-tenant token isolation, and rate limiting per client.",
      "tech_stack": ["ruby_on_rails", "postgresql", "redis"],
      "estimated_hours": 120,
      "complexity": "high",
      "dependencies": []
    }
  ]
}
```

El fichero es un **array JSON** con 15 objetos `Budget`. Ojo: el endpoint espera un objeto `{"budgets": [...]}` — al invocarlo desde CLI, envuelve el array.

Si traes tu propio dataset desde la Sesión 06, puedes usarlo siempre que respete este esquema. En caso contrario, se trabaja con el sample.

> El `Dockerfile` ya copia `data/` dentro de la imagen (lo añadió la sesión 06), así que el mismo fichero sirve para invocar el endpoint desde el host, desde `/docs` y desde el contenedor sin configuración extra.

## ✍ Ejercicio

### Paso 1 — Estructura del módulo y dependencias

Dentro del servicio IA (la raíz del repo), crea el siguiente árbol nuevo:

```
estimador-cag/
├── app/
│   └── embedding_pipeline/
│       ├── __init__.py
│       ├── chunker.py
│       ├── embedder.py
│       ├── schemas.py
│       ├── router.py
│       └── SANITY_CHECK.md        # se añade en el Paso 7
├── scripts/
│   └── compare.py
└── data/
    └── budgets_sample.json        # sample provisto por el programa
```

`__init__.py` es un docstring de módulo que resume la pieza y fija su contexto en el proyecto (los vectores son efímeros; la persistencia es Sesión 08).

Añade la dependencia nueva:

```bash
uv add tiktoken
```

No añadas `numpy`, `scikit-learn` ni ninguna otra librería pesada. La similitud coseno se calcula a mano con la biblioteca estándar. (`numpy` ya está en el proyecto por otras piezas; no se usa aquí.)

Reconstruye el contenedor para que la imagen lleve la dependencia y el código:

```bash
docker compose build api
```

### Paso 2 — Modelos de datos con Pydantic

En `app/embedding_pipeline/schemas.py` define los modelos. Trabajamos con Pydantic v2 y respetamos el patrón establecido en sesiones anteriores: docstrings en español, identificadores en inglés, `Field(description=…)` en los campos del contrato y enums cerrados para los universos conocidos.

```python
"""Modelos del pipeline de embeddings.

Entrada: presupuestos históricos normalizados (un presupuesto con lista de
componentes). Salida: chunks listos para embeber y, una vez vectorizados, los
vectores con sus estadísticas agregadas.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Sector(StrEnum):
    """Sector de negocio del cliente (universo cerrado del sample)."""

    FINANCE = "finance"
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    INDUSTRIAL = "industrial"


class Complexity(StrEnum):
    """Bucket de complejidad del componente."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClientMetadata(BaseModel):
    """Cliente del presupuesto. Viaja como contexto filtrable; no se embebe."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Nombre de la empresa cliente.")
    sector: Sector = Field(..., description="Sector de negocio del cliente.")
    country: str = Field(..., description="Código de país, p. ej. 'ES'.")


class BudgetComponent(BaseModel):
    """Una línea de un presupuesto histórico."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(
        ..., description="Id estable dentro del presupuesto, p. ej. 'AUTH-001'."
    )
    name: str = Field(..., description="Nombre corto y legible del componente.")
    description: str = Field(..., description="Descripción detallada del trabajo.")
    tech_stack: list[str] = Field(
        default_factory=list, description="Tecnologías implicadas en el componente."
    )
    estimated_hours: int = Field(..., ge=0, description="Horas estimadas para el componente.")
    complexity: Complexity = Field(..., description="Bucket de complejidad.")
    dependencies: list[str] = Field(
        default_factory=list, description="component_id de los que depende este componente."
    )


class Budget(BaseModel):
    """Presupuesto histórico completo."""

    model_config = ConfigDict(extra="forbid")

    budget_id: str = Field(..., description="Id estable del presupuesto, p. ej. 'BUD-2024-014'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(..., description="Resumen en una línea del proyecto.")
    main_technology: str = Field(..., description="Tecnología principal del proyecto.")
    year: int = Field(..., ge=2000, le=2100, description="Año en que se produjo el presupuesto.")
    total_estimated_hours: int = Field(..., ge=0, description="Suma de horas de los componentes.")
    components: list[BudgetComponent] = Field(
        ..., min_length=1, description="Líneas del presupuesto."
    )


class Chunk(BaseModel):
    """Fragmento listo para embeber.

    ``text`` es lo que se manda a la API de embeddings; ``metadata`` lleva los
    campos filtrables que viajan con el chunk pero NO se embeben.
    """

    chunk_id: str = Field(..., description="Id trazable con formato '{budget_id}::{component_id}'.")
    text: str = Field(
        ..., description="Texto embebible: contexto del padre + detalle del componente."
    )
    metadata: dict[str, str | int] = Field(
        default_factory=dict, description="Campos filtrables que no se embeben."
    )
    token_count: int = Field(..., ge=0, description="Tokens de ``text`` según tiktoken.")


class EmbeddedChunk(Chunk):
    """Un :class:`Chunk` con su vector de embedding adjunto."""

    embedding: list[float] = Field(
        ..., description="Vector denso (1536 dims con text-embedding-3-small)."
    )


class IngestRequest(BaseModel):
    """Payload de entrada de ``POST /api/v1/embeddings/ingest``."""

    model_config = ConfigDict(extra="forbid")

    budgets: list[Budget] = Field(
        ..., min_length=1, description="Presupuestos a chunquear y embeber."
    )


class IngestStats(BaseModel):
    """Contadores agregados que acompañan a una respuesta de ingesta."""

    total_budgets: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0.0)


class IngestResponse(BaseModel):
    """Respuesta de ``POST /api/v1/embeddings/ingest``."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats
```

> **`StrEnum`, no `Literal`.** El enunciado sugería un `Literal` para el sector; en este repo los universos cerrados de contrato son `StrEnum` (como `ProjectType` y `DetailLevel` en `app/schemas/estimations.py`). Misma garantía de validación y, además, el tipo es usable por el resto del código (`.value`, serialización a string en JSON).

> **`extra="forbid"`.** Los modelos de entrada son un contrato estricto, como `IngestionRunRequest`: un campo desconocido en el JSON del alumno es un 422 que localiza el error, no un campo que se ignora en silencio y nunca llega al filtro ni al embedding.

> **`metadata` tipado como `dict[str, str | int]`.** `mypy` en modo estricto no admite `dict` a secas (genérico sin parámetros). Los valores que viajan son textos y enteros (`year`, `estimated_hours`); el tipo union refleja exactamente eso.

### Paso 3 — Implementar el chunker estructural

En `app/embedding_pipeline/chunker.py` implementa una clase `JSONStructuralChunker` con un único método público:

```python
def chunk(self, budgets: list[Budget]) -> list[Chunk]: ...
```

Reglas de la estrategia:

1. **Granularidad:** un componente del presupuesto = un chunk. Iteras los presupuestos y, por cada `BudgetComponent`, produces un `Chunk`.
2. **Contenido del chunk (`text`):** una representación textual legible que combine el contexto del presupuesto padre con los detalles del componente:

```
[Project: {project_summary}]
[Client sector: {sector} | Year: {year} | Main tech: {main_technology}]

Component: {component.name}
Description: {component.description}
Tech stack: {", ".join(component.tech_stack)}
Complexity: {component.complexity}
Estimated hours: {component.estimated_hours}
```

Este es el texto que se embebe. La decisión de **prepender el contexto del padre es deliberada**: sin ella, un componente "Authentication backend" perdería la pista de a qué cliente y sector pertenece. Esta idea conecta con el concepto de *contextual chunk headers* del que hablamos en el artículo 3 del material asíncrono.

3. **Metadata del chunk:** campos filtrables que **no se embeden** pero que viajan con el chunk para futuras consultas. Incluye al menos: `budget_id`, `component_id`, `client_sector`, `main_technology`, `year`, `complexity`, `estimated_hours`.
4. **`chunk_id`:** formato `{budget_id}::{component_id}` para que sea trazable. Por ejemplo: `BUD-2024-014::AUTH-001`.
5. **`token_count`:** cuenta los tokens del campo `text` usando `tiktoken.encoding_for_model("text-embedding-3-small")`. Esto permite detectar chunks anormalmente grandes antes de mandarlos a la API.

```python
"""Chunker estructural para presupuestos JSON.

Estrategia: confiar en la estructura del documento. Un ``BudgetComponent`` se
convierte en exactamente un ``Chunk``. Sin overlap y sin splitting a tamaño fijo
de las descripciones largas: si una descripción es anormalmente grande, es un
dato que ``token_count`` surfacea y queremos discutir, no tapar.

A cada chunk se le antepone el contexto del presupuesto padre (la *cabecera de
contexto*): sin ella, un componente como "Authentication backend" perdería la
pista de a qué cliente y sector pertenece.
"""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

# Tokenizador contra el que contamos. Cargar el encoding es relativamente caro;
# se resuelve una vez a nivel de módulo y se reutiliza.
TOKENIZER_MODEL = "text-embedding-3-small"
_ENCODING = tiktoken.encoding_for_model(TOKENIZER_MODEL)

# Plantillas del texto embebible. La cabecera aporta el contexto del padre; el
# cuerpo, el detalle del componente. Se mantienen separadas para que el test (y
# la lectura) puedan razonar sobre cada mitad por su cuenta.
_HEADER_TEMPLATE = (
    "[Project: {project_summary}]\n"
    "[Client sector: {sector} | Year: {year} | Main tech: {main_technology}]"
)
_BODY_TEMPLATE = (
    "Component: {name}\n"
    "Description: {description}\n"
    "Tech stack: {tech_stack}\n"
    "Complexity: {complexity}\n"
    "Estimated hours: {estimated_hours}"
)


def count_tokens(text: str) -> int:
    """Tokens de ``text`` con el tokenizador del modelo del ejercicio."""
    return len(_ENCODING.encode(text))


class JSONStructuralChunker:
    """Convierte presupuestos en un chunk por componente."""

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        return [
            self._chunk_component(budget, component)
            for budget in budgets
            for component in budget.components
        ]

    def _chunk_component(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = self._render_text(budget, component)
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata=self._build_metadata(budget, component),
            token_count=count_tokens(text),
        )

    @staticmethod
    def _render_text(budget: Budget, component: BudgetComponent) -> str:
        """Cabecera de contexto del padre + detalle del componente; esto se embebe."""
        header = _HEADER_TEMPLATE.format(
            project_summary=budget.project_summary,
            sector=budget.client_metadata.sector.value,
            year=budget.year,
            main_technology=budget.main_technology,
        )
        body = _BODY_TEMPLATE.format(
            name=component.name,
            description=component.description,
            tech_stack=", ".join(component.tech_stack),
            complexity=component.complexity.value,
            estimated_hours=component.estimated_hours,
        )
        return f"{header}\n\n{body}"

    @staticmethod
    def _build_metadata(budget: Budget, component: BudgetComponent) -> dict[str, str | int]:
        """Campos filtrables que viajan con el chunk pero no se embeben."""
        return {
            "budget_id": budget.budget_id,
            "component_id": component.component_id,
            "client_sector": budget.client_metadata.sector.value,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "complexity": component.complexity.value,
            "estimated_hours": component.estimated_hours,
        }
```

> **No implementes overlap ni fixed-size splitting.** Si una descripción es excepcionalmente larga, eso es un dato que querremos discutir en directo (el `token_count` lo hace visible). Confía en la estructura del JSON: un componente cabe como chunk individual.

> **Sobre el tokenizador.** El conteo asume el modelo del ejercicio. Si cambias `EMBEDDING_MODEL` por un modelo que `tiktoken` no conozca (p. ej. un endpoint custom con otro modelo), `encoding_for_model` protestará al importar el módulo. Es una consecuencia deliberada del scope: aquí contamos tokens contra el modelo del ejercicio.

### Paso 4 — Implementar el embedder

En `app/embedding_pipeline/embedder.py` implementa una clase `OpenAIEmbedder` con dos métodos públicos:

```python
def embed_one(self, text: str) -> list[float]: ...


def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...
```

Reglas:

1. Usa el modelo `text-embedding-3-small` con dimensión por defecto (**1536**). No fuerces dimensión menor en este ejercicio; la discusión sobre Matryoshka es del directo.
2. `embed_many` debe llamar a la API **en batches** (no una llamada por chunk), aprovechando que el endpoint `embeddings.create` acepta listas. Tamaño recomendado: **100 chunks por llamada**. Esto es importante para no martillear la API con peticiones serializadas.
3. Maneja errores básicos: `RateLimitError` con reintento exponencial simple (3 reintentos con esperas 1s, 2s, 4s); **propaga el resto de errores** hacia arriba.
4. Loguea con `structlog` cada batch procesado: número de chunks, número de tokens totales y latencia de la llamada. El logging sigue la convención que viene de la Sesión 03.
5. Calcula el **coste estimado**: el precio de `text-embedding-3-small` es **$0.02 por millón de tokens de entrada**. Trata el precio como una constante de módulo claramente etiquetada (cambia con el tiempo).

```python
"""Embedder de OpenAI para el pipeline de embeddings.

Envuelve ``text-embedding-3-small`` (1536 dims por defecto). ``embed_many``
agrupa los chunks con ``itertools.batched`` en una llamada por batch (el
endpoint de embeddings acepta listas), mucho más barato en latencia que una
petición por chunk.
"""

from __future__ import annotations

import time
from itertools import batched

import structlog
from openai import OpenAI, RateLimitError

from app.config import Settings, reveal_secret
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

# Modelo por defecto del ejercicio. Mantenemos las 1536 dims completas (sin
# recorte Matryoshka: eso es discusión del directo).
MODEL = "text-embedding-3-small"

# Chunks por llamada a la API. El endpoint acepta listas; batchear evita
# martillear la API con peticiones serializadas de un solo item.
BATCH_SIZE = 100

# Constante de precio — CAMBIA CON EL TIEMPO. A fecha de este ejercicio,
# text-embedding-3-small cuesta $0.02 por 1M de tokens de entrada.
PRICE_PER_MILLION_TOKENS_USD = 0.02

# Esperas (segundos) antes de cada reintento por rate limit: 3 reintentos.
_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


def estimate_cost_usd(total_tokens: int) -> float:
    """Coste en USD de embeber ``total_tokens`` tokens de entrada."""
    return total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


class OpenAIEmbedder:
    """Wrapper fino sobre ``client.embeddings.create`` con batching y reintentos."""

    def __init__(self, client: OpenAI, model: str = MODEL) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIEmbedder | None:
        """Construye el embedder desde la configuración del servicio.

        Devuelve ``None`` si no hay embeddings configurados (ni ``OPEN_AI_KEY``
        ni ``EMBEDDING_BASE_URL`` + ``EMBEDDING_API_KEY``). Única fábrica del
        cliente: la usan tanto las dependencias de FastAPI como el CLI.
        """
        if not settings.has_embeddings:
            return None
        api_key = reveal_secret(settings.embedding_api_key) or reveal_secret(settings.open_ai_key)
        client = OpenAI(api_key=api_key, base_url=settings.embedding_base_url or None)
        return cls(client=client, model=settings.embedding_model)

    def embed_one(self, text: str) -> list[float]:
        """Embebe un único texto. Lo usa el script CLI `compare.py`."""
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embebe todos los chunks manteniendo el orden, en batches."""
        embedded: list[EmbeddedChunk] = []
        for batch in batched(chunks, BATCH_SIZE):
            texts = [chunk.text for chunk in batch]
            batch_tokens = sum(chunk.token_count for chunk in batch)

            started_at = time.perf_counter()
            vectors = self._create(texts)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

            log.info(
                "embedding.batch.end",
                chunks=len(batch),
                tokens=batch_tokens,
                latency_ms=latency_ms,
                model=self._model,
            )

            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    def _create(self, texts: list[str]) -> list[list[float]]:
        """Llama a la API con reintentos ante rate limit.

        Se hacen 3 reintentos con esperas de 1s, 2s y 4s. El resto de errores
        (red, autenticación, respuestas no recuperables) se propagan.
        """
        retries = 0
        while True:
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in response.data]
            except RateLimitError:
                if retries >= len(_RETRY_DELAYS_SECONDS):
                    raise
                delay = _RETRY_DELAYS_SECONDS[retries]
                retries += 1
                log.warning("embedding.batch.rate_limited", retry=retries, retry_in_s=delay)
                time.sleep(delay)
```

> **Una única fábrica del cliente.** `OpenAIEmbedder.from_settings` traduce nuestras settings (`OPEN_AI_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `has_embeddings`) a un cliente configurado; la usan tanto `get_embedder()` como `compare.py`. El constructor sigue aceptando un `OpenAI` inyectado para que los tests usen un doble sin red y para que un endpoint OpenAI-compatible funcione sin tocar el embedder.

> **Por qué batches de 100.** El coste en tokens es el mismo, pero la latencia de red no: una llamada por chunk paga el round-trip 52 veces en lugar de una sola. La API impone un máximo de inputs por llamada; 100 es un valor conservador que cabe de sobra con este corpus y deja margen.

### Paso 5 — Endpoint FastAPI

En `app/embedding_pipeline/router.py` define el router (`prefix="/embeddings"`) con el endpoint `POST /ingest`; la ruta completa, una vez registrado, queda `/api/v1/embeddings/ingest`.

Contrato:

- **Input** (`IngestRequest`): lista de presupuestos en el body.
- **Output** (`IngestResponse`): lista de chunks vectorizados más las estadísticas agregadas.
- **Status codes:** 200 para éxito, 422 para validación Pydantic fallida (lo gestiona FastAPI), 500 para errores no controlados de la API de embeddings (con un mensaje genérico al cliente y el detalle en los logs).

El handler debe orquestar la secuencia: `chunker.chunk(budgets)` → `embedder.embed_many(chunks)` → ensamblar `IngestResponse`.

Primero, las fábricas de dependencias en `app/dependencies.py` (nueva sección al final, patrón `lru_cache` como el resto):

```python
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
```

Añade estas dos líneas al bloque de imports de `app/dependencies.py` (el resto —`lru_cache`, `OpenAI`, `reveal_secret`— ya está):

```python
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
```

> **Si añades tests**, incluye `get_chunker.cache_clear()` y `get_embedder.cache_clear()` en el fixture autouse de `tests/conftest.py` que resetea las fábricas `lru_cache` entre tests.

Ahora el router:

```python
"""Capa HTTP del pipeline de embeddings.

Router fino: orquesta ``chunker → embedder → respuesta`` y traduce los fallos a
códigos de estado. Aquí no vive lógica de negocio.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: Annotated[JSONStructuralChunker, Depends(get_chunker)],
    embedder: Annotated[OpenAIEmbedder | None, Depends(get_embedder)],
) -> IngestResponse:
    """Chunquea los presupuestos, embebe cada chunk y devuelve vectores + stats."""
    if embedder is None:
        # Sin clave/endpoint de embeddings configurados. Mensaje genérico al
        # cliente; el motivo real queda en los logs.
        log.error("embedding.ingest.failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron generar los embeddings.",
        )

    chunks = chunker.chunk(request.budgets)
    log.info(
        "embedding.ingest.received",
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
    )

    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:
        log.error(
            "embedding.ingest.failed",
            reason="provider_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron generar los embeddings.",
        ) from exc

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(total_tokens),
    )
    log.info("embedding.ingest.end", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
```

Registra el router en `app/main.py`:

```python
from app.embedding_pipeline.router import router as embeddings_router

# ...

app.include_router(estimations.router, prefix="/api/v1", tags=["estimations"])
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(embeddings_router, prefix="/api/v1")
```

La ruta completa queda **`POST /api/v1/embeddings/ingest`**. Verifica que aparece en `/docs` (Swagger UI de FastAPI) y que se puede invocar desde ahí mismo con el sample de datos. También desde CLI (recuerda envolver el array):

```bash
# httpie (envuelve el array en el campo "budgets")
http POST :8000/api/v1/embeddings/ingest budgets:=@data/budgets_sample.json

# curl equivalente
curl -s -X POST http://localhost:8000/api/v1/embeddings/ingest \
  -H 'Content-Type: application/json' \
  -d "{\"budgets\": $(cat data/budgets_sample.json)}" | python -m json.tool | head -40
```

Con el sample completo, la respuesta esperada ronda **15 presupuestos → 52 chunks → ~4.1k tokens → coste estimado ~$0.00008**. Son cifras orientativas: dependen de la composición exacta de los textos del sample.

### Paso 6 — Script CLI `compare.py`

En `scripts/compare.py` crea un script ejecutable que toma dos textos por línea de comandos, los embebe y devuelve la similitud coseno entre ambos.

Especificación:

```bash
python scripts/compare.py \
  --text-a "OAuth 2.0 authentication backend for fintech" \
  --text-b "JWT-based authorization service for banking app"
```

Salida esperada (formato exacto libre, esto es un ejemplo):

```
Text A: OAuth 2.0 authentication backend for fintech
Text B: JWT-based authorization service for banking app
Cosine similarity: 0.8421
```

El script **reutiliza la clase `OpenAIEmbedder`** que ya tienes. Implementa la similitud coseno a mano: producto escalar dividido por el producto de las normas. No importes `numpy` ni `scikit-learn` solo para esto.

```python
#!/usr/bin/env python3
"""Sanity check de embeddings — similitud coseno entre dos textos.

Embebe dos textos con ``text-embedding-3-small`` (vía ``OpenAIEmbedder``) e
imprime su similitud coseno. El coseno se calcula con la stdlib: ``math.sumprod``
para el producto escalar y ``math.hypot`` para las normas L2. Sin numpy.

Uso::

    # fuera del contenedor (con el .env cargado):
    uv run python scripts/compare.py \\
        --text-a "OAuth 2.0 authentication backend for fintech" \\
        --text-b "JWT-based authorization service for banking app"

    # dentro del contenedor:
    docker compose exec api python scripts/compare.py \\
        --text-a "..." --text-b "..."
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Hace importable ``app`` cuando el script se ejecuta directamente desde ``scripts/``.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Producto escalar dividido por el producto de las normas L2."""
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return math.sumprod(a, b) / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Similitud coseno entre dos textos embebidos.")
    parser.add_argument("--text-a", required=True, help="Primer texto.")
    parser.add_argument("--text-b", required=True, help="Segundo texto.")
    args = parser.parse_args()

    embedder = OpenAIEmbedder.from_settings(get_settings())
    if embedder is None:
        print(
            "ERROR: no hay embeddings configurados "
            "(OPEN_AI_KEY o EMBEDDING_BASE_URL + EMBEDDING_API_KEY en .env).",
            file=sys.stderr,
        )
        return 1

    similarity = cosine_similarity(
        embedder.embed_one(args.text_a),
        embedder.embed_one(args.text_b),
    )

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

El script debe poder ejecutarse de dos formas:

```bash
# Dentro del contenedor (los scripts ya viajan en la imagen desde la sesión 6):
docker compose exec api python scripts/compare.py \
  --text-a "OAuth 2.0 authentication backend for fintech" \
  --text-b "JWT-based authorization service for banking app"

# Fuera del contenedor, con el .env del repo cargado:
uv run python scripts/compare.py \
  --text-a "OAuth 2.0 authentication backend for fintech" \
  --text-b "JWT-based authorization service for banking app"
```

Documenta en el README **ambas formas** (ver entregable).

> El `Dockerfile` copia `scripts/` en la imagen (lo añadió la sesión 06), así que no hay que tocar Docker. Si editas `compare.py`, reconstruye (`docker compose build api`) para que el contenedor vea la versión nueva.

### Paso 7 — Validación con tres parejas

Ejecuta `compare.py` sobre exactamente estas tres parejas y guarda los resultados en `app/embedding_pipeline/SANITY_CHECK.md` (Markdown). Las parejas son:

**Pareja A — Textos semánticamente cercanos.** Se espera similitud alta (orientativamente > 0.6).

- Texto 1: `"OAuth 2.0 authentication backend with JWT tokens for fintech mobile app"`
- Texto 2: `"Authorization service using JSON Web Tokens for a banking application"`

**Pareja B — Textos no relacionados.** Se espera similitud baja (orientativamente < 0.4).

- Texto 1: `"OAuth 2.0 authentication backend with JWT tokens for fintech mobile app"`
- Texto 2: `"Database migration from MySQL to PostgreSQL with zero downtime"`

**Pareja C — Textos genéricos y ambiguos.** Aquí no hay expectativa fija; lo interesante es comentar el resultado obtenido.

- Texto 1: `"Backend services"`
- Texto 2: `"API development"`

En `SANITY_CHECK.md` incluye:

1. La similitud obtenida para cada pareja (los tres valores numéricos, en tabla).
2. Un **comentario breve** (3-5 líneas, en español o inglés, libre) sobre si los resultados encajan con la expectativa intuitiva y qué te llama la atención. Si alguna pareja da un valor sorprendente, déjalo registrado: será buen material de discusión para el directo.

Plantilla de partida:

```markdown
# Embedding sanity check

Modelo: `text-embedding-3-small` (1536 dims, dimensión por defecto).
Generado con `scripts/compare.py` (similitud coseno calculada a mano, solo stdlib).
Fecha: <YYYY-MM-DD>.

| Pareja | Texto A | Texto B | Coseno | Expectativa |
|--------|---------|---------|--------|-------------|
| A — cercanos | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Authorization service using JSON Web Tokens for a banking application" | **0.XXXX** | alta (orientativo > 0.6) |
| B — no relacionados | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Database migration from MySQL to PostgreSQL with zero downtime" | **0.XXXX** | baja (orientativo < 0.4) |
| C — genéricos | "Backend services" | "API development" | **0.XXXX** | sin expectativa fija |

## Comentario

<3-5 líneas: ¿coincide con la intuición? ¿alguna pareja sorprende? ¿por qué crees que pasa?>
```

> Este sanity check **no es una validación formal del modelo**: es un mínimo aceptable que demuestra que el pipeline funciona end-to-end y que los embeddings discriminan razonablemente entre textos cercanos y lejanos.

### Entregable

Una rama nueva en el repositorio: **`feature/pre-session-07`**. Esa rama debe contener:

1. El módulo `app/embedding_pipeline/` completo con `chunker.py`, `embedder.py`, `schemas.py`, `router.py` y `__init__.py`.
2. El script `scripts/compare.py` funcional.
3. El endpoint `POST /api/v1/embeddings/ingest` registrado y accesible desde `/docs`.
4. El fichero `app/embedding_pipeline/SANITY_CHECK.md` con los resultados de las tres parejas y el comentario.
5. `data/budgets_sample.json` (el sample del programa) versionado.
6. README actualizado con las instrucciones de ejecución (cómo invocar el endpoint, cómo correr `compare.py` dentro y fuera del contenedor) y el árbol de `Estructura` ampliado con el módulo nuevo y el spec.
7. `pyproject.toml` + `uv.lock` actualizados con `tiktoken`.

No es necesario que la rama contenga tests automatizados; los tests son un tema deliberadamente diferido a una sesión posterior. Si decides escribir alguno por tu cuenta, perfecto, pero no es requisito.

## Criterios de "hecho"

El ejercicio está completo cuando:

- [ ] `POST /api/v1/embeddings/ingest` acepta `{"budgets": [...]}` y devuelve `chunks` vectorizados más `stats`, y aparece en `/docs` con su contrato completo.
- [ ] Invocado con `data/budgets_sample.json` (15 presupuestos), la respuesta trae **52 chunks** (~4.1k tokens y ~$0.00008 de coste estimado, orientativos).
- [ ] Cada `chunk_id` es trazable (`BUD-...::COMP-...`) y su `metadata` incluye los 7 campos filtrables.
- [ ] `uv run python scripts/compare.py --text-a "..." --text-b "..."` devuelve un coseno en `[-1, 1]` y funciona también vía `docker compose exec api …`.
- [ ] `app/embedding_pipeline/SANITY_CHECK.md` contiene los tres valores numéricos y el comentario.
- [ ] README actualizado (endpoint + `compare.py` dentro y fuera del contenedor) y `pyproject.toml` con `tiktoken`.
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` y `uv run mypy app frontend` pasan.

## Checklist de verificación

- [ ] `app/embedding_pipeline/` con `__init__.py`, `chunker.py`, `embedder.py`, `schemas.py` y `router.py`.
- [ ] `Sector` y `Complexity` son `StrEnum` (no strings sueltos) y los modelos de entrada usan `ConfigDict(extra="forbid")`; un sector inválido o un campo desconocido falla con 422.
- [ ] Un componente = un chunk: el número de chunks es la suma de componentes de los presupuestos de entrada.
- [ ] El `text` del chunk empieza por las líneas `[Project: …]` y `[Client sector: …]` (cabecera de contexto).
- [ ] `token_count` se calcula con `tiktoken.encoding_for_model("text-embedding-3-small")` (encoding resuelto una sola vez, a nivel de módulo).
- [ ] `embed_many` llama a la API en batches de 100 con `itertools.batched` (no una llamada por chunk) y loguea `embedding.batch.end` con chunks, tokens y `latency_ms`.
- [ ] `RateLimitError` reintenta con esperas 1s/2s/4s (evento `embedding.batch.rate_limited`); el resto de errores se propagan.
- [ ] El precio está como constante de módulo etiquetada (`PRICE_PER_MILLION_TOKENS_USD = 0.02`) y `stats.estimated_cost_usd` se calcula con `estimate_cost_usd` y los tokens reales.
- [ ] `get_chunker` y `get_embedder` viven en `app/dependencies.py` con `lru_cache`; el embedder se construye con `from_settings` y degrada a `None` sin clave/endpoint.
- [ ] El router está registrado en `main.py` con prefijo `/api/v1` → ruta final `/api/v1/embeddings/ingest`.
- [ ] 422 lo gestiona FastAPI; un fallo del embedder o de la API devuelve 500 con mensaje genérico y el detalle en logs (`embedding.ingest.failed`).
- [ ] `compare.py` calcula el coseno con `math.sumprod`/`math.hypot` (sin `numpy`/`scikit-learn`) y sale con código 1 si no hay embeddings configurados.
- [ ] `data/budgets_sample.json` está versionado y viaja en la imagen (COPY de `data/`).

## Documentación de referencia

- OpenAI — Embeddings guide: https://platform.openai.com/docs/guides/embeddings
- OpenAI API reference — Create embeddings (`input` como lista, `dimensions`, precios): https://platform.openai.com/docs/api-reference/embeddings/create
- `tiktoken` — encodings y `encoding_for_model`: https://github.com/openai/tiktoken
- FastAPI — Dependencies (`Depends`, `lru_cache`, overrides en tests): https://fastapi.tiangolo.com/tutorial/dependencies/
- structlog — logging estructurado (eventos con kwargs): https://www.structlog.org/en/stable/
- Material asíncrono del programa — artículo 3 (contextual chunk headers).

## Nota

**Chunking estructural antes que chunking "inteligente".** El JSON ya tiene la estructura que necesitamos: un componente es una unidad semántica completa (un trabajo estimable, con descripción, stack y horas). Partirlo por tamaño fijo o con overlap rompería esa unidad y ensuciaría el vector. La consecuencia didáctica es directa: si una descripción es anormalmente larga, el `token_count` lo delata y lo discutimos en directo; no lo tapamos con un split. Las estrategias recursive/semantic/hierarchical/late chunking existen precisamente para documentos que *no* traen estructura — este no es el caso.

**La cabecera de contexto no es "enriquecimiento con LLM".** Prepender proyecto, sector, año y tecnología principal es copiar contexto determinista del documento padre; no interviene ningún modelo generando texto. Sin esa cabecera, decenas de componentes del corpus ("authentication backend", "notification service"…) colapsarían en la misma región borrosa del espacio de embeddings. Con ella, el vector conserva a quién pertenece el componente. Es el mismo mecanismo que el artículo 3 llama *contextual chunk headers*.

**Batching y rate limits.** Una llamada por chunk no cambia el coste en tokens, pero multiplica la latencia y expone el cómputo a más puntas de rate limit. El backoff 1s/2s/4s es la política mínima viable: suficiente para un corpus de este tamaño, insuficiente para producción (jitter, reintentos por 5xx, circuit breaker…). Los límites de la Sesión 03 nos dejan claro que la política de reintentos es una decisión de diseño, no un detalle.

**Los umbrales no son constantes físicas.** En una ejecución de referencia (mismo modelo, mismos textos que pide este enunciado) los resultados fueron: **Pareja A 0.5957** — apenas *por debajo* del 0.6 orientativo, pese a que ambos textos describen lo mismo con vocabulario superficial distinto; **Pareja B 0.1920**, cómodamente baja; **Pareja C 0.5407** — dos frases cortas y genéricas puntúan casi tan "cerca" como la pareja que habla del mismo concepto. La lectura: la similitud depende del fraseo y de la cantidad de señal discriminativa (los textos vagos se parecen a casi todo), y un umbral decimal es una convención operativa, no una propiedad del modelo. Si tu ejecución da valores distintos, no es un error: es el material del directo.

**Coste despreciable, decisión no.** 15 presupuestos → 52 chunks → ~4.1k tokens → ~$0.00008. Embeber es barato; por eso la decisión importante no es "cuánto cuesta hoy", sino **cuándo se re-embebe**: sin persistencia (Sesión 08) cada ingesta vuelve a pagar todo el corpus. La constante `PRICE_PER_MILLION_TOKENS_USD` está etiquetada porque cambia con el tiempo; el `stats.estimated_cost_usd` existe para que esa conversación tenga números.

**Qué es nuestro (y por qué).** Frente a la implementación canónica del enunciado, este repo introduce cinco decisiones deliberadas: (1) `StrEnum` en lugar de `Literal` para los universos cerrados, alineado con los contratos de estimación; (2) `extra="forbid"` en los modelos de entrada, para que un JSON mal formado falle con un 422 localizado; (3) `OpenAIEmbedder.from_settings` como única fábrica del cliente, compartida por la API y el CLI; (4) `itertools.batched`, `math.sumprod` y `math.hypot` como vocabulario estándar de Python 3.12 (sin numpy); y (5) eventos de log con la convención dotted de la sesión 03 (`embedding.batch.end`, `embedding.batch.rate_limited`, `embedding.ingest.*`). El comportamiento observable —un componente = un chunk, batches de 100, coste y umbrales— no cambia: cambia cómo se escribe y se integra en el repo.

**Fuera de alcance.** Otras estrategias de chunking (recursive, semantic, hierarchical, late chunking, contextual retrieval), comparativa de modelos de embeddings, enriquecimiento de chunks con LLM, persistencia en base vectorial, búsqueda semántica, métricas de calidad de retrieval (recall@k, NDCG) y cualquier cambio en la UI o en el cliente de negocio. Todo eso es del directo o de la Sesión 08. El ejercicio **produce y comprueba vectores**; no los guarda ni los busca todavía.
