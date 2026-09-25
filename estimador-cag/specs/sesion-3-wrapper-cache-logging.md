# Spec — Proyecto 1: Wrapper de proveedores, cacheo de respuestas y trazabilidad

## Objetivo

Cerrar el bloque opcional de la sesión 03 añadiendo al Proyecto 1 tres capas transversales sobre la llamada al LLM:

- Un **wrapper de abstracción de proveedores** que unifique las llamadas (con fallback entre modelos) y elimine el `if/elif` de `llm_service.py`.
- Un **cacheo inteligente de respuestas** que evite pagar dos veces la misma estimación, con una abstracción de caché que soporte memoria (por defecto) y Redis (opcional).
- Una **capa de logging/trazabilidad** que correlacione cada petición de extremo a extremo y registre, además de los tokens y la latencia que ya se miden, el coste estimado, los aciertos de caché y si se usó el modelo de fallback.

Al finalizar este ejercicio, tendrás un estimador que:

- Habla con OpenAI, Anthropic o cualquier endpoint OpenAI-compatible a través de una única interfaz.
- Si el modelo primario falla, reintenta y cae al modelo de fallback sin que el llamante se entere.
- Devuelve `cache_hit: true` y coste cero en peticiones repetidas, sin volver a llamar al proveedor.
- Emite logs estructurados con un `request_id`/`call_id` que permiten reconstruir una petición completa.

## Contexto del proyecto

Partimos del proyecto de la sesión 02/03 (backend FastAPI + Streamlit + proveedor `custom`). Hoy la selección de proveedor vive en `app/services/llm_service.py` con un `if/elif` sobre `settings.llm_provider` (`llm_service.py:183` y `llm_service.py:224`) y una función `_estimate_with_*` / `_stream_with_*` por proveedor. Añadir un proveedor es tocar el servicio; y no hay caché ni fallback.

El logging ya existe: `app/logging_config.py` configura [structlog](https://www.structlog.org/) con `LOG_LEVEL`/`LOG_FORMAT`, y `llm_service.py` emite `llm.call.start/end/error` y `llm.stream.*` con tokens y latencia. Esta capa **se amplía**, no se reescribe.

La petición de evolucionar hacia un wrapper ya estaba apuntada en `specs/proveedor-openai-compatible.md:263` y en la nota final de `specs/sesion-3-interfaz-conversacional-streamlit.md:183`.

> **Arquitectura objetivo.** La lógica de negocio (construir el prompt CAG, validar la transcripción, mapear resultados) se queda en `llm_service.py`. La **llamada al proveedor** se mueve a un wrapper (`llm_wrapper.py`) que aporta fallback, caché, coste y logging. El wrapper se inyecta como singleton desde `app/dependencies.py` para poder sustituirlo en tests.

## Requisitos para el ejercicio

- Proyecto de la sesión 02/03 funcionando (`uv run pytest` en verde).
- Al menos una API key configurada (OpenAI y/o Anthropic). Para el fallback conviene tener las dos.
- `uv` como gestor de paquetes.
- Docker y Docker Compose para levantar Redis en local (opcional: el caché en memoria no lo necesita).
- Para probar el proveedor `custom`: un endpoint OpenAI-compatible (Ollama, LM Studio, vLLM, NaN API…).

## ✍ Ejercicio

### Paso 1 — Dependencias y variables de entorno

Añade las dependencias:

```bash
uv add litellm redis
uv add --dev fakeredis
```

- `litellm` unifica la llamada a los proveedores y aporta el `Router` con fallback.
- `redis` es el cliente de la implementación Redis del caché.
- `fakeredis` permite testear la caché Redis sin levantar un servidor.

Amplía `Settings` en `app/config.py`:

```python
llm_provider: Literal["openai", "anthropic", "custom"] = "openai"
llm_model: str = ""  # modelo primario (ya existe)
llm_fallback_model: str = ""  # vacío = sin fallback
llm_routing_mode: Literal["fallback", "balanced"] = "fallback"
temperature: float = DEFAULT_TEMPERATURE

# ... llm_timeout_seconds, llm_max_retries, llm_max_tokens (ya existen) ...

# Caché de respuestas
cache_backend: Literal["memory", "redis", "none"] = "memory"
cache_ttl: int = 86400
redis_url: str = "redis://localhost:6379"
```

Y refléjalo en `.env.example`:

```dotenv
# Fallback del wrapper (vacío = sin fallback). LiteLLM decide el proveedor por
# el prefijo/nombre del modelo: claude-haiku-4-5, gpt-4o-mini, openai/<modelo>...
LLM_FALLBACK_MODEL=
# Modo del Router: fallback (primario estricto) | balanced (reparto simple-shuffle)
LLM_ROUTING_MODE=fallback

# Caché de respuestas: memory (por defecto, sin infra) | redis | none
CACHE_BACKEND=memory
CACHE_TTL=86400
REDIS_URL=redis://localhost:6379
```

> **Sobre los nombres.** Reutilizamos `LLM_MODEL`, `LLM_TIMEOUT_SECONDS` y `LLM_MAX_RETRIES` para no romper `.env`, README ni tests. Los campos nuevos del proveedor son `LLM_FALLBACK_MODEL` y `LLM_ROUTING_MODE`.
>
> **`LLM_ROUTING_MODE`.** LiteLLM reparte las peticiones entre deployments que comparten `model_name` (estrategia por defecto `simple-shuffle`), así que **no basta** con registrar un "fallback" para que actúe como tal: si ambos comparten nombre, se usan ~50/50. Con `fallback` (por defecto) registramos `primary` y `fallback` con nombres distintos y `fallbacks=[{"primary": ["fallback"]}]`, de modo que el primario se usa siempre y el secundario solo entra si el primario lanza. Con `balanced` mantenemos el nombre compartido para repartir carga a propósito.

### Paso 2 — Capa de caché (`app/services/cache.py`)

Crea una abstracción con una clave determinista y dos implementaciones. La clave es un **SHA-256 del system prompt completo + mensaje de usuario + los knobs de generación**, de forma que cualquier cambio en el prompt CAG (ejemplos, instrucciones, límite de datos) invalida la caché solo, sin flush manual.

```python
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class ResponseCache(Protocol):
    """Interfaz mínima de caché exact-match."""

    def get(self, key: str) -> dict[str, Any] | None: ...
    def set(self, key: str, value: dict[str, Any]) -> None: ...


def make_cache_key(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
) -> str:
    payload = json.dumps(
        {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"estimation:{digest}"
```

Implementa `InMemoryCache` (por defecto, con TTL y un `threading.Lock` para ser seguro entre hilos):

```python
class InMemoryCache:
    """Caché TTL en proceso. Cero infraestructura: ideal para desarrollo y tests."""

    def __init__(self, ttl: int = 86400) -> None:
        self.ttl = ttl
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                logger.info("cache.miss")
                return None
            expires_at, payload = entry
            if expires_at < time.time():
                self._store.pop(key, None)
                logger.info("cache.miss", reason="expired")
                return None
        logger.info("cache.hit")
        return payload

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = (time.time() + self.ttl, value)
        logger.info("cache.stored", ttl=self.ttl)
```

E implementa `RedisCache`, que falla de forma **degradada** (si Redis no responde, se comporta como un miss y no tumba la petición):

```python
class RedisCache:
    def __init__(self, client: Any, ttl: int = 86400) -> None:
        self.redis = client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = 86400) -> "RedisCache":
        import redis

        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            cached = self.redis.get(key)
        except Exception as exc:  # redis.RedisError
            logger.warning("cache.get_failed", error=str(exc))
            return None
        if cached:
            logger.info("cache.hit")
            return json.loads(cached)
        logger.info("cache.miss")
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            self.redis.set(key, json.dumps(value), ex=self.ttl)
            logger.info("cache.stored", ttl=self.ttl)
        except Exception as exc:  # redis.RedisError
            logger.warning("cache.set_failed", error=str(exc))
```

> **Qué guardar.** Un dict serializable con `estimation`, `model`, `provider`, `input_tokens`, `output_tokens`, `truncated`, `latency_ms` y `cost_usd`. **No** cacheados respuestas truncadas ni vacías: son fallos, no resultados.

### Paso 3 — Wrapper de abstracción de proveedores (`app/services/llm_wrapper.py`)

Crea `LLMWrapper`, que envuelve `litellm.Router` con un modelo primario y otro de fallback y expone dos primitivas: `complete()` (bloqueante) y `complete_stream()` (generador). El wrapper es agnóstico del prompt CAG: recibe `system_prompt` y `user_message`.

```python
from __future__ import annotations

import time
from typing import Any, Iterator

import litellm
import structlog
from litellm import Router

from app.services.cache import ResponseCache, make_cache_key

log = structlog.get_logger(__name__)

# Coste por 1M de tokens (USD). Actualízalo cuando cambien los precios.
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def _normalise_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _provider_from_model(model: str) -> str:
    name = _normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3")):
        return "openai"
    return "custom"


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    base = _normalise_model_name(model)
    costs = MODEL_COSTS.get(base) or MODEL_COSTS.get(model) or {"input": 0.0, "output": 0.0}
    return round((tokens_in * costs["input"] + tokens_out * costs["output"]) / 1_000_000, 6)
```

El constructor monta el `Router` según `routing_mode`. Ojo: LiteLLM reparte las peticiones entre deployments que comparten `model_name` (estrategia por defecto `simple-shuffle`), así que registrarlos a todos con el mismo nombre **no** da un fallback, da un balanceo. Por eso:

- `fallback` (por defecto): nombres distintos `primary`/`fallback` y `fallbacks=[{"primary": ["fallback"]}]`; el primario se usa siempre y el secundario entra solo si el primario lanza.
- `balanced`: nombre compartido `estimator` para repartir carga a propósito.

```python
ROUTER_PRIMARY_NAME = "primary"
ROUTER_FALLBACK_NAME = "fallback"
ROUTER_BALANCED_NAME = "estimator"


class LLMWrapper:
    def __init__(
        self,
        *,
        primary_model: str,
        fallback_model: str,
        routing_mode: Literal["fallback", "balanced"] = "fallback",
        timeout: float,
        num_retries: int,
        cache: ResponseCache,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        custom_base_url: str = "",
        custom_api_key: str | None = None,
    ) -> None:
        self.primary_model = primary_model
        self.cache = cache
        if routing_mode == "balanced":
            primary_name = fallback_name = ROUTER_BALANCED_NAME
        else:
            primary_name, fallback_name = ROUTER_PRIMARY_NAME, ROUTER_FALLBACK_NAME
        self.router_model_name = primary_name

        deployments = [self._deployment(model_name=primary_name, model=primary_model, ...)]
        fallbacks: list[dict[str, list[str]]] = []
        if fallback_model:
            deployments.append(
                self._deployment(model_name=fallback_name, model=fallback_model, ...)
            )
            fallbacks = [{primary_name: [fallback_name]}]

        self.router = Router(
            model_list=deployments,
            fallbacks=fallbacks,
            num_retries=num_retries,
        )
```

La llamada al Router usa `self.router_model_name`. Para detectar si respondió el secundario, compara el modelo devuelto con `fallback_model` **normalizando el sufijo de fecha** (Anthropic responde `claude-haiku-4-5-20251001` aunque registres `claude-haiku-4-5`): `_canonical_model_name()` quita el prefijo de proveedor y el sufijo `-\d{8}$`.

`complete()` calcula la clave, consulta la caché, y si hay miss llama al router, normaliza la respuesta, estima el coste y **guarda en caché solo si no está truncada**.

> **Clave sobre el contenido lógico, no sobre el artefacto enviado.** La transcripción viaja al modelo envuelta en una etiqueta con un nonce aleatorio (`_wrap_transcription`). Si hashiéramos ese texto tal cual, el nonce cambiaría en cada petición y **la caché nunca acertaría**. Por eso `complete()` acepta `cache_user_message`: el mensaje canónico (la transcripción sin envolver) que se usa para la clave, mientras que a `user_message` —ya envuelto— se le sigue enviando al modelo.

```python
    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float | None,
        max_tokens: int,
        cache_user_message: str | None = None,
    ) -> dict[str, Any]:
        keyed_message = user_message if cache_user_message is None else cache_user_message
        key = make_cache_key(
            system_prompt=system_prompt,
            user_message=keyed_message,
            model=self.primary_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "cache_hit": True}  # conserva fallback_used y coste originales

        log.info("llm.call.start", model=self.primary_model, max_tokens=max_tokens)
        started_at = time.perf_counter()
        try:
            response = self.router.completion(
                model=self.router_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            log.error(
                "llm.call.error",
                error_type=type(exc).__name__,
                latency_ms=_elapsed_ms(started_at),
            )
            raise

        result = self._normalise(response, latency_ms=_elapsed_ms(started_at))
        log.info(
            "llm.call.end",
            model=result["model"],
            provider=result["provider"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=result["cost_usd"],
            latency_ms=result["latency_ms"],
            cache_hit=False,
        )
        if not result["truncated"] and result["estimation"]:
            self.cache.set(key, result)
        return {**result, "cache_hit": False}
```

`complete_stream()` sigue el mismo patrón: si hay hit, **reproduce la estimación cacheada como un único chunk** (para que la UI no note la diferencia); si no, va cediendo deltas y guarda el texto completo al terminar:

```python
def complete_stream(
    self,
    *,
    system_prompt: str,
    user_message: str,
    temperature: float | None,
    max_tokens: int,
    cache_user_message: str | None = None,
) -> Iterator[str]:
    keyed_message = user_message if cache_user_message is None else cache_user_message
    key = make_cache_key(...)  # mismos knobs que complete(), con keyed_message
    cached = self.cache.get(key)
    if cached is not None:
        log.info("llm.stream.cache_hit", chars=len(cached.get("estimation", "")))
        yield cached.get("estimation", "")
        return

    log.info("llm.stream.start", model=self.primary_model)
    started_at = time.perf_counter()
    parts: list[str] = []
    response = self.router.completion(
        model="estimator",
        messages=[...],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in response:
        delta = _extract_delta(chunk)
        if delta:
            parts.append(delta)
            yield delta

    rendered = "".join(parts)
    log.info("llm.stream.end", chars=len(rendered), latency_ms=_elapsed_ms(started_at))
    if rendered:
        self.cache.set(
            key,
            {
                "estimation": rendered,
                "model": self.primary_model,
                "provider": _provider_from_model(self.primary_model),
                "input_tokens": None,
                "output_tokens": None,
                "truncated": False,
                "latency_ms": _elapsed_ms(started_at),
                "cost_usd": 0.0,
            },
        )
```

> **`usage` en streaming.** LiteLLM no siempre reporta tokens en streaming; deja los contadores a `None` y no inventes cifras. El coste de una respuesta cacheada es `0.0`.

### Paso 4 — Inyección y refactor de `llm_service.py`

Crea `app/dependencies.py` con los singletons (caché y wrapper), cacheados con `lru_cache`:

```python
from functools import lru_cache

from app.config import Settings, get_settings
from app.services.cache import InMemoryCache, RedisCache, ResponseCache
from app.services.llm_wrapper import LLMWrapper


@lru_cache
def get_cache() -> ResponseCache:
    settings = get_settings()
    if settings.cache_backend == "redis":
        return RedisCache.from_url(settings.redis_url, ttl=settings.cache_ttl)
    return InMemoryCache(ttl=settings.cache_ttl)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        primary_model=_litellm_model_name(settings),
        fallback_model=settings.llm_fallback_model,
        timeout=settings.llm_timeout_seconds,
        num_retries=settings.llm_max_retries,
        cache=get_cache(),
        openai_api_key=...,
        anthropic_api_key=...,
        custom_base_url=settings.custom_llm_base_url,
        custom_api_key=...,
    )
```

En `llm_service.py`:

1. Elimina `_dispatch_generate`, `_dispatch_stream` y las funciones `_estimate_with_*` / `_stream_with_*`.
2. `generate_estimation()` y `stream_estimation()` construyen el system prompt CAG y el mensaje de usuario (`_wrap_transcription`) y delegan en el wrapper. Al wrapper le pasan **dos** mensajes: `user_message` (envuelto, el que ve el modelo) y `cache_user_message=transcription` (canónico, el que alimenta la clave de caché).
3. Amplía el dataclass `EstimationResult` (y `StreamMetrics`) con los campos nuevos:

```python
@dataclass
class EstimationResult:
    estimation: str
    model: str
    provider: str
    temperature: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
    cache_hit: bool = False
    cost_usd: float | None = None
    fallback_used: bool = False
```

4. Conserva la validación de longitud (`_check_transcription`) y el guard de temperatura de Anthropic: son lógica de negocio del servicio, no del transporte.

```python
    result = wrapper.complete(
        system_prompt=build_system_prompt(),
        user_message=_wrap_transcription(transcription),
        temperature=settings.temperature,
        max_tokens=settings.llm_max_tokens,
        cache_user_message=transcription,  # sin el nonce: la caché sí acierta
    )
```

> **Compatibilidad.** `build_system_prompt()` y `_wrap_transcription()` no cambian: el wrapper recibe exactamente el mismo prompt CAG de siempre. Esa es la garantía de que el comportamiento del modelo no se altera, solo se instrumenta.

### Paso 5 — Capa de logging/trazabilidad

**Correlación.** Añade `structlog.contextvars.merge_contextvars` a los procesadores de `app/logging_config.py` y un middleware en `app/main.py` que propague un `request_id` (de la cabecera `X-Request-ID` o generado) y lo devuelva en la respuesta:

```python
import uuid

import structlog
from fastapi import Request


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["X-Request-ID"] = request_id
    return response
```

**Trazas enriquecidas.** El wrapper ya emite `llm.call.start/end/error` y `llm.stream.*`; añade `cache.hit` / `cache.miss` / `cache.stored` (Paso 2) y enlaza el costo. Como `merge_contextvars` está activo, todos esos eventos heredan el `request_id` sin pasarlo a mano, y `X-Request-ID` permite cruzar el log del backend con el del cliente Streamlit.

**Coste y métricas.** `_estimate_cost` (Paso 3) rellena `cost_usd` en `llm.call.end`. Opcionalmente, expón contadores agregados en `GET /health` (aciertos de caché, fallbacks, coste acumulado de la sesión) con un pequeño registro en memoria.

**Privacidad.** Se mantiene la regla del proyecto: solo metadatos. Nunca la API key ni el texto de la transcripción. Cubre con un test que un secreto configurado no aparece en los logs (ya existe `test_logs_do_not_leak_api_key`).

### Paso 6 — Contrato HTTP y UI

Amplía los schemas en `app/schemas/estimations.py`:

```python
class EstimateResponse(BaseModel):
    # ... campos actuales ...
    cache_hit: bool = Field(False, description="True si la respuesta vino de la caché.")
    cost_usd: float | None = Field(None, description="Coste estimado de la llamada (USD).")
    fallback_used: bool = Field(False, description="True si se usó el modelo de fallback.")
```

Añade los mismos campos a `StreamDoneEvent` y rellénalos en `routers/estimations.py` a partir del resultado/métricas.

En el frontend:

- Añade los campos a `frontend/models.py` (`StreamMetrics`) y parsea el evento `done` en `frontend/client.py`.
- Muestra `cache_hit`, `cost_usd` y `fallback_used` en el panel lateral de `frontend/streamlit_app.py`, junto a las métricas que ya se muestran.

### Paso 7 — Tests

- `tests/test_cache.py`:
  - `make_cache_key` es determinista y cambia al cambiar cualquier knob (system prompt, modelo, `max_tokens`, temperatura).
  - `InMemoryCache`: roundtrip, miss y expiración por TTL.
  - `RedisCache` con `fakeredis`: roundtrip, miss y TTL.
  - Si Redis falla, `get` devuelve `None` y `set` no lanza.
- `tests/test_llm_wrapper.py` (mockea `Router.completion`, sin llamadas reales):
  - `complete` normaliza `estimation`, `model`, `provider`, tokens, `finish_reason` y `cost_usd`.
  - La segunda llamada idéntica devuelve `cache_hit=True` y **no** invoca al router.
  - `cache_user_message` desacopla la clave del envoltorio: dos `user_message` distintos (nonce distinto) con el mismo `cache_user_message` comparten entrada.
  - `routing_mode="fallback"` registra deployments `primary`/`fallback` y llama al router con `model="primary"`; `routing_mode="balanced"` registra ambos como `estimator`.
  - `fallback_used=True` cuando responde el secundario **con alias fechado** (`claude-haiku-4-5-20251001` vs `claude-haiku-4-5`), y `True` también en un cache hit que lo conserva.
  - Una respuesta truncada **no** se cachea.
  - `complete_stream` cede los deltas y guarda el texto completo; un segundo stream se reproduce desde caché.
  - `_estimate_cost` usa la tabla de precios.
- Amplía `tests/test_llm_logging.py`:
  - `llm.call.end` incluye `cache_hit=False` y `cost_usd > 0` en miss; `cache.hit` en el hit.
  - El `request_id` viaja en los eventos cuando está en el contexto.
  - Los secretos siguen sin aparecer.
- Ajusta `tests/test_estimations.py` y `tests/test_llm_service.py` al nuevo shape (`cache_hit`, `cost_usd`) y elimina los tests de los `_estimate_with_*` que ya no existan.
  - **Regresión obligatoria:** con un `LLMWrapper` real (caché en memoria, `Router.completion` mockeado), dos llamadas a `generate_estimation` con la **misma transcripción** deben invocar al router una sola vez y devolver `cache_hit=True` en la segunda. Repite el caso para `stream_estimation`. Sin este test, un nonce cambiante en la clave pasaría desapercibido.
- Revisa `tests/test_project_structure.py` para incluir `llm_wrapper.py`, `cache.py`, `errors.py` y `dependencies.py`.

### Paso 8 — Docker y documentación

Añade Redis a `docker-compose.yml` y haz que `api` dependa de él:

```yaml
  redis:
    image: redis:7-alpine
    container_name: estimador-cag-redis
    ports:
      - "${REDIS_PORT:-6379}:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
```

En el servicio `api`, sobrescribe `REDIS_URL=redis://redis:6379` y `CACHE_BACKEND=redis`, y añade `depends_on: [redis]`.

Documenta en el `README.md`:

- Las nuevas variables (`LLM_FALLBACK_MODEL`, `CACHE_BACKEND`, `CACHE_TTL`, `REDIS_URL`) en la tabla de variables de entorno.
- El wrapper (fallback), la caché (backend/TTL y cómo invalidar) y la trazabilidad (`request_id`, `cost_usd`).
- Cómo verificar la caché con Redis (`docker compose exec redis redis-cli KEYS 'estimation:*'`).

### Paso 9 — Verificación

```bash
# Backend local (caché en memoria no necesita Redis)
uv run uvicorn app.main:app --reload

# Primera llamada: cache_hit=false, coste > 0
curl -s localhost:8000/api/v1/estimate \
  -H 'Content-Type: application/json' \
  -d '{"transcription": "Necesitamos un CRM con auth, contactos y roles. MVP en seis semanas."}' | jq '{cache_hit, cost_usd}'

# Segunda llamada idéntica: cache_hit=true, coste 0, latencia mínima
curl -s localhost:8000/api/v1/estimate \
  -H 'Content-Type: application/json' \
  -d '{"transcription": "Necesitamos un CRM con auth, contactos y roles. MVP en seis semanas."}' | jq '{cache_hit, cost_usd}'
```

Con `CACHE_BACKEND=redis` (y Redis levantado), inspecciona las claves:

```bash
docker compose exec redis redis-cli KEYS 'estimation:*'
```

Comprueba en los logs que cada petición lleva un `request_id`, que el hit emite `cache.hit` y que el endpoint SSE también se sirve desde caché. Prueba el fallback configurando un `LLM_MODEL` inválido y un `LLM_FALLBACK_MODEL` válido: la estimación debe salir igualmente (LiteLLM salta al segundo deployment).

## Checklist de verificación

Antes de considerar el ejercicio completado, verifica:

- [ ] `generate_estimation` y `stream_estimation` delegan en `LLMWrapper`; no queda ningún `if/elif` por proveedor en `llm_service.py`
- [ ] Si el modelo primario falla, se usa `LLM_FALLBACK_MODEL` sin que el llamante reciba un error
- [ ] Con `LLM_ROUTING_MODE=fallback` el primario se usa siempre (no se reparte); con `balanced` sí se reparte entre ambos
- [ ] `fallback_used` es correcto aunque el proveedor devuelva el snapshot fechado, y se conserva en un cache hit
- [ ] Una petición repetida (misma transcripción) devuelve `cache_hit: true` y no vuelve a llamar al proveedor (tampoco en SSE): el nonce de `_wrap_transcription` **no** puede cambiar la clave (`cache_user_message`)
- [ ] `CACHE_BACKEND=memory` funciona sin infraestructura y `CACHE_BACKEND=redis` persiste las claves `estimation:*`
- [ ] Las respuestas truncadas o vacías **no** se cachean
- [ ] `EstimateResponse` y el evento SSE `done` incluyen `cache_hit` y `cost_usd`, y la UI los muestra
- [ ] Cada petición tiene un `request_id` (cabecera `X-Request-ID`) presente en todos sus logs
- [ ] Los logs nunca contienen la API key ni la transcripción
- [ ] El prompt CAG (con los ejemplos) sigue inyectándose exactamente igual a través del wrapper
- [ ] `OPEN_AI_KEY` / `ANTHROPIC_API_KEY` / `CUSTOM_LLM_API_KEY` se leen de `.env`, nunca hardcodeadas
- [ ] `uv run pytest`, `uv run ruff check .` y `uv run mypy app` pasan

## Documentación de referencia

- LiteLLM — Router y fallbacks: https://docs.litellm.ai/docs/routing
- LiteLLM — proveedores soportados (incluye el formato `openai/<model>`): https://docs.litellm.ai/docs/providers
- redis-py — `set(..., ex=...)`, TTL: https://redis-py.readthedocs.io/
- fakeredis — Redis en memoria para tests: https://github.com/cunla/fakeredis-py
- structlog — contextvars y logging correlacionado: https://www.structlog.org/en/stable/contextvars.html
- Redis — expiración de claves (`EX`): https://redis.io/docs/latest/commands/set/

## Nota

El caché exact-match es deliberadamente simple: dos transcripciones con la misma forma literal comparten resultado, pero dos paráfrasis no. Si en módulos posteriores se evoluciona hacia RAG, tiene sentido añadir caché semántico (embeddings + similitud) como capa por encima de esta. El wrapper y la abstracción `ResponseCache` están pensados para permitirlo sin tocar la lógica de estimación.
