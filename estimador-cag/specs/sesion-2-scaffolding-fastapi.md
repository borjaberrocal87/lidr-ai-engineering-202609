# Spec — Proyecto 1: Estimador de software con arquitectura CAG

## Objetivo

Construir la estructura base del Proyecto 1: una aplicación FastAPI con un endpoint que reciba el texto de una transcripción de reunión y devuelva una estimación de software generada por un LLM, utilizando arquitectura CAG (contexto estático inyectado en el prompt).

Al finalizar este ejercicio, tendrás un servicio funcional que:

- Recibe una transcripción de reunión vía API REST
- Inyecta contexto estático (ejemplos de estimaciones previas) directamente en el prompt
- Envía la petición a un LLM (OpenAI o Anthropic)
- Devuelve la estimación generada como respuesta JSON

## Contexto del proyecto

Este es el inicio del proyecto que ejecutaremos a lo largo del programa.

La arquitectura inicial es CAG: todo el contexto que necesita el modelo viaja en cada llamada — no hay base de datos, no hay retrieval, no hay persistencia.

¿Por qué CAG? Porque los datos de referencia (unas pocas estimaciones de ejemplo) caben perfectamente en la ventana de contexto del modelo. No necesitamos infraestructura adicional. Esta simplicidad nos permite centrarnos en la lógica de negocio y en la calidad del prompt antes de evolucionar hacia RAG en módulos posteriores.

## Requisitos para el ejercicio

- Python 3.11+ instalado
- `uv` instalado como gestor de paquetes (instrucciones de instalación)
- Cuenta activa en OpenAI Platform y/o Anthropic Console con créditos disponibles (configurada en la Sesión 01)
- API key disponible como variable de entorno

## ✍ Ejercicio

### Paso 1 — Inicializar el proyecto

Crea un nuevo directorio para el proyecto y configúralo con `uv`:

```
estimador-cag/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── estimations.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── llm_service.py
│   └── context/
│       ├── __init__.py
│       └── examples.py
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

Dependencias necesarias:

- `fastapi`
- `uvicorn[standard]`
- `pydantic-settings`
- `openai` y/o `anthropic` (según el proveedor que uses)
- `python-dotenv`

> **Nota sobre la estructura:** separamos responsabilidades desde el inicio. `routers/` gestiona los endpoints HTTP, `services/` contiene la lógica de negocio (llamada al LLM), y `context/` almacena los datos estáticos que inyectaremos en el prompt. Esta separación es una decisión arquitectónica deliberada — no es necesaria para que funcione, pero lo es para que escale.

### Paso 2 — Configuración con variables de entorno

Implementa la configuración del proyecto usando Pydantic `BaseSettings`. El archivo `config.py` debe cargar las siguientes variables desde `.env`:

<!-- image.png (referencia visual de las variables de configuración) -->

Crea también el archivo `.env.example` con las variables sin valores (para documentar qué variables necesita el proyecto) y el `.env` con tus valores reales. Asegúrate de que `.env` está en `.gitignore`.

### Paso 3 — Datos de contexto estático

En `context/examples.py`, define al menos dos ejemplos de estimaciones previas que se inyectarán en el prompt. Estos ejemplos representan el "conocimiento" del sistema — las estimaciones históricas que el modelo usará como referencia para generar nuevas estimaciones.

Cada ejemplo debe incluir:

- Un resumen de la transcripción de reunión original (qué pedía el cliente)
- La estimación generada (desglose de tareas con horas/costes)

Puedes usar datos ficticios. Ejemplo orientativo de estructura:

```python
ESTIMATION_EXAMPLES = [
    {
        "meeting_summary": "El cliente necesita una plataforma web de gestión de inventario...",
        "estimation": """
        ## Estimación: Plataforma de Gestión de Inventario

        ### Desglose de tareas:
        1. Diseño UI/UX: 40 horas
        2. Backend API (CRUD inventario): 60 horas
        3. Autenticación y roles: 20 horas
        4. Dashboard con métricas: 30 horas
        5. Testing y QA: 25 horas

        **Total estimado: 175 horas**
        **Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
        **Duración estimada: 6-8 semanas**
        """
    },
    # ... segundo ejemplo
]
```

Piensa en estos ejemplos como el equivalente a los "few-shot examples" que usamos en prompt engineering. Cuanto más representativos sean del tipo de estimaciones que queremos generar, mejor será la calidad del output.

### Paso 4 — Servicio de llamada al LLM

En `services/llm_service.py`, implementa una función que:

1. **Construya el system prompt** — Define el rol del modelo: es un estimador de software experto que genera estimaciones basándose en ejemplos previos y en la transcripción de una nueva reunión.
2. **Inyecte los ejemplos de contexto** — Los ejemplos del Paso 3 deben incluirse en el prompt como referencia. Esto es el corazón de la arquitectura CAG: el contexto viaja directamente en la llamada.
3. **Envíe la transcripción del usuario** — La transcripción de la reunión se envía como mensaje de usuario.
4. **Devuelva la respuesta del modelo** — La estimación generada.

La estructura de mensajes debe seguir el patrón:

```
[system]   → Instrucciones + ejemplos de estimaciones previas
[user]     → Transcripción de la reunión a estimar
[assistant] → (respuesta del modelo: la estimación)
```

Implementa la llamada para al menos un proveedor (OpenAI o Anthropic). Si quieres implementar ambos, usa `LLM_PROVIDER` del config para seleccionar cuál usar.

Utiliza los modelos económicos para este ejercicio: `gpt-4o-mini` para OpenAI o `claude-haiku-4-5` para Anthropic.

### Paso 5 — Endpoint de estimación

En `routers/estimations.py`, crea un endpoint `POST /api/v1/estimate` que:

- Reciba un body JSON con al menos el campo `transcription` (texto de la transcripción de la reunión)
- Llame al servicio LLM del Paso 4 mediante Postman/curl o similar
- Devuelva la estimación generada en formato JSON

Define los schemas de request y response con Pydantic:

**Request:**

```json
{
  "transcription": "En la reunión con el cliente se discutió la necesidad de..."
}
```

**Response:**

```json
{
  "estimation": "## Estimación: ...\\n\\n### Desglose de tareas:\\n...",
  "model": "gpt-4o-mini",
  "provider": "openai"
}
```

Puedes añadir campos adicionales al response si lo consideras útil (tokens utilizados, coste estimado, timestamp, etc.).

### Paso 6 — Aplicación FastAPI

En `main.py`, configura la aplicación FastAPI:

- Incluye el router de estimaciones con el prefijo `/api/v1`
- Añade un endpoint `GET /health` que devuelva el estado del servicio
- Configura el título y descripción de la API para la documentación automática (Swagger en `/docs`)

### Paso 7 — Verificación

Arranca el servidor y prueba el endpoint:

```bash
# Arrancar el servidor
uv run uvicorn app.main:app --reload

# En otra terminal, probar el endpoint
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas. El diseño ya existe en Figma."
  }'
```

También puedes usar la interfaz de Swagger UI accediendo a http://localhost:8000/docs desde tu navegador.

## Checklist de verificación

Antes de considerar el ejercicio completado, verifica:

- [ ] El proyecto arranca sin errores con `uv run uvicorn app.main:app --reload`
- [ ] Las API keys se cargan desde `.env` y nunca aparecen en el código
- [ ] El endpoint `GET /health` responde con status 200
- [ ] El endpoint `POST /api/v1/estimate` recibe una transcripción y devuelve una estimación
- [ ] La estimación generada hace referencia o se inspira en los ejemplos de contexto inyectados
- [ ] La documentación Swagger está accesible en `/docs`
- [ ] El archivo `.env` está en `.gitignore`
