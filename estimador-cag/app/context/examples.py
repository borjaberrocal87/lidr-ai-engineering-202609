"""Contexto estático (CAG) del estimador.

Estos ejemplos se inyectan en el system prompt en cada llamada al LLM.
Representan las estimaciones históricas que el modelo usa como referencia
para generar nuevas estimaciones. No hay retrieval ni base de datos: todo
el contexto viaja dentro del prompt.
"""

ESTIMATION_EXAMPLES: list[dict[str, str]] = [
    {
        "meeting_summary": (
            "El cliente necesita una plataforma web de gestión de inventario para "
            "su cadena de 5 tiendas. Requiere alta de productos, control de stock "
            "por tienda, alertas de stock mínimo, gestión de proveedores y un "
            "dashboard con métricas de rotación. El diseño de marca ya existe y "
            "tienen preferencia por un stack web moderno."
        ),
        "estimation": """## Estimación: Plataforma de Gestión de Inventario

### Desglose de tareas:
1. Diseño UI/UX y flujos: 40 horas
2. Backend API (CRUD de productos, stock y proveedores): 60 horas
3. Autenticación, roles y permisos por tienda: 20 horas
4. Dashboard con métricas de rotación y alertas: 30 horas
5. Testing y QA: 25 horas

**Total estimado: 175 horas**
**Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 6-8 semanas**
**Rango de coste: 10.500 - 14.000 EUR (a 60-80 EUR/hora)**
""",
    },
    {
        "meeting_summary": (
            "Una startup de educación quiere un MVP de app móvil para cursos en "
            "vídeo. Los usuarios deben poder registrarse, ver un catálogo de "
            "cursos, reproducir vídeos, marcar progreso y recibir un certificado "
            "al finalizar. Necesitan panel de administración para subir cursos y "
            "un sistema de pagos con suscripción mensual. Buscan lanzar en 3 meses."
        ),
        "estimation": """## Estimación: MVP App Móvil de Cursos

### Desglose de tareas:
1. Diseño UI/UX de app y panel admin: 50 horas
2. App móvil (catálogo, reproductor y progreso): 90 horas
3. Backend API + autenticación: 70 horas
4. Panel de administración de cursos: 40 horas
5. Integración de pagos y suscripciones: 30 horas
6. Certificados y Testing/QA: 40 horas

**Total estimado: 320 horas**
**Equipo recomendado: 2 desarrolladores (1 móvil, 1 backend) + 1 diseñador UX**
**Duración estimada: 10-12 semanas**
**Rango de coste: 19.200 - 25.600 EUR (a 60-80 EUR/hora)**
""",
    },
]
