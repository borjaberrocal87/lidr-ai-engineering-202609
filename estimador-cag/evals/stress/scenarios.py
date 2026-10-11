"""Perfiles sintéticos de conversación multi-turno (sesión 6, bloque 2).

Cada perfil es una secuencia de turnos sobre un mismo proyecto, diseñada para
forzar un comportamiento concreto del CAG:

- ``growing`` — el proyecto crece turno a turno con requisitos coherentes. Mide la
  curva de coste y si el ``project_name`` original sobrevive al turno 20.
- ``pivot`` — en el turno 5 cambia el stack (React → Flutter). Mide si la metadata
  se actualiza limpiamente o si ``mentioned_technologies`` acumula ambos.
- ``contradiction`` — el turno 3 fija un presupuesto de 30k y el turno 8 lo sube a
  80k. Mide cuál se preserva, cuál se promueve a ancla y cuál acaba en el summary.

El ``fact_to_remember`` de cada turno es el hecho que las llamadas posteriores
deberían recordar. Es un string corto y de match exacto (case-insensitive) que
alimenta a ``MemoryDriftMetric``. La estructura es inmutable: el runner consume
el mismo ``Scenario`` en cada repetición y una mutación accidental contaminaría
las pasadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    """Un turno de la conversación sintética."""

    turn_index: int
    transcript: str
    fact_to_remember: str | None = None
    """Hecho durable declarado en este turno, o ``None`` si no introduce ninguno."""


@dataclass(frozen=True)
class Scenario:
    """Un perfil de estrés completo."""

    name: str
    turns: tuple[Turn, ...]
    anchor_fact: str
    """Hecho persistente cuya supervivencia se mide turno a turno (curva recall/N)."""
    anchor_turn: int
    """Turno en el que se declaró ``anchor_fact``."""
    project_type: str = "web_saas"
    detail_level: str = "medium"
    output_format: str = "phases_table"
    description: str = field(default="", compare=False)


# --- Perfil 1: proyecto que crece -------------------------------------------

_GROWING_STEPS: tuple[tuple[str, str], ...] = (
    ("Súmale autenticación con SSO para el equipo de ventas.", "SSO"),
    (
        "Necesita ser multi-tenant con aislamiento de datos por cliente.",
        "multi-tenant",
    ),
    ("Añade un audit log de todas las acciones de los usuarios.", "audit log"),
    ("Queremos exportar los pedidos a CSV para el área financiera.", "CSV"),
    ("Hace falta gestión de roles y permisos granulares.", "roles"),
    (
        "Envía notificaciones por email cuando cambie el estado de un pedido.",
        "notificaciones por email",
    ),
    ("Expón una API REST documentada para integraciones externas.", "API REST"),
    ("Suma un panel de informes con métricas de actividad.", "informes"),
    ("Integra facturación con Stripe para las suscripciones.", "Stripe"),
    ("La interfaz debe ser responsive para usarla desde móvil.", "responsive"),
    ("Necesitamos modo offline con sincronización posterior.", "offline"),
    ("Necesitamos cifrado de los datos sensibles en reposo.", "cifrado"),
    ("Aplica rate limiting en la API pública.", "rate limiting"),
    ("Queremos webhooks para avisar a sistemas externos.", "webhooks"),
    ("Añade búsqueda de texto completo sobre clientes y pedidos.", "búsqueda"),
    ("La app debe estar en español e inglés.", "español"),
    ("Suma analítica de uso por funcionalidad.", "analítica"),
    ("Queremos monitorización con alertas de SLA.", "SLA"),
    ("Define una política de retención y borrado de datos.", "retención"),
)


def build_growing(n: int = 20) -> Scenario:
    """Proyecto que crece de forma coherente hasta ``n`` turnos (default 20).

    El turno 1 declara el nombre del proyecto ("Nimbus"); el resto añade
    requisitos. La pregunta del baseline es si "Nimbus" sigue vivo en el turno N.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    turns: list[Turn] = [
        Turn(
            1,
            "Queremos estimar un CRM llamado Nimbus para el equipo de ventas.",
            "Nimbus",
        )
    ]
    for index in range(1, n):
        transcript, fact = _GROWING_STEPS[(index - 1) % len(_GROWING_STEPS)]
        turns.append(Turn(index + 1, transcript, fact))
    return Scenario(
        name="growing",
        turns=tuple(turns),
        anchor_fact="Nimbus",
        anchor_turn=1,
        project_type="web_saas",
        description="Proyecto que crece: el nombre declarado en el turno 1 debe sobrevivir.",
    )


# --- Perfil 2: proyecto que pivota ------------------------------------------

_PIVOT_TURNS: tuple[Turn, ...] = (
    Turn(1, "App de campo en React para inspectores de obra.", "React"),
    Turn(2, "Los inspectores la usarán desde el móvil en obra.", None),
    Turn(3, "Necesita fotos geolocalizadas por inspección.", None),
    Turn(4, "Suma firma digital del inspector al cerrar el parte.", None),
    Turn(5, "Cambiamos de idea: finalmente lo haremos en Flutter.", "Flutter"),
    Turn(6, "Añade sincronización con el backend corporativo.", None),
    Turn(7, "Soporta funcionamiento sin conexión en sótanos.", None),
    Turn(8, "Genera un informe PDF de cada inspección.", None),
)


def build_pivot() -> Scenario:
    """El turno 5 cambia el stack; medimos si React se acumula con Flutter."""
    return Scenario(
        name="pivot",
        turns=_PIVOT_TURNS,
        anchor_fact="Flutter",
        anchor_turn=5,
        project_type="mobile_app",
        description="Proyecto que pivota: React (turno 1) → Flutter (turno 5).",
    )


# --- Perfil 3: proyecto que se contradice -----------------------------------

_CONTRADICTION_TURNS: tuple[Turn, ...] = (
    Turn(1, "Portal interno de gestión documental para 40 empleados.", None),
    Turn(2, "Debe integrarse con el directorio corporativo (LDAP).", "LDAP"),
    Turn(3, "El presupuesto está locked at 30000 EUR.", "30000 EUR"),
    Turn(4, "Necesita permisos por departamento.", None),
    Turn(5, "Suma un buscador de documentos con OCR.", "OCR"),
    Turn(6, "Los documentos deben quedar firmados digitalmente.", None),
    Turn(7, "Queremos un panel de auditoría de accesos.", None),
    Turn(
        8,
        "Lo hemos ampliado: el presupuesto está locked at 80000 EUR.",
        "80000 EUR",
    ),
    Turn(9, "Con ese margen, añade integración con DocuSign.", "DocuSign"),
    Turn(10, "Hace falta exportación masiva a PDF/A.", "PDF/A"),
)


def build_contradiction() -> Scenario:
    """Presupuesto 30k (turno 3) contradicho por 80k (turno 8).

    El ``anchor_fact`` es el valor *original*: medir su supervivencia dice si el
    resumen/anclas preservan el dato viejo o si la metadata lo actualiza.
    """
    return Scenario(
        name="contradiction",
        turns=_CONTRADICTION_TURNS,
        anchor_fact="30000 EUR",
        anchor_turn=3,
        project_type="internal_tool",
        description="Proyecto que se contradice: presupuesto 30k (turno 3) vs 80k (turno 8).",
    )


def all_scenarios() -> tuple[Scenario, ...]:
    """Los tres perfiles canónicos, en orden estable."""
    return (build_growing(), build_pivot(), build_contradiction())


def select_scenarios(names: list[str], *, growing_turns: int = 20) -> tuple[Scenario, ...]:
    """Filtra los perfiles por nombre (para el flag ``--scenarios`` del runner)."""
    builders = {
        "growing": lambda: build_growing(growing_turns),
        "pivot": build_pivot,
        "contradiction": build_contradiction,
    }
    unknown = [name for name in names if name not in builders]
    if unknown:
        raise ValueError(f"Escenarios desconocidos: {unknown}. Válidos: {sorted(builders)}")
    return tuple(builders[name]() for name in names)
