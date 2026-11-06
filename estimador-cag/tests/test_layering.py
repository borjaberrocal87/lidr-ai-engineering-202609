"""Contrato ejecutable de la arquitectura por capas (CAG / RAG / Agentic).

Estas reglas no son un acuerdo verbal: si una capa importa algo que no le
corresponde, el test rompe. El objetivo es que el proyecto no vuelva a
degenerar en una sucesión de carpetas acopladas.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"

DOMAIN_SERVICES = {
    "app.domain.estimation_service",
    "app.domain.streaming_service",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            if node.module == "app":
                modules.update(f"app.{alias.name}" for alias in node.names)
    return modules


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _layer_violations(
    root: Path,
    *,
    forbidden: tuple[str, ...],
) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        bad = sorted(
            module
            for module in _imported_modules(path)
            if any(_matches(module, prefix) for prefix in forbidden)
        )
        if bad:
            violations[str(path.relative_to(APP))] = bad
    return violations


def test_foundation_does_not_import_upper_layers() -> None:
    violations = _layer_violations(
        APP / "foundation",
        forbidden=(
            "app.generation",
            "app.api",
            "app.ingestion",
            *DOMAIN_SERVICES,
        ),
    )
    assert not violations, f"foundation no puede mirar hacia arriba: {violations}"


def test_domain_contracts_are_pure() -> None:
    violations: dict[str, list[str]] = {}
    for package in (APP / "domain" / "schemas", APP / "domain" / "ports"):
        violations.update(
            _layer_violations(
                package,
                forbidden=(
                    "app.foundation",
                    "app.generation",
                    "app.api",
                    "app.ingestion",
                    *DOMAIN_SERVICES,
                ),
            )
        )
    assert not violations, f"los contratos de dominio deben ser puros: {violations}"


def test_generation_siblings_do_not_import_each_other() -> None:
    violations: dict[str, list[str]] = {}
    generation = APP / "generation"

    for path in generation.rglob("*.py"):
        current = path.relative_to(generation).parts[0]
        bad: list[str] = []
        for module in _imported_modules(path):
            if not _matches(module, "app.generation"):
                continue
            imported = module.split("app.generation.", 1)[1].split(".", 1)[0]
            if imported == current:
                continue
            if current == "agentic" and imported == "conversation":
                continue
            bad.append(module)
        if bad:
            violations[str(path.relative_to(APP))] = sorted(bad)

    assert not violations, f"los hermanos de generation se tocan solo en el conductor: {violations}"


def test_ingestion_does_not_import_api_or_conductor() -> None:
    violations = _layer_violations(
        APP / "ingestion",
        forbidden=("app.api", *DOMAIN_SERVICES),
    )
    assert not violations, f"ingestion es una capa offline: {violations}"


def test_only_the_composition_root_and_bridges_import_dependencies() -> None:
    allowed = {
        APP / "main.py",
        APP / "dependencies.py",
        APP / "domain" / "estimation_service.py",
        APP / "domain" / "streaming_service.py",
    }
    for path in (APP / "api").rglob("*.py"):
        allowed.add(path)

    violations: dict[str, list[str]] = {}
    for path in APP.rglob("*.py"):
        if path in allowed:
            continue
        if "app.dependencies" in _imported_modules(path):
            violations[str(path.relative_to(APP))] = ["app.dependencies"]

    assert not violations, f"solo api/main y los conductores pueden usar dependencies: {violations}"
