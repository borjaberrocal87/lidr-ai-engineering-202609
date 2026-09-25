"""Errores del dominio LLM compartidos por el servicio y el wrapper.

Viven en un módulo propio para que `llm_wrapper` pueda traducir los fallos del
proveedor sin depender de `llm_service` (y evitar un import circular).
"""


class LLMConfigurationError(RuntimeError):
    """Error de configuración del proveedor LLM (API key ausente, etc.)."""


class LLMProviderError(RuntimeError):
    """Error devuelto por el proveedor LLM (red, timeout, rate limit, estado)."""


class LLMInputError(RuntimeError):
    """La entrada no cumple las restricciones de longitud del servicio."""
