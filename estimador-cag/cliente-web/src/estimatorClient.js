export class EstimatorError extends Error {
  constructor(message, { status = 0, reason = "" } = {}) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
    this.reason = reason;
  }
}

export class GuardrailError extends EstimatorError {}
export class InputError extends EstimatorError {}
export class ConfigError extends EstimatorError {}
export class ProviderError extends EstimatorError {}
export class UnavailableError extends EstimatorError {}

async function readDetail(response) {
  try {
    const payload = await response.json();
    const detail = payload?.detail ?? payload;
    if (detail && typeof detail === "object") {
      return { message: detail.message || "", reason: detail.reason || "" };
    }
    return { message: String(detail ?? ""), reason: "" };
  } catch {
    return { message: response.statusText, reason: "" };
  }
}

export class HttpEstimatorClient {
  constructor({ baseUrl, timeoutMs = 60000 }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
  }

  async #request(path, { method = "GET", body } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      return await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      throw new UnavailableError(`No se pudo contactar con la API en ${this.baseUrl}.`, {
        reason: error.name,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  async getContext(promptVersion) {
    const query = promptVersion ? `?prompt_version=${encodeURIComponent(promptVersion)}` : "";
    const response = await this.#request(`/api/v1/context${query}`);
    if (!response.ok) {
      throw new ProviderError("No se pudo cargar el contexto de la API.", {
        status: response.status,
      });
    }
    return response.json();
  }

  async estimate(payload, { promptVersion } = {}) {
    const query = promptVersion ? `?prompt_version=${encodeURIComponent(promptVersion)}` : "";
    const response = await this.#request(`/api/v1/estimate${query}`, {
      method: "POST",
      body: payload,
    });

    if (response.ok) {
      return response.json();
    }

    const { message, reason } = await readDetail(response);
    const status = response.status;
    if (status === 400) {
      throw new GuardrailError(message || "Entrada rechazada por los guardrails.", {
        status,
        reason,
      });
    }
    if (status === 422) {
      throw new InputError(message || "La entrada no es válida.", { status, reason });
    }
    if (status === 503) {
      throw new ConfigError(message || "El proveedor LLM no está configurado.", { status });
    }
    if (status === 502) {
      throw new ProviderError(message || "El proveedor LLM falló.", { status });
    }
    throw new EstimatorError(`Error ${status} de la API.`, { status });
  }
}
