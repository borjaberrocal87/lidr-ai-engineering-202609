import { EstimatorError, UnavailableError } from "./estimatorClient.js";

export class AgentEstimationError extends EstimatorError {}

async function readDetail(response) {
  try {
    const payload = await response.json();
    const detail = payload?.detail ?? payload;
    if (detail && typeof detail === "object") {
      return { message: detail.message || "" };
    }
    return { message: String(detail ?? "") };
  } catch {
    return { message: response.statusText };
  }
}

/**
 * Cliente de la capa agéntica (Sesión 12).
 * Habla con /api/v1/agent/estimate/from-transcript y devuelve informe + traza.
 */
export class HttpAgentEstimationClient {
  constructor({ baseUrl, timeoutMs = 600000 }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
  }

  async #post(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      throw new UnavailableError(`No se pudo contactar con el servicio IA en ${this.baseUrl}.`, {
        reason: error.name,
      });
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      const { message } = await readDetail(response);
      throw new AgentEstimationError(message || `Error ${response.status} del servicio IA.`, {
        status: response.status,
      });
    }
    return response.json();
  }

  async estimateFromTranscript(transcript) {
    return this.#post("/api/v1/agent/estimate/from-transcript", { transcript });
  }
}
