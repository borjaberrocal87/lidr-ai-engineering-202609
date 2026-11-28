import { EstimatorError, UnavailableError } from "./estimatorClient.js";

export class GraphEstimationError extends EstimatorError {}

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
 * Cliente del grafo de estimación (Sesión 13).
 *
 * Una sola superficie: ``estimateFromTranscript`` ejecuta el ``StateGraph`` sobre
 * una transcripción y devuelve la estimación con su ``status`` y el ``run_text``
 * ya renderizado por el servidor.
 */
export class HttpGraphEstimationClient {
  constructor({ baseUrl, timeoutMs = 300000, estimateApiKey = "" }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
    this.estimateApiKey = estimateApiKey;
  }

  async #post(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.estimateApiKey ? { "X-API-Key": this.estimateApiKey } : {}),
        },
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
      throw new GraphEstimationError(message || `Error ${response.status} del servicio IA.`, {
        status: response.status,
      });
    }
    return response.json();
  }

  async estimateFromTranscript(transcript, estimationId = null) {
    const body = { transcript };
    if (estimationId) body.estimation_id = estimationId;
    return this.#post("/api/v1/graph/estimate/from-transcript", body);
  }
}
