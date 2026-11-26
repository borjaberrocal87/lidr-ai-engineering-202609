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
 *
 * Tres superficies: el shot autónomo (``estimateFromTranscript``), la fase 1 del
 * wizard (``proposeStructure``) y la fase 2 híbrida (``estimateTaskHours``). Las
 * dos últimas exigen la API key de estimación, igual que sus hermanas
 * deterministas.
 */
export class HttpAgentEstimationClient {
  constructor({ baseUrl, timeoutMs = 600000, estimateApiKey = "" }) {
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
      throw new AgentEstimationError(message || `Error ${response.status} del servicio IA.`, {
        status: response.status,
      });
    }
    return response.json();
  }

  async estimateFromTranscript(transcript) {
    return this.#post("/api/v1/agent/estimate/from-transcript", { transcript });
  }

  /** Fase 1: el agente propone la estructura del brief (con overrides del perfil). */
  async proposeStructure(brief, overrides = null) {
    const body = { query: brief };
    if (overrides && Object.keys(overrides).length) body.agent = overrides;
    return this.#post("/api/v1/agent/estimate/structure", body);
  }

  /** Fase 2: horas híbridas sobre la estructura revisada. */
  async estimateTaskHours(modules, overrides = null) {
    const body = { modules };
    if (overrides && Object.keys(overrides).length) body.agent = overrides;
    return this.#post("/api/v1/agent/estimate/hours", body);
  }
}
