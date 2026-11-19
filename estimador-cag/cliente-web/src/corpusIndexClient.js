import { EstimatorError, UnavailableError } from "./estimatorClient.js";

export class CorpusIndexError extends EstimatorError {}

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
 * Cliente de la expansión incremental del corpus (Sesión 11).
 * Habla con /api/v1/embeddings/index/*: alta de lote, polling de job y stats.
 */
export class HttpCorpusIndexClient {
  constructor({ baseUrl, timeoutMs = 10000 }) {
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

  async startRun({ documents, chunkType }) {
    const response = await this.#request("/api/v1/embeddings/index/runs", {
      method: "POST",
      body: { documents, chunk_type: chunkType },
    });
    if (response.ok) {
      return response.json();
    }
    const detail = await readDetail(response);
    throw new CorpusIndexError(
      detail.message || `Error ${response.status} al lanzar la expansión.`,
      { status: response.status },
    );
  }

  async getJob(jobId) {
    const response = await this.#request(
      `/api/v1/embeddings/index/jobs/${encodeURIComponent(jobId)}`,
    );
    if (response.status === 404) {
      throw new CorpusIndexError("El job no existe.", { status: 404 });
    }
    if (!response.ok) {
      throw new CorpusIndexError(`Error ${response.status} al consultar el job.`, {
        status: response.status,
      });
    }
    return response.json();
  }

  async getStats() {
    const response = await this.#request("/api/v1/embeddings/index/stats");
    if (!response.ok) {
      throw new CorpusIndexError(`Error ${response.status} al leer las estadísticas.`, {
        status: response.status,
      });
    }
    return response.json();
  }
}
