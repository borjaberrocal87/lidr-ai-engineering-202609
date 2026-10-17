import { EstimatorError, UnavailableError } from "./estimatorClient.js";

export class IngestionError extends EstimatorError {}

async function readDetail(response) {
  try {
    const payload = await response.json();
    const detail = payload?.detail ?? payload;
    if (detail && typeof detail === "object") {
      return {
        message: detail.message || "",
        reason: detail.reason || "",
        decision: detail.decision || "",
      };
    }
    return { message: String(detail ?? ""), reason: "", decision: "" };
  } catch {
    return { message: response.statusText, reason: "", decision: "" };
  }
}

export class HttpIngestionClient {
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

  async listSources({ decision } = {}) {
    const query = decision ? `?decision=${encodeURIComponent(decision)}` : "";
    const response = await this.#request(`/api/v1/ingestion/sources${query}`);
    if (!response.ok) {
      throw new IngestionError("No se pudo cargar el catálogo de fuentes.", {
        status: response.status,
      });
    }
    return response.json();
  }

  async startRun(sourceName) {
    const response = await this.#request("/api/v1/ingestion/runs", {
      method: "POST",
      body: { source_name: sourceName },
    });
    if (response.ok) {
      return response.json();
    }

    const detail = await readDetail(response);
    if (response.status === 404) {
      throw new IngestionError(detail.message || "La fuente no existe en el catálogo.", {
        status: 404,
        reason: "unknown_source",
      });
    }
    if (response.status === 400) {
      throw new IngestionError(
        detail.message || "La fuente no está incluida en el catálogo.",
        { status: 400, reason: detail.reason || "source_not_included" },
      );
    }
    throw new IngestionError(`Error ${response.status} al lanzar la ingesta.`, {
      status: response.status,
    });
  }

  async getJob(jobId) {
    const response = await this.#request(
      `/api/v1/ingestion/jobs/${encodeURIComponent(jobId)}`,
    );
    if (response.status === 404) {
      throw new IngestionError("El job no existe.", { status: 404, reason: "not_found" });
    }
    if (!response.ok) {
      throw new IngestionError(`Error ${response.status} al consultar el job.`, {
        status: response.status,
      });
    }
    return response.json();
  }

  async listJobs({ limit, offset } = {}) {
    const params = new URLSearchParams();
    if (limit !== undefined) {
      params.set("limit", String(limit));
    }
    if (offset !== undefined) {
      params.set("offset", String(offset));
    }
    const query = params.toString() ? `?${params.toString()}` : "";
    const response = await this.#request(`/api/v1/ingestion/jobs${query}`);
    if (!response.ok) {
      throw new IngestionError(`Error ${response.status} al listar los jobs.`, {
        status: response.status,
      });
    }
    return response.json();
  }
}
