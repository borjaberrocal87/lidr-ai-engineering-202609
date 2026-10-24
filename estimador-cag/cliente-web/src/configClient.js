import { EstimatorError, UnavailableError } from "./estimatorClient.js";

export class RuntimeConfigError extends EstimatorError {}

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

export class HttpConfigClient {
  constructor({ baseUrl, timeoutMs = 3000, cacheTtlMs = 15000 }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
    this.cacheTtlMs = cacheTtlMs;
    this.cached = null;
    this.cachedAt = 0;
    this.primaryValue = null;
    this.primaryCheckedAt = 0;
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
      throw new UnavailableError("No se pudo contactar con la API de configuración.", {
        reason: error.name,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  async getModels() {
    const response = await this.#request("/api/v1/config/models");
    if (!response.ok) {
      const { message } = await readDetail(response);
      throw new RuntimeConfigError(message || "No se pudo leer la configuración.", {
        status: response.status,
      });
    }
    return response.json();
  }

  async getModelsCached() {
    const now = Date.now();
    if (this.cached && now - this.cachedAt < this.cacheTtlMs) {
      return this.cached;
    }
    const payload = await this.getModels();
    this.cached = payload;
    this.cachedAt = now;
    return payload;
  }

  async getPrimaryModel() {
    try {
      const payload = await this.getModelsCached();
      return payload?.models?.PRIMARY_MODEL?.effective || null;
    } catch {
      return null;
    }
  }

  invalidate() {
    this.cached = null;
    this.cachedAt = 0;
    this.primaryValue = null;
    this.primaryCheckedAt = 0;
  }

  async updateModels(models) {
    const response = await this.#request("/api/v1/config/models", {
      method: "PUT",
      body: { models },
    });
    if (response.ok) {
      this.invalidate();
      return response.json();
    }
    const { message, reason } = await readDetail(response);
    throw new RuntimeConfigError(message || `Error ${response.status} al guardar.`, {
      status: response.status,
      reason,
    });
  }
}
