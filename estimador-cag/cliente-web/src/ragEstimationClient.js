export class RagClientError extends Error {
  constructor(message, { status = 0, reason = "" } = {}) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
    this.reason = reason;
  }
}

export class RagAuthError extends RagClientError {}
export class RagRateLimitError extends RagClientError {}
export class RagInputError extends RagClientError {}
export class RagPipelineError extends RagClientError {}
export class RagUnavailableError extends RagClientError {}

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

export class HttpRagEstimationClient {
  constructor({
    baseUrl,
    retrievalApiKey = "",
    estimateApiKey = "",
    timeoutMs = 120000,
    generateTimeoutMs = 300000,
  }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.retrievalApiKey = retrievalApiKey;
    this.estimateApiKey = estimateApiKey;
    this.timeoutMs = timeoutMs;
    this.generateTimeoutMs = generateTimeoutMs;
  }

  async #request(path, { body, apiKey, timeoutMs }) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { "X-API-Key": apiKey } : {}),
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      throw new RagUnavailableError("No se pudo contactar con el servicio IA.", {
        reason: error.name,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  async #post(path, body, { apiKey, timeoutMs }) {
    const response = await this.#request(path, { body, apiKey, timeoutMs });
    if (response.ok) {
      return response.json();
    }

    const { message } = await readDetail(response);
    const { status } = response;
    if (status === 401 || status === 403) {
      throw new RagAuthError(
        "El servicio rechazó la API key del wizard. Revisa RAG_*_API_KEY.",
        { status },
      );
    }
    if (status === 422) {
      throw new RagInputError(message || "La petición no es válida.", { status });
    }
    if (status === 429) {
      throw new RagRateLimitError(
        "Demasiadas peticiones al servicio IA. Espera un minuto y reinténtalo.",
        { status },
      );
    }
    if (status === 502) {
      throw new RagPipelineError(
        message || "El pipeline RAG falló. Revisa los logs del estimador.",
        { status },
      );
    }
    throw new RagClientError(`Error ${status} del servicio IA.`, { status });
  }

  async brief(transcript) {
    return this.#post(
      "/api/v1/estimate/stages/brief",
      { transcript },
      { apiKey: this.estimateApiKey, timeoutMs: this.timeoutMs },
    );
  }

  async retrieve(payload) {
    return this.#post("/api/v1/retrieval/search", payload, {
      apiKey: this.retrievalApiKey,
      timeoutMs: this.timeoutMs,
    });
  }

  async assemble(payload) {
    return this.#post("/api/v1/estimate/stages/assemble", payload, {
      apiKey: this.estimateApiKey,
      timeoutMs: this.timeoutMs,
    });
  }

  async generate(payload) {
    return this.#post("/api/v1/estimate/stages/generate", payload, {
      apiKey: this.estimateApiKey,
      timeoutMs: this.generateTimeoutMs,
    });
  }

  async estimateFromTranscript(transcript, idempotencyKey) {
    return this.#post(
      "/api/v1/estimate/from-transcript",
      { transcript, idempotency_key: idempotencyKey || null },
      { apiKey: this.estimateApiKey, timeoutMs: this.generateTimeoutMs },
    );
  }
}
