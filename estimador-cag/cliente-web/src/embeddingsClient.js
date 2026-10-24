import { EstimatorError, UnavailableError } from "./estimatorClient.js";

export class EmbeddingsError extends EstimatorError {}

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

export class HttpEmbeddingsClient {
  constructor({ baseUrl, timeoutMs = 600000 }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
  }

  async compareChunking({ budgets, queries = [], strategies = [], topK = 3 }) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/embeddings/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          budgets,
          queries,
          strategies,
          top_k: topK,
        }),
        signal: controller.signal,
      });
      if (response.ok) {
        return response.json();
      }
      const { message, reason } = await readDetail(response);
      throw new EmbeddingsError(message || `Error ${response.status} en la comparativa.`, {
        status: response.status,
        reason,
      });
    } catch (error) {
      if (error instanceof EmbeddingsError) {
        throw error;
      }
      throw new UnavailableError(
        `No se pudo completar la comparativa (${error.name}). Revisa que la API siga viva.`,
        { reason: error.name },
      );
    } finally {
      clearTimeout(timer);
    }
  }
}
