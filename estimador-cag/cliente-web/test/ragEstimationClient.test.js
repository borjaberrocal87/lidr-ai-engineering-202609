import assert from "node:assert/strict";
import test from "node:test";

import {
  HttpRagEstimationClient,
  RagAuthError,
  RagInputError,
  RagPipelineError,
  RagRateLimitError,
  RagUnavailableError,
} from "../src/ragEstimationClient.js";

const originalFetch = global.fetch;

test.afterEach(() => {
  global.fetch = originalFetch;
});

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function client() {
  return new HttpRagEstimationClient({
    baseUrl: "http://api.test",
    retrievalApiKey: "retrieval-key",
    estimateApiKey: "estimate-key",
    timeoutMs: 1000,
    generateTimeoutMs: 2000,
  });
}

test("brief posts to the estimate stage with the estimate key", async () => {
  let seen;
  global.fetch = async (url, options) => {
    seen = { url, options };
    return jsonResponse(200, { brief: { functionality: "x" }, retrieval_text: "x" });
  };

  const body = await client().brief("transcripción");

  assert.equal(seen.url, "http://api.test/api/v1/estimate/stages/brief");
  assert.equal(seen.options.headers["X-API-Key"], "estimate-key");
  assert.deepEqual(JSON.parse(seen.options.body), { transcript: "transcripción" });
  assert.equal(body.retrieval_text, "x");
});

test("retrieve uses the retrieval endpoint and key", async () => {
  let seen;
  global.fetch = async (url, options) => {
    seen = { url, options };
    return jsonResponse(200, { chunks: [], low_confidence: true, candidates_evaluated: 0 });
  };

  await client().retrieve({ query_text: "pagos", top_k: 5 });

  assert.equal(seen.url, "http://api.test/api/v1/retrieval/search");
  assert.equal(seen.options.headers["X-API-Key"], "retrieval-key");
});

test("maps 401 to RagAuthError", async () => {
  global.fetch = async () => jsonResponse(401, { detail: "no" });

  await assert.rejects(client().brief("x"), RagAuthError);
});

test("maps 422 to RagInputError", async () => {
  global.fetch = async () => jsonResponse(422, { detail: "inválido" });

  await assert.rejects(client().brief("x"), RagInputError);
});

test("maps 429 to RagRateLimitError", async () => {
  global.fetch = async () => jsonResponse(429, { detail: "límite" });

  await assert.rejects(client().brief("x"), RagRateLimitError);
});

test("maps 502 to RagPipelineError", async () => {
  global.fetch = async () => jsonResponse(502, { detail: "pipeline" });

  await assert.rejects(client().brief("x"), RagPipelineError);
});

test("maps a network failure to RagUnavailableError", async () => {
  global.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  await assert.rejects(client().brief("x"), RagUnavailableError);
});
