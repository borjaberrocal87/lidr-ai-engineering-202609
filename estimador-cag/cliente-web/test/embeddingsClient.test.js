import assert from "node:assert/strict";
import test from "node:test";

import { EmbeddingsError, HttpEmbeddingsClient } from "../src/embeddingsClient.js";
import { UnavailableError } from "../src/estimatorClient.js";

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

const client = () =>
  new HttpEmbeddingsClient({ baseUrl: "http://api.test", timeoutMs: 1000 });

test("compareChunking sends the payload and returns the response", async () => {
  let body = null;
  global.fetch = async (url, options) => {
    body = JSON.parse(options.body);
    return jsonResponse(200, { stats_per_strategy: {}, queries_per_strategy: {} });
  };

  const result = await client().compareChunking({
    budgets: [],
    queries: ["q"],
    strategies: ["structural"],
    topK: 2,
  });

  assert.equal(body.top_k, 2);
  assert.deepEqual(body.strategies, ["structural"]);
  assert.deepEqual(result, { stats_per_strategy: {}, queries_per_strategy: {} });
});

test("compareChunking maps a 400 to EmbeddingsError", async () => {
  global.fetch = async () => jsonResponse(400, { detail: "estrategia desconocida" });

  await assert.rejects(
    client().compareChunking({ budgets: [], strategies: ["nope"] }),
    (error) => {
      assert.ok(error instanceof EmbeddingsError);
      assert.equal(error.status, 400);
      return true;
    },
  );
});

test("compareChunking maps a network failure to UnavailableError", async () => {
  global.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  await assert.rejects(
    client().compareChunking({ budgets: [], strategies: ["structural"] }),
    UnavailableError,
  );
});
