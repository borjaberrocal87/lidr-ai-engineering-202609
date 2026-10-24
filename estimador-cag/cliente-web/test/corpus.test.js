import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { budgetComponentCount, loadBudgets, loadQueries } from "../src/corpus.js";

const here = dirname(fileURLToPath(import.meta.url));

test("el corpus del cliente coincide con el del backend", () => {
  const backend = JSON.parse(
    readFileSync(join(here, "..", "..", "data", "budgets_sample.json"), "utf8"),
  );
  const backendQueries = JSON.parse(
    readFileSync(join(here, "..", "..", "data", "test_queries.json"), "utf8"),
  );

  assert.deepEqual(loadBudgets(), backend);
  assert.deepEqual(loadQueries(), backendQueries);
});

test("el corpus tiene 17 presupuestos y 60 componentes", () => {
  const budgets = loadBudgets();

  assert.equal(budgets.length, 17);
  assert.equal(budgetComponentCount(budgets), 60);
  assert.equal(loadQueries().length, 6);
});
