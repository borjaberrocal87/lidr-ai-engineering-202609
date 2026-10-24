import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, "..", "data");

export const CORPUS_LABEL = "budgets_sample";

export function loadBudgets() {
  return JSON.parse(readFileSync(join(dataDir, "budgets_sample.json"), "utf8"));
}

export function loadQueries() {
  return JSON.parse(readFileSync(join(dataDir, "test_queries.json"), "utf8"));
}

export function budgetComponentCount(budgets) {
  return budgets.reduce((total, budget) => total + budget.components.length, 0);
}
