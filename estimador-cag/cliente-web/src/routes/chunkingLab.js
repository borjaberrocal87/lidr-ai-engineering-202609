import { performance } from "node:perf_hooks";

import { Router } from "express";

import {
  CHUNK_STRATEGIES,
  DEFAULT_STRATEGIES,
  STRATEGY_BY_NAME,
  isPaidStrategy,
} from "../chunkingConstants.js";
import {
  CORPUS_LABEL,
  budgetComponentCount,
  loadBudgets,
  loadQueries,
} from "../corpus.js";
import { EmbeddingsError } from "../embeddingsClient.js";
import { UnavailableError } from "../estimatorClient.js";

function readStrategies(body) {
  const raw = body.strategies;
  const values = Array.isArray(raw) ? raw : raw ? [raw] : [];
  return values.filter((name) => STRATEGY_BY_NAME[name]);
}

function readQueries(body) {
  return [...new Set(
    String(body.queries || "")
      .split(/\r?\n|,/)
      .map((query) => query.trim())
      .filter(Boolean),
  )];
}

function readTopK(body) {
  const value = Number.parseInt(body.top_k, 10);
  if (Number.isNaN(value)) {
    return 3;
  }
  return Math.min(10, Math.max(1, value));
}

function baseContext() {
  const budgets = loadBudgets();
  return {
    strategies: CHUNK_STRATEGIES,
    defaultStrategies: DEFAULT_STRATEGIES,
    corpusLabel: CORPUS_LABEL,
    corpusCount: budgets.length,
    componentCount: budgetComponentCount(budgets),
    sampleQueries: loadQueries(),
  };
}

function newContext(overrides = {}) {
  return {
    ...baseContext(),
    selectedStrategies: DEFAULT_STRATEGIES,
    queriesText: "",
    topK: 3,
    error: "",
    ...overrides,
  };
}

export function chunkingLabRouter({ chunkingRepo, embeddings }) {
  const router = Router();

  router.get("/chunking-lab", async (req, res, next) => {
    try {
      const comparisons = await chunkingRepo.list(20);
      res.render("chunking-lab/index", { comparisons });
    } catch (error) {
      next(error);
    }
  });

  router.get("/chunking-lab/new", (req, res) => {
    res.render("chunking-lab/new", newContext());
  });

  router.post("/chunking-lab", async (req, res, next) => {
    const selectedStrategies = readStrategies(req.body);
    const queries = readQueries(req.body);
    const topK = readTopK(req.body);
    const context = newContext({
      selectedStrategies,
      queriesText: queries.join("\n"),
      topK,
    });

    if (selectedStrategies.length === 0) {
      res.status(422).render("chunking-lab/new", {
        ...context,
        error: "Selecciona al menos una estrategia.",
      });
      return;
    }

    const budgets = loadBudgets();
    const started = performance.now();
    try {
      const payload = await embeddings.compareChunking({
        budgets,
        queries,
        strategies: selectedStrategies,
        topK,
      });
      const durationMs = Math.round(performance.now() - started);
      const id = await chunkingRepo.create({
        strategies: selectedStrategies,
        queries,
        topK,
        corpusLabel: CORPUS_LABEL,
        corpusCount: budgets.length,
        responsePayload: payload,
        durationMs,
      });
      res.redirect(303, `/chunking-lab/${id}`);
    } catch (error) {
      if (error instanceof EmbeddingsError || error instanceof UnavailableError) {
        const status = error.status === 400 || error.status === 422 ? 422 : 502;
        const paidHint = selectedStrategies.some(isPaidStrategy)
          ? " Las estrategias de pago pueden tardar varios minutos; prueba con menos estrategias."
          : "";
        res.status(status).render("chunking-lab/new", {
          ...context,
          error: `${error.message}${paidHint}`,
        });
        return;
      }
      next(error);
    }
  });

  router.get("/chunking-lab/:id", async (req, res, next) => {
    try {
      const comparison = await chunkingRepo.getById(req.params.id);
      if (!comparison) {
        res.status(404).render("not-found");
        return;
      }
      res.render("chunking-lab/show", {
        comparison,
        strategies: CHUNK_STRATEGIES,
        strategyLabel: (name) => STRATEGY_BY_NAME[name]?.label || name,
      });
    } catch (error) {
      next(error);
    }
  });

  return router;
}
