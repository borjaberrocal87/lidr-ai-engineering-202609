import { Router } from "express";

import { CorpusIndexError } from "../corpusIndexClient.js";
import { UnavailableError } from "../estimatorClient.js";

const EXAMPLE_JSON = JSON.stringify(
  [
    {
      budget_id: "BUDGET-2026-9001",
      client_metadata: { name: "ACME", sector: "finance", country: "ES" },
      project_summary: "Portal de pagos B2B con conciliación",
      main_technology: "python_fastapi",
      year: 2026,
      total_estimated_hours: 120,
      components: [
        {
          component_id: "PAY-001",
          name: "Payment reconciliation job",
          description: "Conciliación nocturna entre procesador y pedidos.",
          module: "Payments & Billing",
          tech_stack: ["python", "postgresql"],
          estimated_hours: 120,
          complexity: "medium",
          dependencies: [],
        },
      ],
    },
  ],
  null,
  2,
);

function parseDocuments(raw) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    return { error: `JSON inválido: ${error.message}` };
  }
  const documents = Array.isArray(parsed) ? parsed : [parsed];
  if (!documents.length) {
    return { error: "El lote está vacío: pega al menos un documento." };
  }
  if (!documents.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
    return { error: "Cada documento del lote debe ser un objeto JSON." };
  }
  return { documents };
}

async function safeStats(client) {
  try {
    return await client.getStats();
  } catch {
    return null;
  }
}

export function corpusIndexRouter({ corpusIndex, corpusRuns }) {
  const router = Router();

  router.get("/corpus", async (req, res, next) => {
    try {
      const [stats, runs] = await Promise.all([
        safeStats(corpusIndex),
        corpusRuns.list(20),
      ]);
      res.render("corpus/index", { stats, runs, error: "" });
    } catch (error) {
      next(error);
    }
  });

  router.get("/corpus/new", (req, res) => {
    res.render("corpus/new", {
      values: { documents_json: EXAMPLE_JSON, chunk_type: "budget_component" },
      error: "",
    });
  });

  router.post("/corpus/runs", async (req, res, next) => {
    const raw = String(req.body.documents_json || "");
    const chunkType = String(req.body.chunk_type || "budget_component");
    const { documents, error } = parseDocuments(raw);
    if (error) {
      res.status(422).render("corpus/new", {
        values: { documents_json: raw, chunk_type: chunkType },
        error,
      });
      return;
    }

    try {
      const beforeStats = await safeStats(corpusIndex);
      const job = await corpusIndex.startRun({ documents, chunkType });
      const id = await corpusRuns.create({
        jobId: job.job_id,
        chunkType,
        documentsTotal: job.documents_total,
        status: job.status,
        beforeStats,
      });
      res.redirect(303, `/corpus/runs/${id}`);
    } catch (error) {
      if (error instanceof CorpusIndexError || error instanceof UnavailableError) {
        const status = error.status === 404 ? 404 : 400;
        res.status(status).render("corpus/new", {
          values: { documents_json: raw, chunk_type: chunkType },
          error: error.message,
        });
        return;
      }
      next(error);
    }
  });

  router.get("/corpus/runs/:id", async (req, res, next) => {
    try {
      const run = await corpusRuns.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }

      let job = null;
      let jobError = "";
      try {
        job = await corpusIndex.getJob(run.jobId);
        // Refleja el estado final del job en el historial local.
        if (job.status === "completed" || job.status === "failed") {
          await corpusRuns.update(run.id, { status: job.status, error: job.error_message });
          run.status = job.status;
          run.error = job.error_message;
        }
      } catch (error) {
        if (error instanceof CorpusIndexError && error.status !== 404) {
          jobError = error.message;
        }
      }
      const stats = await safeStats(corpusIndex);

      res.render("corpus/show", { run, job, jobError, stats });
    } catch (error) {
      next(error);
    }
  });

  return router;
}
