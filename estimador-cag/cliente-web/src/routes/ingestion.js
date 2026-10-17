import { Router } from "express";

import { UnavailableError } from "../estimatorClient.js";
import { IngestionError } from "../ingestionClient.js";

async function loadOverview(ingestion) {
  try {
    const [catalog, jobs] = await Promise.all([
      ingestion.listSources(),
      ingestion.listJobs({ limit: 10 }),
    ]);
    return {
      version: catalog.version,
      sources: catalog.sources,
      jobs: jobs.items,
      error: "",
    };
  } catch (error) {
    return {
      version: "",
      sources: [],
      jobs: [],
      error: error.message,
    };
  }
}

export function ingestionRouter({ ingestion }) {
  const router = Router();

  router.get("/ingestion", async (req, res, next) => {
    try {
      res.render("ingestion", await loadOverview(ingestion));
    } catch (error) {
      next(error);
    }
  });

  router.post("/ingestion/runs", async (req, res, next) => {
    const sourceName = (req.body.source_name || "").trim();
    try {
      if (!sourceName) {
        const overview = await loadOverview(ingestion);
        res.status(422).render("ingestion", {
          ...overview,
          error: "Selecciona una fuente del catálogo.",
        });
        return;
      }
      const job = await ingestion.startRun(sourceName);
      res.redirect(303, `/ingestion/jobs/${job.job_id}`);
    } catch (error) {
      if (error instanceof IngestionError || error instanceof UnavailableError) {
        const overview = await loadOverview(ingestion);
        const status = error.status === 404 ? 404 : 400;
        res.status(status).render("ingestion", {
          ...overview,
          error: error.message,
        });
        return;
      }
      next(error);
    }
  });

  router.get("/ingestion/jobs/:id", async (req, res, next) => {
    try {
      const job = await ingestion.getJob(req.params.id);
      res.render("ingestion-job", { job });
    } catch (error) {
      if (error instanceof IngestionError && error.status === 404) {
        res.status(404).render("not-found");
        return;
      }
      next(error);
    }
  });

  return router;
}
