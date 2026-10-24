import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import express from "express";

import { DETAIL_LEVELS, OUTPUT_FORMATS, PROJECT_TYPES, humanize } from "./constants.js";
import { aiSettingsRouter } from "./routes/aiSettings.js";
import { chunkingLabRouter } from "./routes/chunkingLab.js";
import { estimationsRouter } from "./routes/estimations.js";
import { ingestionRouter } from "./routes/ingestion.js";

const here = dirname(fileURLToPath(import.meta.url));

export function createApp({ repo, estimator, ingestion, chunkingRepo, embeddings, config }) {
  const app = express();

  app.set("view engine", "ejs");
  app.set("views", join(here, "..", "views"));
  app.use(express.urlencoded({ extended: false }));
  app.use(express.static(join(here, "..", "public")));

  app.locals.humanize = humanize;
  app.locals.projectTypes = PROJECT_TYPES;
  app.locals.detailLevels = DETAIL_LEVELS;
  app.locals.outputFormats = OUTPUT_FORMATS;
  app.locals.formatDate = (value) =>
    new Date(value).toISOString().slice(0, 16).replace("T", " ");
  app.locals.year = new Date().getFullYear();

  app.get("/health", (req, res) => res.json({ status: "ok", service: "cliente-web" }));

  app.use(async (req, res, next) => {
    res.locals.primaryModel = config ? await config.getPrimaryModel() : null;
    next();
  });

  app.use(estimationsRouter({ repo, estimator }));
  app.use(ingestionRouter({ ingestion }));
  if (chunkingRepo && embeddings) {
    app.use(chunkingLabRouter({ chunkingRepo, embeddings }));
  }
  if (config) {
    app.use(aiSettingsRouter({ config }));
  }

  app.use((req, res) => res.status(404).render("not-found"));
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    console.error("Unhandled error:", err);
    res.status(500).render("error", { message: "Se produjo un error inesperado." });
  });

  return app;
}
