import { Router } from "express";

import { MODEL_KEYS, MODEL_KNOBS } from "../aiSettings.js";
import { RuntimeConfigError } from "../configClient.js";
import { UnavailableError } from "../estimatorClient.js";

function readChanges(body) {
  const changes = {};
  for (const key of MODEL_KEYS) {
    const value = String(body[key] || "").trim();
    changes[key] = value || null;
  }
  return changes;
}

async function loadCatalog(config) {
  try {
    return { catalog: await config.getModels(), error: "" };
  } catch (error) {
    return { catalog: null, error: error.message };
  }
}

export function aiSettingsRouter({ config }) {
  const router = Router();

  router.get("/ai-settings", async (req, res, next) => {
    try {
      const { catalog, error } = await loadCatalog(config);
      res.render("ai-settings/show", {
        knobs: MODEL_KNOBS,
        catalog,
        error,
        notice: req.query.ok ? "Modelos actualizados. El cambio aplica en la siguiente llamada." : "",
      });
    } catch (error) {
      next(error);
    }
  });

  router.post("/ai-settings", async (req, res, next) => {
    const changes = readChanges(req.body);
    try {
      await config.updateModels(changes);
      res.redirect(303, "/ai-settings?ok=1");
    } catch (error) {
      if (error instanceof RuntimeConfigError || error instanceof UnavailableError) {
        const { catalog } = await loadCatalog(config);
        const status = error.status === 400 || error.status === 422 ? 422 : 503;
        res.status(status).render("ai-settings/show", {
          knobs: MODEL_KNOBS,
          catalog,
          error: `No se pudo guardar: ${error.message}`,
          notice: "",
        });
        return;
      }
      next(error);
    }
  });

  return router;
}
