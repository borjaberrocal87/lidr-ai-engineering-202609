import { Router } from "express";

import { MODEL_KEYS, MODEL_KNOBS } from "../aiSettings.js";
import { RuntimeConfigError } from "../configClient.js";
import { UnavailableError } from "../estimatorClient.js";

const QUALITY_TOGGLES = [
  {
    field: "hallucination_gate_enabled",
    key: "HALLUCINATION_GATE_ENABLED",
    label: "Gate semántico de alucinaciones",
    description:
      "Comprueba que la cifra esté respaldada por la evidencia citada, no solo que la cita exista.",
  },
  {
    field: "augmentation_enabled",
    key: "AUGMENTATION_ENABLED",
    label: "Augmentación del contexto",
    description:
      "Comprime cada fuente a sus líneas clave y coloca las mejores al principio y al final.",
  },
  {
    field: "synthesis_enabled",
    key: "SYNTHESIS_ENABLED",
    label: "Síntesis de horas con rango",
    description:
      "Si los análogos históricos discrepan, añade un rango honesto junto al punto.",
  },
];

function readChanges(body) {
  const changes = {};
  for (const key of MODEL_KEYS) {
    const value = String(body[key] || "").trim();
    changes[key] = value || null;
  }
  return changes;
}

function readQualityChanges(body) {
  const changes = {};
  for (const toggle of QUALITY_TOGGLES) {
    changes[toggle.field] = body[toggle.field] === "on";
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

async function loadRetrieval(config) {
  try {
    return { retrieval: await config.getRetrieval() };
  } catch {
    return { retrieval: null };
  }
}

export function aiSettingsRouter({ config }) {
  const router = Router();

  router.get("/ai-settings", async (req, res, next) => {
    try {
      const { catalog, error } = await loadCatalog(config);
      const { retrieval } = await loadRetrieval(config);
      const notices = {
        "1": "Modelos actualizados. El cambio aplica en la siguiente llamada.",
        toggles: "Calidad de generación actualizada. Aplica en la siguiente petición.",
      };
      res.render("ai-settings/show", {
        knobs: MODEL_KNOBS,
        qualityToggles: QUALITY_TOGGLES,
        catalog,
        retrieval,
        error,
        notice: notices[req.query.ok] || "",
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
        const { retrieval } = await loadRetrieval(config);
        const status = error.status === 400 || error.status === 422 ? 422 : 503;
        res.status(status).render("ai-settings/show", {
          knobs: MODEL_KNOBS,
          qualityToggles: QUALITY_TOGGLES,
          catalog,
          retrieval,
          error: `No se pudo guardar: ${error.message}`,
          notice: "",
        });
        return;
      }
      next(error);
    }
  });

  router.post("/ai-settings/quality", async (req, res, next) => {
    try {
      await config.updateRetrieval(readQualityChanges(req.body));
      res.redirect(303, "/ai-settings?ok=toggles");
    } catch (error) {
      if (error instanceof RuntimeConfigError || error instanceof UnavailableError) {
        const { catalog } = await loadCatalog(config);
        const { retrieval } = await loadRetrieval(config);
        const status = error.status === 400 || error.status === 422 ? 422 : 503;
        res.status(status).render("ai-settings/show", {
          knobs: MODEL_KNOBS,
          qualityToggles: QUALITY_TOGGLES,
          catalog,
          retrieval,
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
