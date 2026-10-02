import { Router } from "express";

import {
  DEFAULT_PROMPT_VERSIONS,
  DESCRIPTION_MAX,
  DESCRIPTION_MIN,
  humanize,
} from "../constants.js";
import {
  ConfigError,
  GuardrailError,
  InputError,
  ProviderError,
  UnavailableError,
} from "../estimatorClient.js";

async function loadFormContext(estimator) {
  try {
    const context = await estimator.getContext();
    return {
      availableVersions: context.available_versions?.length
        ? context.available_versions
        : DEFAULT_PROMPT_VERSIONS,
      descriptionMin: context.description_min_length ?? DESCRIPTION_MIN,
      descriptionMax: context.description_max_length ?? DESCRIPTION_MAX,
    };
  } catch {
    return {
      availableVersions: DEFAULT_PROMPT_VERSIONS,
      descriptionMin: DESCRIPTION_MIN,
      descriptionMax: DESCRIPTION_MAX,
    };
  }
}

function readForm(body) {
  return {
    description: (body.description || "").trim(),
    project_type: body.project_type || "web_saas",
    detail_level: body.detail_level || "medium",
    output_format: body.output_format || "phases_table",
    prompt_version: body.prompt_version || "v1",
  };
}

export function estimationsRouter({ repo, estimator }) {
  const router = Router();

  router.get("/", async (req, res, next) => {
    try {
      const context = await loadFormContext(estimator);
      res.render("index", { context, values: {}, error: "" });
    } catch (error) {
      next(error);
    }
  });

  router.post("/estimations", async (req, res, next) => {
    const values = readForm(req.body);
    try {
      const context = await loadFormContext(estimator);
      if (values.description.length < context.descriptionMin) {
        res.status(422).render("index", {
          context,
          values,
          error: `La descripción debe tener al menos ${context.descriptionMin} caracteres.`,
        });
        return;
      }

      const response = await estimator.estimate(
        {
          description: values.description,
          project_type: values.project_type,
          detail_level: values.detail_level,
          output_format: values.output_format,
        },
        { promptVersion: values.prompt_version },
      );

      const id = await repo.create({
        projectType: values.project_type,
        detailLevel: values.detail_level,
        outputFormat: values.output_format,
        promptVersion: response.prompt_version,
        cached: Boolean(response.cached),
        description: values.description,
        result: response.result,
      });
      res.redirect(303, `/estimations/${id}`);
    } catch (error) {
      if (
        error instanceof GuardrailError ||
        error instanceof InputError ||
        error instanceof ProviderError ||
        error instanceof ConfigError ||
        error instanceof UnavailableError
      ) {
        const context = await loadFormContext(estimator);
        res.status(error instanceof GuardrailError ? 400 : 502).render("index", {
          context,
          values,
          error: error.message,
        });
        return;
      }
      next(error);
    }
  });

  router.get("/estimations", async (req, res, next) => {
    try {
      const estimations = await repo.list();
      res.render("list", { estimations });
    } catch (error) {
      next(error);
    }
  });

  router.get("/estimations/:id", async (req, res, next) => {
    try {
      const estimation = await repo.getById(req.params.id);
      if (!estimation) {
        res.status(404).render("not-found");
        return;
      }
      res.render("show", { estimation, humanize });
    } catch (error) {
      next(error);
    }
  });

  return router;
}
