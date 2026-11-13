import { Router } from "express";

import { RagClientError } from "../ragEstimationClient.js";

const STEPS = ["brief", "generation", "review", "hours", "verification"];
// Pasos de la Sesión 9 que se conservan para el modo comparación; no aparecen
// en la barra de progreso del flujo principal.
const LEGACY_STEPS = ["retrieval", "assembly"];
const DEFAULT_RATE_EUR_PER_HOUR = 75;

function clamp(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function readList(value) {
  const raw = Array.isArray(value) ? value : String(value || "").split(",");
  return raw.map((item) => String(item).trim()).filter(Boolean);
}

function readSources(value) {
  return readList(value)
    .map((source) => Number.parseInt(source, 10))
    .filter((source) => !Number.isNaN(source));
}

function readFilters(body = {}) {
  return {
    top_k: clamp(body.top_k, 1, 30, 10),
    distance_threshold: Number.isFinite(Number(body.distance_threshold))
      ? Number(body.distance_threshold)
      : 0.6,
    sectors: readList(body.sectors),
    project_year_min: body.project_year_min ? Number(body.project_year_min) : null,
    project_year_max: body.project_year_max ? Number(body.project_year_max) : null,
    chunk_types: readList(body.chunk_types),
  };
}

function asArray(value) {
  if (value == null) return [];
  return Array.isArray(value) ? value : Object.values(value);
}

function normalizeStructure(raw) {
  return asArray(raw)
    .filter((module) => module && typeof module === "object")
    .map((module) => ({
      name: String(module.name || "").trim() || "Módulo sin nombre",
      description: String(module.description || "").trim() || null,
      tasks: asArray(module.tasks)
        .filter((task) => task && typeof task === "object")
        .map((task) => ({
          name: String(task.name || "").trim() || "Tarea sin nombre",
          description: String(task.description || "").trim() || null,
          sources: readSources(task.sources),
        })),
    }))
    .filter((module) => module.tasks.length > 0);
}

function normalizeCostModules(raw) {
  return asArray(raw)
    .filter((module) => module && typeof module === "object")
    .map((module) => ({
      name: String(module.name || "").trim() || "Módulo sin nombre",
      description: String(module.description || "").trim() || null,
      tasks: asArray(module.tasks)
        .filter((task) => task && typeof task === "object")
        .map((task) => {
          const hoursValue = Number.parseInt(task.estimated_hours, 10);
          const estimatedHours = Number.isNaN(hoursValue) ? null : Math.max(0, hoursValue);
          const rateValue = Number.parseInt(task.rate_eur_per_hour, 10);
          return {
            name: String(task.name || "").trim() || "Tarea sin nombre",
            description: String(task.description || "").trim() || null,
            estimated_hours: estimatedHours,
            rate_eur_per_hour: Number.isNaN(rateValue)
              ? DEFAULT_RATE_EUR_PER_HOUR
              : Math.max(0, rateValue),
            has_match: estimatedHours !== null,
            sources: readSources(task.sources),
          };
        }),
    }))
    .filter((module) => module.tasks.length > 0);
}

function sumHours(modules) {
  return modules.reduce(
    (total, module) =>
      total +
      module.tasks.reduce((partial, task) => partial + (task.estimated_hours || 0), 0),
    0,
  );
}

function sumCost(modules) {
  return modules.reduce(
    (total, module) =>
      total +
      module.tasks.reduce(
        (partial, task) =>
          partial + (task.estimated_hours || 0) * (task.rate_eur_per_hour || 0),
        0,
      ),
    0,
  );
}

function seedStructure(generation) {
  const estimate = generation?.estimate || {};
  return {
    modules: asArray(estimate.modules).map((module) => ({
      name: String(module.name || "").trim(),
      description: String(module.description || "").trim() || null,
      tasks: asArray(module.tasks).map((task) => ({
        name: String(task.name || "").trim(),
        description: String(task.description || "").trim() || null,
        sources: Array.isArray(task.sources) ? task.sources : [],
      })),
    })),
  };
}

function structureForApi(modules) {
  return modules.map((module) => ({
    name: module.name,
    tasks: module.tasks.map((task) => ({
      name: task.name,
      description: task.description,
    })),
  }));
}

function seedBreakdownWithHours(modules, hoursResult) {
  const lookup = new Map();
  for (const estimate of hoursResult?.tasks || []) {
    lookup.set(`${estimate.module}\u0000${estimate.task}`, estimate);
  }

  const seeded = modules.map((module) => ({
    name: module.name,
    description: module.description,
    tasks: module.tasks.map((task) => {
      const hit = lookup.get(`${module.name}\u0000${task.name}`) || {};
      const matched = Boolean(hit.has_match);
      return {
        name: task.name,
        description: task.description,
        sources: task.sources || [],
        estimated_hours: matched ? hit.estimated_hours ?? null : null,
        hours_reliability: matched ? hit.reliability ?? null : null,
        has_match: matched,
        rate_eur_per_hour: DEFAULT_RATE_EUR_PER_HOUR,
      };
    }),
  }));

  const totalHours = sumHours(seeded);
  return {
    modules: seeded,
    total_hours: totalHours,
    total_cost_eur: totalHours * DEFAULT_RATE_EUR_PER_HOUR,
    confirmed_at: null,
  };
}

function resolveStep(queryStep, run) {
  const known = [...STEPS, ...LEGACY_STEPS];
  if (known.includes(queryStep)) return queryStep;
  return known.includes(run?.currentStep) ? run.currentStep : "brief";
}

function viewContext(run, step, { error = "", notice = "" } = {}) {
  const generation = run.generation || null;
  const estimate = generation?.estimate || null;
  const structure = run.structure || seedStructure(generation);
  return {
    run,
    step,
    steps: STEPS,
    stepIndex: STEPS.indexOf(step),
    error: error || run.error || "",
    notice,
    filters: readFilters({}),
    estimate,
    structure: structure?.modules || [],
    taskHours: run.taskHours || null,
    fabricatedSourceIds: generation?.fabricated_source_ids || [],
    coherent: generation?.coherent,
    verification: run.verification,
  };
}

export function ragEstimationsRouter({ ragEstimations, ragClient }) {
  const router = Router();

  const failed = async (run, error, step) => {
    const message = error instanceof RagClientError ? error.message : "Error inesperado.";
    await ragEstimations.update(run.id, { status: "failed", currentStep: step, error: message });
    return message;
  };

  const redirectWithError = (res, runId, step, message) =>
    res.redirect(
      303,
      `/rag-estimations/${runId}?step=${step}&error=${encodeURIComponent(message)}`,
    );

  router.get("/rag-estimations", async (req, res, next) => {
    try {
      const runs = await ragEstimations.list(30);
      res.render("rag-estimations/index", { runs });
    } catch (error) {
      next(error);
    }
  });

  router.get("/rag-estimations/new", (req, res) => {
    res.render("rag-estimations/new", { values: { transcript: "" }, error: "" });
  });

  router.post("/rag-estimations", async (req, res, next) => {
    const transcript = String(req.body.transcript || "").trim();
    if (transcript.length < 100) {
      res.status(422).render("rag-estimations/new", {
        values: { transcript },
        error: "La transcripción debe tener al menos 100 caracteres.",
      });
      return;
    }

    let run;
    try {
      const id = await ragEstimations.create({
        transcript,
        idempotencyKey: req.body.idempotency_key || null,
      });
      run = await ragEstimations.getById(id);
      const outcome = await ragClient.brief(transcript);
      await ragEstimations.update(id, {
        brief: outcome.brief,
        retrievalText: outcome.retrieval_text,
        status: "brief",
        currentStep: "generation",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${id}?step=brief`);
    } catch (error) {
      if (run) {
        const message = await failed(run, error, "brief");
        redirectWithError(res, run.id, "brief", message);
        return;
      }
      res.status(502).render("rag-estimations/new", {
        values: { transcript },
        error: error instanceof RagClientError ? error.message : "No se pudo crear el run.",
      });
    }
  });

  router.get("/rag-estimations/:id", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      const step = resolveStep(req.query.step, run);
      res.render(
        "rag-estimations/show",
        viewContext(run, step, {
          error: req.query.error ? String(req.query.error) : "",
          notice: req.query.notice ? String(req.query.notice) : "",
        }),
      );
    } catch (error) {
      next(error);
    }
  });

  router.post("/rag-estimations/:id/brief", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      if (req.body.save === "1") {
        const brief = {
          functionality: String(req.body.functionality || "").trim(),
          technologies: readList(req.body.technologies),
          sector: String(req.body.sector || "").trim() || null,
          scale: String(req.body.scale || "unknown").trim() || "unknown",
          country: String(req.body.country || "").trim() || null,
          regulations: readList(req.body.regulations),
          constraints: readList(req.body.constraints),
        };
        // Un brief nuevo invalida la estructura, las horas y la verificación.
        await ragEstimations.update(run.id, {
          brief,
          retrievalText: String(req.body.retrieval_text || "").trim(),
          generation: null,
          structure: null,
          taskHours: null,
          verification: null,
          status: "brief",
          currentStep: "generation",
          error: null,
        });
        res.redirect(
          303,
          `/rag-estimations/${run.id}?step=brief&notice=${encodeURIComponent("Brief guardado.")}`,
        );
        return;
      }
      const outcome = await ragClient.brief(run.transcript);
      await ragEstimations.update(run.id, {
        brief: outcome.brief,
        retrievalText: outcome.retrieval_text,
        generation: null,
        structure: null,
        taskHours: null,
        verification: null,
        status: "brief",
        currentStep: "generation",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=brief`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "brief");
        redirectWithError(res, req.params.id, "brief", error.message);
        return;
      }
      next(error);
    }
  });

  // Genera la ESTRUCTURA módulo→tarea sin retrieval, fuentes ni horas.
  router.post("/rag-estimations/:id/generate", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      if (!run.brief) {
        redirectWithError(res, run.id, "generation", "Necesitas un brief antes de generar.");
        return;
      }
      const generation = await ragClient.structure(run.brief);
      await ragEstimations.update(run.id, {
        generation,
        structure: seedStructure(generation),
        taskHours: null,
        verification: null,
        status: "generated",
        currentStep: "review",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=review`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "generation");
        redirectWithError(res, req.params.id, "generation", error.message);
        return;
      }
      next(error);
    }
  });

  // Guarda la estructura revisada y deriva las horas por búsqueda vectorial.
  router.post("/rag-estimations/:id/estimate-hours", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      const modules = normalizeStructure(req.body.modules);
      if (!modules.length) {
        redirectWithError(
          res,
          run.id,
          "review",
          "La estructura necesita al menos un módulo con tareas.",
        );
        return;
      }

      const taskHours = await ragClient.taskHours(structureForApi(modules));
      await ragEstimations.update(run.id, {
        structure: { modules },
        taskHours,
        verification: seedBreakdownWithHours(modules, taskHours),
        status: "hours_estimated",
        currentStep: "hours",
        error: null,
      });
      res.redirect(
        303,
        `/rag-estimations/${run.id}?step=hours&notice=${encodeURIComponent(
          "Horas estimadas por búsqueda vectorial.",
        )}`,
      );
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "review");
        redirectWithError(res, req.params.id, "review", error.message);
        return;
      }
      next(error);
    }
  });

  router.post("/rag-estimations/:id/verify", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      const modules = normalizeCostModules(req.body.modules);
      const verification = {
        modules,
        total_hours: sumHours(modules),
        total_cost_eur: sumCost(modules),
        confirmed_at: new Date().toISOString(),
        notes: String(req.body.notes || "").trim() || null,
      };
      await ragEstimations.update(run.id, {
        verification,
        status: "verified",
        currentStep: "verification",
        error: null,
      });
      res.redirect(
        303,
        `/rag-estimations/${run.id}?step=verification&notice=${encodeURIComponent(
          "Estimación validada y guardada.",
        )}`,
      );
    } catch (error) {
      next(error);
    }
  });

  // --- Modo comparación S9 (fuera del flujo principal) ---------------------

  router.post("/rag-estimations/:id/retrieve", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      const filters = readFilters(req.body);
      const queryText =
        String(req.body.query_text || "").trim() ||
        run.retrievalText ||
        run.brief?.functionality ||
        "";
      if (queryText.length < 10) {
        redirectWithError(
          res,
          run.id,
          "retrieval",
          "Necesitas un texto de búsqueda de al menos 10 caracteres.",
        );
        return;
      }
      const retrieval = await ragClient.retrieve({
        query_text: queryText,
        top_k: filters.top_k,
        distance_threshold: filters.distance_threshold,
        sectors: filters.sectors.length ? filters.sectors : null,
        project_year_min: filters.project_year_min,
        project_year_max: filters.project_year_max,
        chunk_types: filters.chunk_types.length ? filters.chunk_types : null,
      });
      await ragEstimations.update(run.id, {
        retrieval,
        status: "retrieved",
        currentStep: "retrieval",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=retrieval`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "retrieval");
        redirectWithError(res, req.params.id, "retrieval", error.message);
        return;
      }
      next(error);
    }
  });

  router.post("/rag-estimations/:id/assemble", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      const chunks = run.retrieval?.chunks || [];
      if (!chunks.length) {
        redirectWithError(
          res,
          run.id,
          "assembly",
          "No hay chunks recuperados; ejecuta primero el retrieval.",
        );
        return;
      }
      const maxContextTokens = req.body.max_context_tokens
        ? Number.parseInt(req.body.max_context_tokens, 10)
        : null;
      const assembly = await ragClient.assemble({
        chunks,
        max_context_tokens: Number.isNaN(maxContextTokens) ? null : maxContextTokens,
      });
      await ragEstimations.update(run.id, {
        assembly,
        status: "assembled",
        currentStep: "assembly",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=assembly`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "assembly");
        redirectWithError(res, req.params.id, "assembly", error.message);
        return;
      }
      next(error);
    }
  });

  return router;
}
