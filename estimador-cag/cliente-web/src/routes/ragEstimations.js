import { Router } from "express";

import { RagClientError } from "../ragEstimationClient.js";

const STEPS = ["brief", "retrieval", "assembly", "generation", "verification"];

function clamp(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function readList(value) {
  const raw = Array.isArray(value) ? value : String(value || "").split(",");
  return raw.map((item) => String(item).trim()).filter(Boolean);
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

function normalizeModules(raw) {
  const modules = raw == null ? [] : Array.isArray(raw) ? raw : Object.values(raw);
  return modules
    .filter((module) => module && typeof module === "object")
    .map((module) => {
      const tasks = module.tasks == null
        ? []
        : Array.isArray(module.tasks)
          ? module.tasks
          : Object.values(module.tasks);
      return {
        name: String(module.name || "").trim() || "Módulo sin nombre",
        description: String(module.description || "").trim() || null,
        tasks: tasks
          .filter((task) => task && typeof task === "object")
          .map((task) => ({
            name: String(task.name || "").trim() || "Tarea sin nombre",
            description: String(task.description || "").trim() || null,
            engineer_days: Math.max(0, Number.parseInt(task.engineer_days, 10) || 0),
            sources: readList(task.sources)
              .map((source) => Number.parseInt(source, 10))
              .filter((source) => !Number.isNaN(source)),
          })),
      };
    });
}

function sumDays(modules) {
  return modules.reduce(
    (total, module) =>
      total + module.tasks.reduce((partial, task) => partial + task.engineer_days, 0),
    0,
  );
}

function resolveStep(queryStep, run) {
  if (STEPS.includes(queryStep)) return queryStep;
  return STEPS.includes(run?.currentStep) ? run.currentStep : "brief";
}

function viewContext(run, step, { error = "", notice = "" } = {}) {
  const generation = run.generation || null;
  const estimate = generation?.estimate || null;
  return {
    run,
    step,
    steps: STEPS,
    stepIndex: STEPS.indexOf(step),
    error: error || run.error || "",
    notice,
    filters: readFilters({}),
    estimate,
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
        currentStep: "retrieval",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${id}?step=brief`);
    } catch (error) {
      if (run) {
        const message = await failed(run, error, "brief");
        res.redirect(303, `/rag-estimations/${run.id}?step=brief&error=${encodeURIComponent(message)}`);
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
        await ragEstimations.update(run.id, {
          brief,
          retrievalText: String(req.body.retrieval_text || "").trim(),
          status: "brief",
          currentStep: "retrieval",
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
        status: "brief",
        currentStep: "retrieval",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=brief`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "brief");
        res.redirect(
          303,
          `/rag-estimations/${req.params.id}?step=brief&error=${encodeURIComponent(error.message)}`,
        );
        return;
      }
      next(error);
    }
  });

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
        res.redirect(
          303,
          `/rag-estimations/${run.id}?step=retrieval&error=${encodeURIComponent(
            "Necesitas un texto de búsqueda de al menos 10 caracteres.",
          )}`,
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
        currentStep: "assembly",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=retrieval`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "retrieval");
        res.redirect(
          303,
          `/rag-estimations/${req.params.id}?step=retrieval&error=${encodeURIComponent(error.message)}`,
        );
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
        res.redirect(
          303,
          `/rag-estimations/${run.id}?step=assembly&error=${encodeURIComponent(
            "No hay chunks recuperados; ejecuta primero el retrieval.",
          )}`,
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
        currentStep: "generation",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=assembly`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "assembly");
        res.redirect(
          303,
          `/rag-estimations/${req.params.id}?step=assembly&error=${encodeURIComponent(error.message)}`,
        );
        return;
      }
      next(error);
    }
  });

  router.post("/rag-estimations/:id/generate", async (req, res, next) => {
    try {
      const run = await ragEstimations.getById(req.params.id);
      if (!run) {
        res.status(404).render("not-found");
        return;
      }
      const contextBlock =
        String(req.body.context_block || "").trim() || run.assembly?.context_block || "";
      if (!contextBlock) {
        res.redirect(
          303,
          `/rag-estimations/${run.id}?step=generation&error=${encodeURIComponent(
            "No hay contexto ensamblado; ejecuta primero la augmentación.",
          )}`,
        );
        return;
      }
      const generation = await ragClient.generate({
        context_block: contextBlock,
        brief: run.brief,
        kept_chunks: run.assembly?.kept_chunks || [],
      });
      await ragEstimations.update(run.id, {
        generation,
        status: "generated",
        currentStep: "verification",
        error: null,
      });
      res.redirect(303, `/rag-estimations/${run.id}?step=generation`);
    } catch (error) {
      if (error instanceof RagClientError) {
        const run = await ragEstimations.getById(req.params.id);
        if (run) await failed(run, error, "generation");
        res.redirect(
          303,
          `/rag-estimations/${req.params.id}?step=generation&error=${encodeURIComponent(error.message)}`,
        );
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
      const modules = normalizeModules(req.body.modules);
      const totalEngineerDays = sumDays(modules);
      const durationWeeks = req.body.duration_weeks
        ? Number.parseInt(req.body.duration_weeks, 10)
        : run.generation?.estimate?.duration_weeks || null;
      await ragEstimations.update(run.id, {
        verification: {
          modules,
          total_engineer_days: totalEngineerDays,
          duration_weeks: Number.isNaN(durationWeeks) ? null : durationWeeks,
          notes: String(req.body.notes || "").trim() || null,
        },
        status: "verified",
        currentStep: "verification",
        error: null,
      });
      res.redirect(
        303,
        `/rag-estimations/${run.id}?step=verification&notice=${encodeURIComponent(
          "Estimación verificada y guardada.",
        )}`,
      );
    } catch (error) {
      next(error);
    }
  });

  return router;
}
