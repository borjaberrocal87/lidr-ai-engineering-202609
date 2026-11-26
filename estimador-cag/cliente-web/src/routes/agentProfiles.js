import { Router } from "express";

export const AGENT_TYPES = ["handwritten"];
export const EFFORTS = ["minimal", "low", "medium", "high"];
export const CONFIG_KEYS = [
  "model",
  "reasoning_effort",
  "max_iterations",
  "search_top_k",
  "search_distance_threshold",
];
export const MAX_AVATAR_BYTES = 512 * 1024;
const AVATAR_RE = /^data:(image\/(?:png|jpeg|gif|webp));base64,([A-Za-z0-9+/=\s]+)$/;

const NAME_MAX = 120;
const PERSONA_MAX = 2000;

function parseInteger(value, { min, max }) {
  const raw = String(value ?? "").trim();
  if (!raw) return { value: null };
  const parsed = Number(raw);
  if (!Number.isInteger(parsed)) return { error: "debe ser un número entero" };
  if (parsed < min || parsed > max) return { error: `debe estar entre ${min} y ${max}` };
  return { value: parsed };
}

function parseAvatar(rawValue) {
  const raw = String(rawValue || "").trim();
  if (!raw) return { dataUrl: null, contentType: null };
  const match = AVATAR_RE.exec(raw);
  if (!match) {
    return { error: "el avatar debe ser una imagen PNG, JPEG, GIF o WebP" };
  }
  const base64 = match[2].replace(/\s+/g, "");
  const bytes = Math.floor((base64.length * 3) / 4);
  if (bytes > MAX_AVATAR_BYTES) {
    return { error: `el avatar no puede superar ${MAX_AVATAR_BYTES / 1024} KB` };
  }
  return { dataUrl: raw, contentType: match[1] };
}

/** Normaliza y valida el formulario de un perfil de agente. */
export function normalizeProfileInput(body = {}) {
  const errors = [];
  const name = String(body.name || "").trim();
  if (!name) errors.push("El nombre es obligatorio.");
  if (name.length > NAME_MAX) errors.push(`El nombre no puede superar ${NAME_MAX} caracteres.`);

  const agentType = String(body.agent_type || "handwritten").trim();
  if (!AGENT_TYPES.includes(agentType)) errors.push("Tipo de agente no válido.");

  const persona = String(body.persona || "").trim();
  if (persona.length > PERSONA_MAX) {
    errors.push(`La persona no puede superar ${PERSONA_MAX} caracteres.`);
  }

  const config = {};
  const model = String(body.model || "").trim();
  if (model) config.model = model;

  const effort = String(body.reasoning_effort || "").trim();
  if (effort) {
    if (!EFFORTS.includes(effort)) errors.push("Esfuerzo de razonamiento no válido.");
    else config.reasoning_effort = effort;
  }

  const maxIterations = parseInteger(body.max_iterations, { min: 1, max: 20 });
  if (maxIterations.error) errors.push(`Máximo de iteraciones ${maxIterations.error}.`);
  else if (maxIterations.value !== null) config.max_iterations = maxIterations.value;

  const topK = parseInteger(body.search_top_k, { min: 1, max: 30 });
  if (topK.error) errors.push(`Top K ${topK.error}.`);
  else if (topK.value !== null) config.search_top_k = topK.value;

  let threshold = null;
  const rawThreshold = String(body.search_distance_threshold ?? "").trim();
  if (rawThreshold) {
    const parsed = Number(rawThreshold);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 2) {
      errors.push("El umbral de distancia debe estar entre 0 y 2.");
    } else {
      threshold = parsed;
      config.search_distance_threshold = parsed;
    }
  }

  const avatar = parseAvatar(body.avatar_data_url);
  if (avatar.error) errors.push(`Avatar: ${avatar.error}.`);

  const isDefault = ["1", "on", "true"].includes(String(body.is_default || "").toLowerCase());

  return {
    errors,
    values: {
      name,
      agentType,
      persona,
      model,
      reasoning_effort: effort,
      max_iterations: maxIterations.value,
      search_top_k: topK.value,
      search_distance_threshold: threshold,
      isDefault,
      avatarDataUrl: avatar.dataUrl,
    },
    record: {
      name,
      agentType,
      persona,
      config,
      isDefault,
      avatarDataUrl: avatar.dataUrl,
      avatarContentType: avatar.contentType,
    },
  };
}

/** Body compacto que el perfil envía al servicio IA (knobs no fijados se omiten). */
export function configPayload(profile) {
  const config = profile?.config || {};
  const payload = {};
  for (const key of CONFIG_KEYS) {
    const value = config[key];
    if (value === undefined || value === null || value === "") continue;
    payload[key] = value;
  }
  return payload;
}

export function agentProfilesRouter({ agentProfiles }) {
  const router = Router();

  const findNameClash = async (values, excludeId = null) => {
    const items = await agentProfiles.list();
    const target = values.name.toLowerCase();
    return items.find(
      (item) =>
        item.id !== excludeId &&
        item.agentType === values.agentType &&
        item.name.toLowerCase() === target,
    );
  };

  const renderForm = (res, { profile, values, errors, status = 200 }) => {
    const view = profile ? "agent-profiles/edit" : "agent-profiles/new";
    res.status(status).render(view, {
      profile: profile || null,
      values,
      errors,
      efforts: EFFORTS,
      agentTypes: AGENT_TYPES,
      maxAvatarKb: MAX_AVATAR_BYTES / 1024,
    });
  };

  router.get("/agent-profiles", async (req, res, next) => {
    try {
      const profiles = await agentProfiles.list();
      res.render("agent-profiles/index", {
        profiles,
        notice: req.query.notice ? String(req.query.notice) : "",
      });
    } catch (error) {
      next(error);
    }
  });

  router.get("/agent-profiles/new", (req, res) => {
    renderForm(res, {
      values: { agentType: "handwritten", isDefault: false },
      errors: [],
    });
  });

  router.post("/agent-profiles", async (req, res, next) => {
    try {
      const { errors, values, record } = normalizeProfileInput(req.body);
      const clash = errors.length ? null : await findNameClash(values);
      if (clash) errors.push("Ya existe un perfil con ese nombre para este tipo de agente.");
      if (errors.length) {
        renderForm(res, { values, errors, status: 422 });
        return;
      }
      const id = await agentProfiles.create(record);
      res.redirect(303, `/agent-profiles/${id}?notice=${encodeURIComponent("Perfil creado.")}`);
    } catch (error) {
      next(error);
    }
  });

  router.get("/agent-profiles/:id", async (req, res, next) => {
    try {
      const profile = await agentProfiles.getById(req.params.id);
      if (!profile) {
        res.status(404).render("not-found");
        return;
      }
      res.render("agent-profiles/show", {
        profile,
        configPayload: configPayload(profile),
        notice: req.query.notice ? String(req.query.notice) : "",
      });
    } catch (error) {
      next(error);
    }
  });

  router.get("/agent-profiles/:id/edit", async (req, res, next) => {
    try {
      const profile = await agentProfiles.getById(req.params.id);
      if (!profile) {
        res.status(404).render("not-found");
        return;
      }
      renderForm(res, {
        profile,
        values: {
          ...profile.config,
          name: profile.name,
          agentType: profile.agentType,
          persona: profile.persona,
          isDefault: profile.isDefault,
          avatarDataUrl: profile.avatarDataUrl,
        },
        errors: [],
      });
    } catch (error) {
      next(error);
    }
  });

  router.post("/agent-profiles/:id", async (req, res, next) => {
    try {
      const profile = await agentProfiles.getById(req.params.id);
      if (!profile) {
        res.status(404).render("not-found");
        return;
      }
      const { errors, values, record } = normalizeProfileInput(req.body);
      const clash = errors.length ? null : await findNameClash(values, profile.id);
      if (clash) errors.push("Ya existe un perfil con ese nombre para este tipo de agente.");
      if (errors.length) {
        values.avatarDataUrl = values.avatarDataUrl || profile.avatarDataUrl;
        renderForm(res, { profile, values, errors, status: 422 });
        return;
      }
      await agentProfiles.update(profile.id, record);
      res.redirect(
        303,
        `/agent-profiles/${profile.id}?notice=${encodeURIComponent("Perfil actualizado.")}`,
      );
    } catch (error) {
      next(error);
    }
  });

  router.post("/agent-profiles/:id/default", async (req, res, next) => {
    try {
      await agentProfiles.setDefault(req.params.id);
      res.redirect(
        303,
        `/agent-profiles?notice=${encodeURIComponent("Perfil marcado como predeterminado.")}`,
      );
    } catch (error) {
      next(error);
    }
  });

  router.post("/agent-profiles/:id/delete", async (req, res, next) => {
    try {
      await agentProfiles.remove(req.params.id);
      res.redirect(303, `/agent-profiles?notice=${encodeURIComponent("Perfil eliminado.")}`);
    } catch (error) {
      next(error);
    }
  });

  return router;
}
