import { randomUUID } from "node:crypto";

const FIELDS = [
  "status",
  "current_step",
  "brief",
  "retrieval_text",
  "retrieval",
  "assembly",
  "generation",
  "structure",
  "task_hours",
  "verification",
  "error",
];

function parseJson(value, fallback = null) {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return fallback;
    }
  }
  return value;
}

function preview(text, length = 140) {
  const clean = (text || "").replace(/\s+/g, " ").trim();
  return clean.length > length ? `${clean.slice(0, length)}…` : clean;
}

function mapRow(row, { withPayload = true } = {}) {
  const mapped = {
    id: row.id,
    createdAt: row.created_at instanceof Date ? row.created_at.toISOString() : row.created_at,
    updatedAt: row.updated_at instanceof Date ? row.updated_at.toISOString() : row.updated_at,
    status: row.status,
    currentStep: row.current_step,
    transcript: row.transcript,
    transcriptPreview: preview(row.transcript),
    idempotencyKey: row.idempotency_key || null,
  };
  if (!withPayload) {
    const generation = parseJson(row.generation);
    return {
      ...mapped,
      brief: parseJson(row.brief),
      estimate: generation?.estimate || null,
      verified: row.status === "verified",
    };
  }
  return {
    ...mapped,
    brief: parseJson(row.brief),
    retrievalText: row.retrieval_text || "",
    retrieval: parseJson(row.retrieval),
    assembly: parseJson(row.assembly),
    generation: parseJson(row.generation),
    structure: parseJson(row.structure),
    taskHours: parseJson(row.task_hours),
    verification: parseJson(row.verification),
    error: row.error || null,
  };
}

export class PgRagEstimationRepo {
  constructor(pool) {
    this.pool = pool;
  }

  async create({ transcript, idempotencyKey = null }) {
    const { rows } = await this.pool.query(
      `INSERT INTO rag_estimations (transcript, idempotency_key)
       VALUES ($1, $2)
       RETURNING id`,
      [transcript, idempotencyKey],
    );
    return rows[0].id;
  }

  async list(limit = 30) {
    const { rows } = await this.pool.query(
      `SELECT id, created_at, updated_at, status, current_step, transcript,
              brief, generation, idempotency_key
         FROM rag_estimations
        ORDER BY created_at DESC
        LIMIT $1`,
      [limit],
    );
    return rows.map((row) => mapRow(row, { withPayload: false }));
  }

  async getById(id) {
    const { rows } = await this.pool.query("SELECT * FROM rag_estimations WHERE id = $1", [id]);
    return rows[0] ? mapRow(rows[0]) : null;
  }

  async update(id, patch) {
    const current = await this.getById(id);
    if (!current) return null;
    const next = { ...current, ...patch };
    const values = FIELDS.map((field) => {
      const camel = field.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
      const value = next[camel];
      if (value === undefined || value === null) return null;
      if (
        [
          "brief",
          "retrieval",
          "assembly",
          "generation",
          "structure",
          "task_hours",
          "verification",
        ].includes(field)
      ) {
        return JSON.stringify(value);
      }
      if (field === "retrieval_text") return value;
      return value;
    });
    const { rows } = await this.pool.query(
      `UPDATE rag_estimations
          SET status = $1,
              current_step = $2,
              brief = $3,
              retrieval_text = $4,
              retrieval = $5,
              assembly = $6,
              generation = $7,
              structure = $8,
              task_hours = $9,
              verification = $10,
              error = $11,
              updated_at = now()
        WHERE id = $12
        RETURNING *`,
      [...values, id],
    );
    return rows[0] ? mapRow(rows[0]) : null;
  }
}

export class InMemoryRagEstimationRepo {
  constructor() {
    this.items = new Map();
  }

  async create({ transcript, idempotencyKey = null }) {
    const id = randomUUID();
    const now = new Date().toISOString();
    this.items.set(id, {
      id,
      createdAt: now,
      updatedAt: now,
      status: "draft",
      currentStep: "brief",
      transcript,
      transcriptPreview: preview(transcript),
      idempotencyKey,
      brief: null,
      retrievalText: "",
      retrieval: null,
      assembly: null,
      generation: null,
      structure: null,
      taskHours: null,
      verification: null,
      error: null,
    });
    return id;
  }

  async list(limit = 30) {
    return [...this.items.values()]
      .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
      .slice(0, limit)
      .map((item) => ({
        id: item.id,
        createdAt: item.createdAt,
        updatedAt: item.updatedAt,
        status: item.status,
        currentStep: item.currentStep,
        transcript: item.transcript,
        transcriptPreview: item.transcriptPreview,
        idempotencyKey: item.idempotencyKey,
        brief: item.brief,
        estimate: item.generation?.estimate || null,
        totalHours: item.verification?.total_hours ?? null,
        verified: item.status === "verified",
      }));
  }

  async getById(id) {
    return this.items.get(id) || null;
  }

  async update(id, patch) {
    const current = this.items.get(id);
    if (!current) return null;
    const next = {
      ...current,
      ...patch,
      transcriptPreview: preview(patch.transcript || current.transcript),
      updatedAt: new Date().toISOString(),
    };
    this.items.set(id, next);
    return next;
  }
}
