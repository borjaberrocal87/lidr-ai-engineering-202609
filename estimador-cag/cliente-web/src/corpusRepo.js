import { randomUUID } from "node:crypto";

function mapRun(row) {
  let beforeStats = row.before_stats;
  if (typeof beforeStats === "string") {
    try {
      beforeStats = JSON.parse(beforeStats);
    } catch {
      beforeStats = null;
    }
  }
  return {
    id: row.id,
    jobId: row.job_id,
    chunkType: row.chunk_type,
    documentsTotal: row.documents_total,
    status: row.status,
    beforeStats,
    error: row.error,
    createdAt:
      row.created_at instanceof Date ? row.created_at.toISOString() : row.created_at,
  };
}

/** Historial de expansiones del corpus (Sesión 11) para la vista /corpus. */
export class PgCorpusRunRepo {
  constructor(pool) {
    this.pool = pool;
  }

  async create(record) {
    const { rows } = await this.pool.query(
      `INSERT INTO corpus_index_runs
         (job_id, chunk_type, documents_total, status, before_stats)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING id`,
      [
        record.jobId,
        record.chunkType,
        record.documentsTotal,
        record.status || "pending",
        JSON.stringify(record.beforeStats || null),
      ],
    );
    return rows[0].id;
  }

  async update(id, { status, error }) {
    await this.pool.query(
      `UPDATE corpus_index_runs
          SET status = COALESCE($2, status),
              error = COALESCE($3, error),
              updated_at = now()
        WHERE id = $1`,
      [id, status || null, error || null],
    );
  }

  async list(limit = 20) {
    const { rows } = await this.pool.query(
      `SELECT id, job_id, chunk_type, documents_total, status, before_stats, error, created_at
         FROM corpus_index_runs
        ORDER BY created_at DESC
        LIMIT $1`,
      [limit],
    );
    return rows.map(mapRun);
  }

  async getById(id) {
    const { rows } = await this.pool.query("SELECT * FROM corpus_index_runs WHERE id = $1", [
      id,
    ]);
    return rows[0] ? mapRun(rows[0]) : null;
  }
}

export class InMemoryCorpusRunRepo {
  constructor() {
    this.items = new Map();
  }

  async create(record) {
    const id = randomUUID();
    this.items.set(id, {
      id,
      jobId: record.jobId,
      chunkType: record.chunkType,
      documentsTotal: record.documentsTotal,
      status: record.status || "pending",
      beforeStats: record.beforeStats || null,
      error: null,
      createdAt: new Date().toISOString(),
    });
    return id;
  }

  async update(id, { status, error }) {
    const item = this.items.get(id);
    if (!item) return;
    if (status) item.status = status;
    if (error) item.error = error;
  }

  async list(limit = 20) {
    return [...this.items.values()]
      .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
      .slice(0, limit);
  }

  async getById(id) {
    return this.items.get(id) || null;
  }
}
