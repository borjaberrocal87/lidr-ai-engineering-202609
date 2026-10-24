import { randomUUID } from "node:crypto";

function mapRow(row, { withPayload = false } = {}) {
  let payload = row.response_payload;
  if (typeof payload === "string") {
    payload = JSON.parse(payload);
  }
  const mapped = {
    id: row.id,
    createdAt: row.created_at instanceof Date ? row.created_at.toISOString() : row.created_at,
    strategies: row.strategies,
    queries: row.queries,
    topK: row.top_k,
    corpusLabel: row.corpus_label,
    corpusCount: row.corpus_count,
    durationMs: row.duration_ms,
  };
  return withPayload ? { ...mapped, responsePayload: payload } : mapped;
}

export class PgChunkingRepo {
  constructor(pool) {
    this.pool = pool;
  }

  async create(record) {
    const { rows } = await this.pool.query(
      `INSERT INTO chunking_comparisons
         (strategies, queries, top_k, corpus_label, corpus_count, response_payload, duration_ms)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id`,
      [
        JSON.stringify(record.strategies),
        JSON.stringify(record.queries),
        record.topK,
        record.corpusLabel,
        record.corpusCount,
        JSON.stringify(record.responsePayload),
        record.durationMs,
      ],
    );
    return rows[0].id;
  }

  async list(limit = 20) {
    const { rows } = await this.pool.query(
      `SELECT id, created_at, strategies, queries, top_k, corpus_label, corpus_count,
              duration_ms
         FROM chunking_comparisons
        ORDER BY created_at DESC
        LIMIT $1`,
      [limit],
    );
    return rows.map((row) => mapRow(row));
  }

  async getById(id) {
    const { rows } = await this.pool.query(
      "SELECT * FROM chunking_comparisons WHERE id = $1",
      [id],
    );
    return rows[0] ? mapRow(rows[0], { withPayload: true }) : null;
  }
}

export class InMemoryChunkingRepo {
  constructor() {
    this.items = new Map();
  }

  async create(record) {
    const id = randomUUID();
    this.items.set(id, {
      id,
      createdAt: new Date().toISOString(),
      strategies: record.strategies,
      queries: record.queries,
      topK: record.topK,
      corpusLabel: record.corpusLabel,
      corpusCount: record.corpusCount,
      responsePayload: record.responsePayload,
      durationMs: record.durationMs,
    });
    return id;
  }

  async list(limit = 20) {
    return [...this.items.values()]
      .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
      .slice(0, limit)
      .map(({ responsePayload, ...rest }) => rest);
  }

  async getById(id) {
    return this.items.get(id) || null;
  }
}
