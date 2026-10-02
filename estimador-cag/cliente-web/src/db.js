import { randomUUID } from "node:crypto";

import pg from "pg";

export function createPool(databaseUrl) {
  if (!databaseUrl) {
    return null;
  }
  return new pg.Pool({ connectionString: databaseUrl });
}

function preview(description, length = 90) {
  const clean = (description || "").replace(/\s+/g, " ").trim();
  return clean.length > length ? `${clean.slice(0, length)}…` : clean;
}

function mapRow(row, { withResult = false } = {}) {
  let result = row.result_json;
  if (typeof result === "string") {
    result = JSON.parse(result);
  }
  const mapped = {
    id: row.id,
    createdAt: row.created_at,
    projectType: row.project_type,
    detailLevel: row.detail_level,
    outputFormat: row.output_format,
    promptVersion: row.prompt_version,
    cached: row.cached,
    description: row.description,
    descriptionPreview: preview(row.description),
  };
  return withResult ? { ...mapped, result } : mapped;
}

export class PgEstimationRepo {
  constructor(pool) {
    this.pool = pool;
  }

  async create(record) {
    const { rows } = await this.pool.query(
      `INSERT INTO estimations
         (project_type, detail_level, output_format, prompt_version, cached, description, result_json)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id`,
      [
        record.projectType,
        record.detailLevel,
        record.outputFormat,
        record.promptVersion,
        record.cached,
        record.description,
        JSON.stringify(record.result),
      ],
    );
    return rows[0].id;
  }

  async list() {
    const { rows } = await this.pool.query(
      `SELECT id, created_at, project_type, detail_level, output_format,
              prompt_version, cached, description
         FROM estimations
        ORDER BY created_at DESC
        LIMIT 200`,
    );
    return rows.map((row) => mapRow(row));
  }

  async getById(id) {
    const { rows } = await this.pool.query("SELECT * FROM estimations WHERE id = $1", [id]);
    return rows[0] ? mapRow(rows[0], { withResult: true }) : null;
  }
}

export class InMemoryEstimationRepo {
  constructor() {
    this.items = new Map();
  }

  async create(record) {
    const id = randomUUID();
    this.items.set(id, {
      id,
      createdAt: new Date().toISOString(),
      projectType: record.projectType,
      detailLevel: record.detailLevel,
      outputFormat: record.outputFormat,
      promptVersion: record.promptVersion,
      cached: record.cached,
      description: record.description,
      descriptionPreview: preview(record.description),
      result: record.result,
    });
    return id;
  }

  async list() {
    return [...this.items.values()]
      .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
      .map(({ result, ...rest }) => rest);
  }

  async getById(id) {
    return this.items.get(id) || null;
  }
}
