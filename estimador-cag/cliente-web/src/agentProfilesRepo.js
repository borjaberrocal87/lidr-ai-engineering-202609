import { randomUUID } from "node:crypto";

function parseConfig(value) {
  if (value === null || value === undefined) return {};
  if (typeof value === "string") {
    try {
      return JSON.parse(value) || {};
    } catch {
      return {};
    }
  }
  return value;
}

function mapRow(row) {
  return {
    id: row.id,
    name: row.name,
    agentType: row.agent_type,
    persona: row.persona || "",
    config: parseConfig(row.config),
    isDefault: Boolean(row.is_default),
    avatarDataUrl: row.avatar_data_url || null,
    avatarContentType: row.avatar_content_type || null,
    createdAt:
      row.created_at instanceof Date ? row.created_at.toISOString() : row.created_at,
    updatedAt:
      row.updated_at instanceof Date ? row.updated_at.toISOString() : row.updated_at,
  };
}

function sortProfiles(items) {
  return [...items].sort((a, b) => {
    if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1;
    return a.name.localeCompare(b.name, "es");
  });
}

/** Perfiles de agente (Sesión 12) persistidos en Postgres. */
export class PgAgentProfilesRepo {
  constructor(pool) {
    this.pool = pool;
  }

  async list() {
    const { rows } = await this.pool.query(
      "SELECT * FROM agent_profiles ORDER BY is_default DESC, lower(name) ASC",
    );
    return rows.map(mapRow);
  }

  async getById(id) {
    const { rows } = await this.pool.query("SELECT * FROM agent_profiles WHERE id = $1", [id]);
    return rows[0] ? mapRow(rows[0]) : null;
  }

  async #clearDefault(agentType) {
    await this.pool.query(
      "UPDATE agent_profiles SET is_default = false WHERE agent_type = $1 AND is_default",
      [agentType],
    );
  }

  async create(record) {
    if (record.isDefault) await this.#clearDefault(record.agentType);
    const { rows } = await this.pool.query(
      `INSERT INTO agent_profiles
         (name, agent_type, persona, config, is_default, avatar_data_url, avatar_content_type)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id`,
      [
        record.name,
        record.agentType,
        record.persona || null,
        JSON.stringify(record.config || {}),
        Boolean(record.isDefault),
        record.avatarDataUrl || null,
        record.avatarContentType || null,
      ],
    );
    return rows[0].id;
  }

  async update(id, record) {
    if (record.isDefault) await this.#clearDefault(record.agentType);
    await this.pool.query(
      `UPDATE agent_profiles
          SET name = $2,
              persona = $3,
              config = $4,
              is_default = $5,
              avatar_data_url = $6,
              avatar_content_type = $7,
              updated_at = now()
        WHERE id = $1`,
      [
        id,
        record.name,
        record.persona || null,
        JSON.stringify(record.config || {}),
        Boolean(record.isDefault),
        record.avatarDataUrl || null,
        record.avatarContentType || null,
      ],
    );
    return id;
  }

  async remove(id) {
    await this.pool.query("DELETE FROM agent_profiles WHERE id = $1", [id]);
  }

  async setDefault(id) {
    const profile = await this.getById(id);
    if (!profile) return;
    await this.#clearDefault(profile.agentType);
    await this.pool.query(
      "UPDATE agent_profiles SET is_default = true, updated_at = now() WHERE id = $1",
      [id],
    );
  }
}

/** Misma interfaz en memoria, para tests y desarrollo sin base de datos. */
export class InMemoryAgentProfilesRepo {
  constructor() {
    this.items = new Map();
  }

  async list() {
    return sortProfiles([...this.items.values()]);
  }

  async getById(id) {
    return this.items.get(id) || null;
  }

  async create(record) {
    const id = randomUUID();
    const now = new Date().toISOString();
    if (record.isDefault) {
      for (const item of this.items.values()) {
        if (item.agentType === record.agentType) item.isDefault = false;
      }
    }
    this.items.set(id, {
      id,
      name: record.name,
      agentType: record.agentType,
      persona: record.persona || "",
      config: record.config || {},
      isDefault: Boolean(record.isDefault),
      avatarDataUrl: record.avatarDataUrl || null,
      avatarContentType: record.avatarContentType || null,
      createdAt: now,
      updatedAt: now,
    });
    return id;
  }

  async update(id, record) {
    const current = this.items.get(id);
    if (!current) return id;
    if (record.isDefault) {
      for (const item of this.items.values()) {
        if (item.agentType === record.agentType) item.isDefault = false;
      }
    }
    this.items.set(id, {
      ...current,
      name: record.name,
      persona: record.persona || "",
      config: record.config || {},
      isDefault: Boolean(record.isDefault),
      avatarDataUrl: record.avatarDataUrl || null,
      avatarContentType: record.avatarContentType || null,
      updatedAt: new Date().toISOString(),
    });
    return id;
  }

  async remove(id) {
    this.items.delete(id);
  }

  async setDefault(id) {
    const current = this.items.get(id);
    if (!current) return;
    for (const item of this.items.values()) {
      if (item.agentType === current.agentType) item.isDefault = false;
    }
    current.isDefault = true;
    current.updatedAt = new Date().toISOString();
  }
}
