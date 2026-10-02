import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadConfig } from "./config.js";
import { createPool } from "./db.js";

const here = dirname(fileURLToPath(import.meta.url));

export async function runMigrations(pool) {
  const sql = await readFile(join(here, "..", "migrations", "001_create_estimations.sql"), "utf8");
  await pool.query(sql);
}

async function main() {
  const config = loadConfig();
  if (!config.databaseUrl) {
    console.error("DATABASE_URL no está configurada; no hay nada que migrar.");
    process.exitCode = 1;
    return;
  }
  const pool = createPool(config.databaseUrl);
  await runMigrations(pool);
  await pool.end();
  console.log("Migraciones aplicadas.");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error("Fallo aplicando migraciones:", error);
    process.exitCode = 1;
  });
}
