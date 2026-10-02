export const PROJECT_TYPES = ["mobile_app", "web_saas", "internal_tool", "data_pipeline"];
export const DETAIL_LEVELS = ["summary", "medium", "detailed"];
export const OUTPUT_FORMATS = ["phases_table", "line_items", "narrative"];
export const DEFAULT_PROMPT_VERSIONS = ["v1"];

export const DESCRIPTION_MIN = 20;
export const DESCRIPTION_MAX = 50000;
export const LOW_CONFIDENCE_THRESHOLD = 30;
export const OUT_OF_SCOPE_PREFIX = "Out of scope:";

export function humanize(value) {
  return String(value)
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function isOutOfScope(result) {
  return result.confidence_pct < LOW_CONFIDENCE_THRESHOLD;
}
