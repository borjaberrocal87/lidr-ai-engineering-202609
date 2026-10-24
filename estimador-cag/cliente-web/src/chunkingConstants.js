export const CHUNK_STRATEGIES = [
  {
    name: "structural",
    label: "Structural",
    paid: false,
    description: "Un chunk por componente del presupuesto. La referencia base.",
  },
  {
    name: "fixed_size",
    label: "Fixed size",
    paid: false,
    description: "Ventana fija de tokens con solape. El suelo de la comparativa.",
  },
  {
    name: "recursive",
    label: "Recursive",
    paid: false,
    description: "Divide por párrafo, línea, frase y palabra. Buen coste/calidad.",
  },
  {
    name: "sentence_window",
    label: "Sentence window",
    paid: false,
    description: "Indexa frases y guarda una ventana de contexto alrededor.",
  },
  {
    name: "semantic",
    label: "Semantic",
    paid: true,
    description: "Corta donde cae la similitud entre frases. Embebe durante la ingesta.",
  },
  {
    name: "propositional",
    label: "Propositional",
    paid: true,
    description: "Un LLM descompone cada componente en proposiciones atómicas.",
  },
  {
    name: "contextual_retrieval",
    label: "Contextual retrieval",
    paid: true,
    description: "Un LLM sitúa cada chunk en su presupuesto padre (prompt caching).",
  },
  {
    name: "hierarchical",
    label: "Hierarchical",
    paid: false,
    description: "Padres (presupuesto completo) e hijos (componentes) a la vez.",
  },
];

export const DEFAULT_STRATEGIES = CHUNK_STRATEGIES.filter((item) => !item.paid).map(
  (item) => item.name,
);

export const STRATEGY_BY_NAME = Object.fromEntries(
  CHUNK_STRATEGIES.map((item) => [item.name, item]),
);

export function strategyLabel(name) {
  return STRATEGY_BY_NAME[name]?.label || name;
}

export function isPaidStrategy(name) {
  return Boolean(STRATEGY_BY_NAME[name]?.paid);
}
