export const MODEL_KNOBS = [
  {
    key: "PRIMARY_MODEL",
    label: "Modelo primario",
    description: "Modelo principal de la estimación estructurada y conversacional.",
  },
  {
    key: "FALLBACK_MODEL",
    label: "Modelo de fallback",
    description: "Segundo modelo cuando el primario falla.",
  },
  {
    key: "CRITIC_MODEL",
    label: "Modelo del Critic",
    description: "Auditor del patrón Actor-Critic-Boss.",
  },
  {
    key: "METADATA_EXTRACTOR_MODEL",
    label: "Extractor de metadata",
    description: "Modelo que refresca la metadata del proyecto en cada turno.",
  },
  {
    key: "COMPRESSION_MODEL",
    label: "Compresión de memoria",
    description: "Modelo del resumidor acumulativo y las anclas.",
  },
  {
    key: "PROPOSITIONAL_CHUNKER_MODEL",
    label: "Chunker proposicional",
    description: "Modelo que descompone componentes en proposiciones atómicas.",
  },
  {
    key: "CONTEXTUAL_CHUNKER_MODEL",
    label: "Chunker contextual",
    description: "Modelo que sitúa cada chunk en su presupuesto padre.",
  },
];

export const MODEL_KEYS = MODEL_KNOBS.map((knob) => knob.key);
