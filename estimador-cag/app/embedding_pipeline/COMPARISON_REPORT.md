# Chunking comparison report

> Generado por `scripts/compare_chunkers.py --output ...` como red de seguridad del directo. Números reales del corpus instrumentado.

## Estadísticos por estrategia

| strategy | n_chunks | min | p50 | p95 | max | orphans (<20) | obese (>800) | cost_usd | seconds |
|---|---|---|---|---|---|---|---|---|---|
| structural | 60 | 57 | 78.0 | 96.5 | 1391 | 0 | 1 | 0.000000 | 0.00 |
| fixed_size | 63 | 57 | 79.0 | 108.7 | 512 | 0 | 0 | 0.000000 | 0.00 |
| recursive | 23 | 10 | 187.0 | 488.5 | 494 | 1 | 0 | 0.000000 | 0.00 |
| sentence_window | 78 | 3 | 21.0 | 82.0 | 101 | 34 | 0 | 0.000000 | 1.69 |
| hierarchical | 77 | 57 | 81.0 | 236.8 | 1472 | 0 | 2 | 0.000000 | 0.00 |
