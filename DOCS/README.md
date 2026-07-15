# MLX Swift LM Research Reports

Fecha inicial: 2026-07-01. Actualizado: 2026-07-15 (rama `docs/improvement-backlog-review`).

Esta carpeta resume una investigación técnica del framework `mlx-swift-lm` en la rama `docs/framework-performance-research`, creada desde `origin/main`, y su implementación posterior en `docs/improvement-backlog-review`.

## Informes originales (2026-07-01)

- [Rendimiento y generación](performance-and-generation.md): prefill, decode, speculative decoding, MTP, sampling y medición.
- [Compatibilidad y modelos](compatibility-and-models.md): arquitecturas soportadas, VLM, embeddings, cuantizaciones y portabilidad.
- [Compatibilidad de Bonsai 1-bit](bonsai-1bit-compatibility.md): port affine-only, revisiones exactas, validación y rollback.
- [Memoria y KV cache](memory-kv-cache.md): caché rotatoria, caché cuantizada, wired memory, límites Apple y patrones de uso.
- [Playbook de implementación](implementation-playbook.md): recetas prácticas para exprimir el framework en apps y herramientas.
- [Tech debt e investigación](tech-debt-and-research-backlog.md): backlog priorizado de mejoras, deuda técnica y experimentos.
- [Fuentes](sources.md): fuentes locales y externas consultadas.

## Implementación del backlog (2026-07-15)

> **Continuación**: si vas a retomar este trabajo, empieza por
> [backlog-continuation-plan.md](backlog-continuation-plan.md) — tiene el
> estado exacto, los comandos de build/test ya verificados en este
> entorno y el orden recomendado de próximos pasos, para no tener que
> re-investigar nada de lo ya resuelto en esta rama.

Cada ítem de `tech-debt-and-research-backlog.md` tiene ahora un documento de
resultado, marcado explícitamente como **código verificado** o **investigación
/ nota de alcance** (no implementado, por diseño — ver cada nota para el porqué).

**Código verificado (build + tests reales, ver commits en la rama):**

- Prioridad alta #1, benchmark suite: [`BenchmarkReport`](../Libraries/BenchmarkHelpers/BenchmarkReport.swift) (JSON, comparación entre runs y tests de round-trip/comparación) y [`SamplingBenchmarks`](../Libraries/BenchmarkHelpers/SamplingBenchmarks.swift) (microbenchmarks de sampling, ítem #8).
- Prioridad alta #2, matriz de compatibilidad: [compatibility-matrix.md](compatibility-matrix.md), generada por [`CompatibilityMatrixGeneratorTests`](../Tests/MLXLMTests/CompatibilityMatrixGeneratorTests.swift) desde los registros en vivo.
- Prioridad alta #5, reducir `fatalError`: `KVCache.swift` — deserialización corrupta y la nueva conversión segura `quantized(...)` lanzan un `KVCacheError` público. Las firmas históricas `toQuantized(...)` se conservan para compatibilidad 3.x y quedan deprecadas.
- Prioridad media #7 (parcial), MTP: el `print()` directo en el passthrough de `MTPSpeculativeTokenIterator` ahora usa el `Logger` del módulo.

**Investigación / notas de alcance (documentadas, no implementadas — cada una explica por qué):**

- Prioridad alta #3: [unified-memory-policy-scoping.md](unified-memory-policy-scoping.md).
- Prioridad alta #4: [long-context-recipes.md](long-context-recipes.md) (esta sí es una guía de uso completa, no solo alcance).
- Prioridad media #6: [adaptive-speculative-decoding-scoping.md](adaptive-speculative-decoding-scoping.md).
- Prioridad media #7 (resto): [mtp-production-scoping.md](mtp-production-scoping.md).
- Prioridad media #9: [vlm-media-budgets.md](vlm-media-budgets.md).
- Prioridad media #10: [tool-thinking-compatibility-layer.md](tool-thinking-compatibility-layer.md).
- Prioridad baja-estratégica #11: [porting-from-mlx-lm-playbook.md](porting-from-mlx-lm-playbook.md) (+ plantillas en `scripts/porting/templates/`).
- Prioridad baja-estratégica #12: [kvscheme-research-note.md](kvscheme-research-note.md).
- Prioridad baja-estratégica #13: [cache-persistence-api.md](cache-persistence-api.md).
- Prioridad baja-estratégica #14: [conformance-dashboard.md](conformance-dashboard.md) (+ `scripts/conformance-dashboard/generate-report.sh`).
- Prioridad baja-estratégica #15: [detokenizer-media-investigation.md](detokenizer-media-investigation.md).

**Hallazgo no listado originalmente en el backlog, cerrado en la pasada de release:**

- [gemma4-chunked-prefill-investigation.md](gemma4-chunked-prefill-investigation.md): las cuatro variantes de `chunkSizeInvariance` que fallaban con el runtime anterior pasan con el pin M5-safe de Bonsai/NAX, sin modificar `RotatingKVCache`. La hipótesis de caché queda descartada como base para un fix de este release; una atribución causal estricta a NAX requeriría A/B aislado.

## Lectura rápida (original, 2026-07-01)

El framework ya tenía una base muy avanzada: Swift 6.1, arquitectura modular 3.x, desacoplo de tokenizer/downloader, 58 `model_type` LLM registrados, 18 registros VLM, embeddings, `GenerateParameters` con control de KV cache y sampling, caché cuantizada extensible, wired-memory coordination, tool calling y speculative decoding/MTP.

Las mejoras con más retorno no estaban en "añadir un flag más", sino en cerrar tres bucles:

1. Medición reproducible: benchmarks de TTFT, tokens/s, memoria pico, acceptance rate de speculative decoding y calidad tras KV quantization por dispositivo/modelo. → Mecanismo (JSON) cerrado el 2026-07-15; datos reales por dispositivo siguen pendientes.
2. Políticas adaptativas: seleccionar `prefillStepSize`, `maxKVSize`, `kvBits`, draft model y memoria wired a partir del presupuesto real de dispositivo y de la conversación. → Alcance documentado el 2026-07-15, no implementado (ver notas de scoping arriba).
3. Compatibilidad verificable: matriz viva de modelos, cuantizaciones, chat templates, thinking/tool formats y media budgets con tests de integración etiquetados. → Matriz mecánica y guía de media budgets cerradas el 2026-07-15; thinking/tool formats documentados (tool ya existía, thinking sigue siendo un gap real).

## Hallazgos críticos (original, con estado actualizado)

- MTP mejora rendimiento en tests de integración; el fallback a passthrough con KV cuantizado **ya existía y ya tenía test dedicado** (`MTPQuantizationOnsetTests`) — el gap real, documentado el 2026-07-15, es que no está integrado en `ChatSession`, no que falte el fallback en sí. Ver `mtp-production-scoping.md`.
- `ChatSession` reutiliza KV cache y soporta speculative decoding clásico, pero no expone wired-memory ticket por turno ni sincroniza todo su estado mutable público. **Confirmado literalmente** el 2026-07-15 (`WiredMemoryPolicy` no tiene ningún consumidor en `ChatSession.swift`). Ver `unified-memory-policy-scoping.md`.
- La cuantización de pesos cubre affine, MXFP4/MXFP8, NVFP4 y rutas especiales como ParoQuant, pero conversión sigue centrada en safetensors; `.bin`, GGUF y requantización de modelos ya cuantizados quedan fuera.
- Varias rutas públicas o semi-públicas de KV cache usaban `fatalError`; la deserialización de prompt cache corrupto y las nuevas APIs `quantized(...)` son recuperables mediante `KVCacheError`. Las firmas no throwing anteriores se mantienen deprecadas para no introducir una ruptura de API en 3.x.
