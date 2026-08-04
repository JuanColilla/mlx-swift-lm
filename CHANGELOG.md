# Changelog del fork

Este documento registra las variantes publicadas por
`JuanColilla/mlx-swift-lm`. Los tags terminados en `v` contienen cambios propios
del fork sobre la base oficial; las novedades del proyecto original se
mantienen en sus releases de GitHub.

## 3.31.6v — 2026-07-15

Base: `3.31.5v`. Detalle verificable por bloque, commit y test en
[`DOCS/backlog-completion-status.md`](DOCS/backlog-completion-status.md).

### Generación y memoria

- Añade estimación pública de memoria de KV cache, composición con
  `WiredBudgetPolicy` y regresiones de long-context para trim, máscaras,
  wraparound, persistencia y configuraciones rotatorias/cuantizadas.
- Expone `ChatSession.wiredMemoryTicket` por turno, prefill sin generación
  visible y persistencia versionada de prompt cache.
- Incorpora speculative decoding adaptativo opt-in, con ventana de acceptance
  rate, fallback autoregresivo sticky y telemetría explicable.
- Integra MTP en `ChatSession` mediante `MTPSpeculativeDecodingConfig`, con carga
  eager/deferred, gates de memoria antes y después de cargar, continuidad
  multiturno y benchmark JSON.

### Compatibilidad y observabilidad

- Añade metadata tipada y conservadora para capacidades de thinking y tools.
- Genera una matriz de compatibilidad desde los registros reales e incorpora un
  dashboard de conformance ejecutable para LLM, VLM y long-context.
- Añade guardas VLM verificables, incluido un límite de frames para SmolVLM2,
  sin convertir checkpoints ausentes en resultados satisfactorios.
- Amplía `BenchmarkHelpers` con informes JSON comparables, microbenchmarks de
  sampling, detokenización y adaptación de telemetría MTP.

### Rendimiento y experimentación

- Acota el contexto de detokenización del adaptador oficial de
  swift-transformers y reduce buffers CPU transitorios de vídeo/audio sin
  romper tokenizers personalizados 3.x.
- Añade `wht4` y `wht8` como prototipo experimental y opt-in de KV cache con
  Walsh-Hadamard, fallback recuperable y persistencia identificada.

### Validación

- Conserva las APIs 3.x y no introduce breaking changes en `MLXLMCommon`.
- Mantiene la compatibilidad affine 1-bit de Bonsai sobre el runtime M5-safe
  fijado en `JuanColilla/mlx-swift`.
- Valida build macOS, compilación de integración macOS/iOS y 454 tests
  lógicos/489 ejecuciones sin fallos ni omisiones.

### Límites conocidos

- MTP pasa de forma sticky a target-only cuando aparece shared K/V cuantizado.
- WHT KV dequantiza y aplica la transformada inversa por paso; sigue siendo
  experimental y no implica una mejora demostrada de latencia o RSS.
- `LMOutput.State` se conserva dentro de la sesión, pero no se serializa en el
  prompt cache para rehidratación cross-process.
- Los checkpoints con red y las cifras de rendimiento por dispositivo deben
  ejecutarse y registrarse por separado.
