# Plan de continuación — backlog de mejoras

Fecha de actualización: 2026-07-15.

## Estado de release

La rama `feature/complete-improvement-backlog` parte de `main` en `ac178f6` y
cierra los 11 bloques operativos definidos para esta continuación. El estado,
los commits y los límites están en
[`backlog-completion-status.md`](backlog-completion-status.md).

La composición final conserva:

- la compatibilidad Bonsai affine 1-bit y su validación física confirmada;
- el runtime M5-safe fijado en
  `JuanColilla/mlx-swift@5e27a4cb2604599c72615cf058e09801c123b831`;
- compatibilidad de API 3.x, comprobada contra `main`;
- documentación separada entre comportamiento implementado, prototipos
  experimentales y mediciones físicas todavía necesarias.

No se ha creado tag ni se ha publicado esta rama: esos pasos pertenecen al
flujo de release posterior al merge.

## Correcciones de la pasada de release

- Las APIs históricas `KVCacheSimple.toQuantized(...)` y
  `RotatingKVCache.toQuantized(...)` conservan su firma no throwing para no
  romper consumidores 3.x. La nueva API `quantized(...) throws` permite manejar
  el error de forma recuperable y `KVCacheError` es público.
- `BenchmarkReport` tiene cobertura de serialización, carga y comparación entre
  runs.
- El hallazgo Gemma4 se considera cerrado para este árbol: las cuatro variantes
  de `chunkSizeInvariance` pasan con el runtime M5-safe. Ver
  `DOCS/gemma4-chunked-prefill-investigation.md`.
- El índice conserva tanto Bonsai como los documentos del backlog.

## Build y tests

Requisito de esta máquina:

```bash
xcrun -f metal
```

Build CLI con artefactos fuera del árbol:

```bash
swift build --scratch-path /private/tmp/mlx-swift-lm-release-build
```

Suite completa recomendada:

```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -derivedDataPath /private/tmp/mlx-swift-lm-release-derived
```

Suites con nombre estable para una comprobación corta:

```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -derivedDataPath /private/tmp/mlx-swift-lm-release-derived \
  -only-testing:MLXLMTests/BenchmarkReportTests \
  -only-testing:MLXLMTests/CompatibilityMatrixGeneratorTests \
  -only-testing:MLXLMTests/BonsaiOneBitCompatibilityTests
```

Los tests de `KVCacheTests.swift` son funciones Swift Testing de nivel de
archivo. El selector histórico `MLXLMTests/KVCacheTests` no garantiza que se
ejecuten; para el gate de release se usa la suite completa y se comprueba el
`.xcresult` estructurado.

## Gate de compatibilidad pública

Antes de publicar una variante 3.x:

```bash
swift package \
  --scratch-path /private/tmp/mlx-swift-lm-release-api-diff \
  diagnose-api-breaking-changes main \
  --targets MLXLMCommon
```

El comando debe terminar sin `API breakage`. No mantener un cambio breaking en
una variante patch sólo porque compila o porque los tests internos pasan.

## Trabajo futuro — no bloquea este release

1. Recoger benchmarks físicos reproducibles por dispositivo/modelo para
   long-context, speculative adaptativo, MTP y materialización de media.
2. Ejecutar los checkpoints de red del dashboard y conservar sus result
   bundles; hasta entonces deben seguir como `not_run`.
3. Evaluar si WHT KV merece optimización fused antes de promoverlo fuera de su
   estado experimental.
4. Diseñar una representación persistible de `LMOutput.State` solo si aparece
   un requisito real de rehidratación cross-process.
5. Medir RSS con Instruments para cuantificar los ahorros de vídeo/audio y
   decidir si el resultado multimedia final también debe ser incremental.

Ninguno de estos puntos invalida la corrección o compatibilidad de la API
entregada; sí limita las afirmaciones de rendimiento que pueden publicarse.

## Gate final de publicación

- `git diff --check` limpio;
- formato limpio en todos los ficheros modificados;
- build completo aprobado;
- suite completa aprobada y revisada mediante `.xcresult`;
- cero breaking changes frente a `main` anterior;
- documentación sin afirmaciones obsoletas;
- tag anotado con sufijo `v`, después de comprobar el último tag oficial.

La validación local del 2026-07-15 aprobó los primeros seis puntos: build
macOS, 454 tests lógicos/489 ejecuciones sin fallos ni skips, formato, diff,
dashboard offline, compilación de integración macOS/iphoneos y cero breaking
changes en `MLXLMCommon`. El tag continúa pendiente por diseño.
