# Plan de continuación — backlog de mejoras

Fecha de actualización: 2026-07-15.

## Estado de release

La rama `docs/improvement-backlog-review` ya integra el `main` del fork que
contiene:

- el `main` oficial hasta `10e0cb7`;
- la compatibilidad Bonsai affine 1-bit;
- el runtime M5-safe fijado en
  `JuanColilla/mlx-swift@5e27a4cb2604599c72615cf058e09801c123b831`;
- los cambios de implementación y documentación del backlog.

De los 15 ítems originales, cinco tienen una entrega de código verificable y
los diez restantes tienen una guía o nota de alcance. Esto no significa que
los quince estén implementados: el índice exacto está en `DOCS/README.md`,
sección "Implementación del backlog (2026-07-15)".

La política de release es integrar sólo la parte madura y conservar como
trabajo futuro los cambios que requieren decisiones públicas de arquitectura,
datos de modelos reales o validación física adicional.

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

1. Añadir `estimateKVCacheBytes(...)` como función pura y cubierta por tests.
2. Evaluar su composición con `WiredBudgetPolicy` usando casos de uso reales.
3. Diseñar el ticket de memoria de `ChatSession` por turno o por sesión.
4. Recoger acceptance rate real antes de fijar speculative decoding adaptativo.
5. Integrar MTP en `ChatSession` después de resolver la política de memoria.
6. Ejecutar la validación física de Bonsai descrita en
   `DOCS/bonsai-1bit-compatibility.md` antes de distribuir el modelo a usuarios.

Los puntos 2–5 afectan superficies públicas o hot paths de generación y no se
deben introducir como remate de una versión sin casos de producto y pruebas
end-to-end con modelos reales.

## Gate final de publicación

- `git diff --check` limpio;
- formato limpio en todos los ficheros modificados;
- build completo aprobado;
- suite completa aprobada y revisada mediante `.xcresult`;
- cero breaking changes frente a `main` anterior;
- documentación sin afirmaciones obsoletas;
- tag anotado con sufijo `v`, después de comprobar el último tag oficial.
