# Plan de continuación — backlog de mejoras (rama `docs/improvement-backlog-review`)

> Documento de handoff. Objetivo: que otra sesión (con este mismo agente u
> otro) pueda retomar exactamente donde se dejó, sin tener que
> re-investigar el entorno ni releer los 19 commits para reconstruir el
> estado. Generado el 2026-07-15 al cerrar la sesión por presupuesto de
> tokens, no por falta de trabajo pendiente identificado.

## Estado exacto al cerrar

- Rama: `docs/improvement-backlog-review`, creada desde `origin/main` en
  `f97da4e`. 19 commits por delante (`git log --oneline f97da4e..HEAD`).
  Árbol de trabajo limpio (`git status --short` sin salida). **Nada
  pusheado a remoto.**
- De los 15 items de `DOCS/tech-debt-and-research-backlog.md`: 5 tienen
  código verificado (build + tests reales pasando), 10 tienen solo
  investigación/nota de alcance. El desglose completo, item por item, está
  en `DOCS/README.md`, sección "Implementación del backlog (2026-07-15)" —
  **leer ese índice primero**, no repetir la investigación.
- Hallazgo adicional no pedido: bug de correctness real en
  `Gemma4ChunkedPrefillTests`, confirmado preexistente en el baseline de
  `origin/main` (no introducido en esta rama), causa raíz investigada con
  alta confianza en `DOCS/gemma4-chunked-prefill-investigation.md`, no
  arreglado.
- Motivo del cierre: instrucción explícita del usuario de no seguir
  lanzando subagentes en paralelo por presupuesto de tokens agotado. No es
  un bloqueo técnico ni una falta de camino a seguir — los cuatro items
  restantes (#3, #6, #7 resto, bug Gemma4) tienen cada uno una nota con
  recomendación concreta de próximo paso, no son un vacío.

## Entorno de build/test — ya resuelto, no re-investigar

Verificado funcional en esta máquina durante esta sesión:

```bash
# Requisito: xcode-select debe apuntar a Xcode.app completo, no Command Line
# Tools (Metal compiler no está en CLT). Verificar con:
xcrun -f metal   # si falla, pedir al usuario: sudo xcode-select -s /Applications/Xcode.app

# swift build/test directos rompen si el repo está bajo iCloud Drive
# (codesign falla por "resource fork... not allowed"). Usar scratch-path
# fuera del árbol sincronizado:
swift build --scratch-path /tmp/mlx-swift-lm-build

# `swift test` (CLI) falla en runtime cargando el metallib por diferencias
# de resolución de resource bundles. Usar xcodebuild en su lugar:
set -o pipefail && xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  2>&1 | tee /tmp/verify.log | tail -60

# Para un test/suite concreto (nótese la sintaxis Swift Testing, no XCTest):
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:MLXLMTests/KVCacheTests/testName\(\) \
  2>&1 | tee /tmp/verify.log | tail -60

# Integration tests con modelos reales (descarga de HF) están en un
# .xcodeproj separado:
xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting -destination 'platform=macOS' \
  2>&1 | tee /tmp/integration.log | tail -60
```

**Trampa de verificación**: nunca confiar en el exit code de un pipe con
`tail`/`tee` sin `set -o pipefail` (y ojo: un `echo "EXIT=$?"` posterior
también lo pisa). Siempre `grep -n "error:\|TEST FAILED\|Build failed\|Failing tests:"`
sobre el log crudo antes de reportar éxito.

**Fallo conocido y ya diagnosticado** (no es una regresión nueva si
reaparece): `Gemma4ChunkedPrefillTests.chunkSizeInvariance` (3 variantes
de parámetro). Confirmado presente en el baseline `origin/main` limpio
(aislado con `git stash push -u` + rebuild + `git stash pop`). Si una
verificación futura muestra solo estos 3 fallos, la rama está limpia. Si
aparece cualquier otro fallo, eso sí es nuevo y hay que investigarlo.

## Próximos pasos concretos, en orden recomendado

Cada uno de estos ya tiene una nota de alcance con el análisis completo —
leerla antes de tocar código, no repetir la investigación:

1. **`estimateKVCacheBytes(...)` — bajo riesgo, buen primer paso.**
   Ver `DOCS/unified-memory-policy-scoping.md`, sección "Recomendación",
   punto 2. Función pura y sin estado, testeable con valores conocidos, no
   toca `ChatSession` ni políticas existentes. Cierra parte del gap de
   item #3 sin comprometerse a la unificación completa.
2. **Componer `WiredBudgetPolicy` con la estimación de KV** (mismo doc,
   mismo punto). Solo después de (1).
3. **Diseñar el ticket de memoria por turno de `ChatSession`** — el paso
   más delicado, requiere decisiones de producto (¿ticket por turno o por
   sesión? ¿cómo compone con MTP?). Ver el mismo doc, sección "Por qué
   `ChatSession` es la parte más delicada". No intentar sin casos de uso
   reales.
4. **Item #6 (speculative decoding adaptativo)**: ver
   `DOCS/adaptive-speculative-decoding-scoping.md` para el análisis y la
   recomendación de próximo paso.
5. **Item #7 resto (MTP en `ChatSession`)**: ver
   `DOCS/mtp-production-scoping.md`. Depende de que (1)-(3) estén
   resueltos primero (el ticket de memoria por turno necesita saber sumar
   el coste del drafter).
6. **Bug de Gemma4**: ver `DOCS/gemma4-chunked-prefill-investigation.md`
   para la causa raíz ya identificada. Antes de tocar `RotatingKVCache`,
   invocar `superpowers:systematic-debugging` desde Phase 3 (ya se
   completó Phase 1-2 en la nota) y verificar el fix con un modelo real
   sliding-window, no solo con el test sintético — `RotatingKVCache` es
   compartida por todos los modelos de esa familia en el repo.

## Cómo verificar que un futuro cambio no rompe nada ya entregado

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:MLXLMTests/KVCacheTests \
  -only-testing:MLXLMTests/CompatibilityMatrixGeneratorTests \
  2>&1 | tee /tmp/verify.log | tail -60
```

Estos dos suites cubren el código realmente verificado de esta rama
(#1/#2/#5/#8). `BenchmarkReport`/`SamplingBenchmarks` no tienen suite de
tests dedicada propia — se verificaron por compilación + ejecución manual
del reporte JSON durante esta sesión, no hay regresión automática que
correr salvo el build general.
