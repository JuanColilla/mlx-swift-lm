# MTP de producción — scoping note

Backlog: `DOCS/tech-debt-and-research-backlog.md` #7 — "Integrar MTP en
`ChatSession`, añadir config pública, benchmark oficial y soporte/fallback
claro para KV cuantizado. Sustituir prints directos por logging o eventos."

Nota de alcance, no implementación completa. Un ítem de los cuatro
("sustituir prints directos") ya se resolvió en esta rama (ver
`git log --oneline -- Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift`)
porque era un cambio aislado, de una línea, fuera del hot path por-token. Los
otros tres son cambios estructurales sobre generación real y quedan
documentados aquí, no implementados.

## Qué existe hoy (verificado)

- `MTPSpeculativeTokenIterator` (`Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift`)
  es una API completa y testeada a nivel de iterador de bajo nivel:
  `TokenIteratorProtocol`, con `blockSize`, telemetría (`SpeculativeDecodingTelemetry`
  reutilizada), y **ya tiene** un modo `passthrough` real (no un TODO):
  `switchToPassthrough(reason:)` cambia a generación de un solo token cuando
  el target deja de emitir `sharedKV`/hidden state (p.ej. al cruzar el
  umbral de cuantización de KV, cubierto explícitamente por
  `IntegrationTesting/IntegrationTestingTests/MTPQuantizationOnsetTests.swift`
  — el "R13" mencionado
  en el doc comment del fichero, `MTPSpeculativeTokenIterator.swift:26`).
  Es decir: **el fallback a KV cuantizado ya existe y ya tiene test
  dedicado** — el gap real no es "no hay fallback", es que ese fallback
  vive únicamente en la API de bajo nivel, invisible para quien use
  `ChatSession`.
- Tests de integración reales ya cubren MTP con modelos descargados:
  `IntegrationTesting/IntegrationTestingTests/MTPAcceptanceRateTests.swift`,
  `MTPIteratorEndToEndDiagnosticTests.swift`, `MTPRung4TokenParityTests.swift`,
  `MTPDrafterModelFactoryIntegrationTests.swift`, `Gemma4AssistantDraftModelIntegrationTests.swift`
  — así que "benchmark oficial" tiene más sustrato del que el ticket sugiere;
  el gap es que no está en el formato JSON comparable entre commits que
  `BenchmarkReport` (añadido en esta rama, ítem #1) ya habilita para otros
  benchmarks.

## Qué falta de verdad

1. **Integración en `ChatSession`**: `ChatSession` (`Libraries/MLXLMCommon/ChatSession.swift`)
   no construye ni ofrece `MTPSpeculativeTokenIterator` en ningún punto —
   confirmado por `grep -n "MTP" Libraries/MLXLMCommon/ChatSession.swift`
   sin resultados. Un usuario de `ChatSession` (la API de alto nivel que la
   mayoría de apps usa) no tiene forma de pedir MTP; solo quien construye
   `TokenIteratorProtocol` a mano lo tiene disponible. Esto es el gap
   estructural real del ticket.
2. **Config pública**: no existe un `MTPSpeculativeDecodingConfig` (el
   ticket lo menciona por nombre pero no existe en el código —
   `grep -rn "MTPSpeculativeDecodingConfig" Libraries/` no devuelve nada).
   `blockSize` y el drafter se pasan hoy directamente al inicializador de
   `MTPSpeculativeTokenIterator`, sin un tipo de configuración reusable ni
   almacenable en `GenerateParameters`.
3. **Benchmark oficial en formato comparable**: los tests de integración
   miden acceptance rate (`MTPAcceptanceRateTests.swift`) pero no emiten
   `BenchmarkReport` JSON — sería trabajo de conectar esos tests (o una
   nueva función en `BenchmarkHelpers`, siguiendo el patrón de
   `benchmarkLLMGeneration`) al tipo `BenchmarkReport` ya añadido en esta
   rama.

## Por qué no se implementa aquí

Integrar MTP en `ChatSession` significa tocar el tipo más usado y con más
estado mutable público del repo (`ChatSession` ya se documenta a sí mismo
como no thread-safe con estado mutable — ver `DOCS/memory-kv-cache.md`
sección "Estado actual"), decidiendo cómo convive con:
- Reuso de KV cache entre turnos (`ChatSession` ya evita re-prefill de
  historial — cualquier MTP config tiene que respetar esa invariante).
- El modo `passthrough` de MTP, que hoy es *sticky* dentro de un solo
  `MTPSpeculativeTokenIterator` — ¿debe reiniciarse en cada turno de
  `ChatSession`, o permanecer sticky entre turnos si el KV cache sigue
  cuantizado? Esto no tiene una respuesta obvia sin decidir primero el
  diseño de la política de memoria unificada (backlog #3, tampoco
  implementada en esta rama por la misma razón).
- `additionalContext`/`wiredMemoryTicket` por turno (gap ya documentado en
  `DOCS/memory-kv-cache.md` punto 5, "`ChatSession` con política de
  memoria") — MTP añade un segundo modelo (drafter) cuyo presupuesto de
  memoria también debería entrar en esa misma política, no en una paralela.

Diseñar esto bien requiere resolver primero el diseño de política de
memoria de `ChatSession` (#3) — hacerlo al revés (MTP primero) arriesga
tener que rehacer la integración cuando #3 se resuelva.

## Recomendación de orden

1. Backlog #3 (política de memoria de `ChatSession`) primero — define cómo
   `ChatSession` expone/gestiona presupuesto de memoria por turno para
   *cualquier* modelo auxiliar (draft clásico o drafter MTP), no solo MTP.
2. Con ese diseño fijado, añadir `MTPSpeculativeDecodingConfig` (struct
   simple: drafter, `blockSize`, política de memoria) y un punto de entrada
   en `ChatSession` que lo acepte, reutilizando `MTPSpeculativeTokenIterator`
   tal cual (no reescribirlo — ya está testeado).
3. Conectar `MTPAcceptanceRateTests`/`MTPIteratorEndToEndDiagnosticTests` a
   `BenchmarkReport` para tener benchmark oficial en JSON comparable.

Ningún paso de esta lista requiere tocar la lógica interna de
`MTPSpeculativeTokenIterator` (draft/verify/accept, passthrough) — está
verificada y estable; el trabajo real es de superficie pública y
composición con `ChatSession`, no de algoritmo.
