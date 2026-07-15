# Runtime adaptive speculative decoding — scoping note

Backlog: `DOCS/tech-debt-and-research-backlog.md` #6 — "Usar
`SpeculativeDecodingTelemetry` para ajustar o abandonar speculative en
caliente. Investigar pares target/draft mantenidos oficialmente."

Esta es una nota de alcance/diseño, no una implementación. Toca el loop de
generación de `SpeculativeTokenIterator` (`Libraries/MLXLMCommon/Evaluate.swift`),
que es ruta caliente y correctness-crítica compartida por cualquier app que
use speculative decoding clásico. Cambiarlo sin verificación end-to-end con
un modelo real (no solo tests con pesos sintéticos) es exactamente el tipo de
riesgo que este backlog advierte evitar — así que se deja como diseño para
una sesión dedicada, no como código en esta rama.

## Qué existe hoy (verificado leyendo el código actual)

`SpeculativeDecodingTelemetry` (`Libraries/MLXLMCommon/SpeculativeDecoding.swift:14-98`)
ya acumula, en vivo, ronda a ronda: `roundCount`, `draftTokenCount`,
`acceptedDraftTokenCount`, `targetModelCallCount`, `draftModelCallCount`,
`targetVerifiedTokenCount`, `emittedTokenCount`, más las propiedades
derivadas `acceptanceRate`, `meanAcceptedDraftTokensPerRound`,
`meanEmittedTokensPerTargetCall`.

`SpeculativeTokenIterator` (`Evaluate.swift:783-1035`) mantiene esta
telemetría como `private var telemetry` (`Evaluate.swift:814`), expuesta
públicamente vía `speculativeDecodingTelemetry` (`Evaluate.swift:815-817`,
`nil` hasta la primera ronda). Cada llamada a `next()` que agota el buffer
de tokens pendientes invoca `speculateRound()` (`Evaluate.swift:1024`), que
al final llama `telemetry.recordRound(...)` (`Evaluate.swift:978-982`) — es
decir, **la telemetría ya está actualizada y disponible en el momento exacto
en que se decidiría si vale la pena la siguiente ronda especulativa**, antes
de la llamada a `speculateRound()` del siguiente `next()`.

Lo que **no existe**: ningún punto de `speculateRound()` o `next()` lee
`telemetry.acceptanceRate` (o cualquier otra métrica) para decidir nada.
`speculateRound()` se invoca incondicionalmente cada vez que el buffer de
pendientes se vacía (`Evaluate.swift:1021-1024`); `numDraftTokens` es un
`let` fijado una sola vez en `init` (`Evaluate.swift:806`, `840`, `857`) — no
hay ningún mecanismo para reducirlo, ni para abandonar speculative y caer a
generación normal a mitad de stream.

## Qué requeriría implementarlo

1. **Decisión de abandono ("fall back a generación normal")**: `next()`
   tendría que poder, en algún punto, dejar de llamar `speculateRound()` y
   en su lugar hacer una llamada normal `mainModel(...)` de un solo token —
   es decir, `SpeculativeTokenIterator` necesita un segundo modo de
   operación, análogo al `passthrough` que ya existe en
   `MTPSpeculativeTokenIterator` (`Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift`,
   ver `switchToPassthrough`/`passthroughStep`) pero que hoy **no existe**
   para el speculative decoding clásico. Este es el cambio estructural
   principal, no un simple `if`.
2. **Umbral de abandono**: el backlog no fija un número, y no debería
   inventarse uno sin datos — `acceptanceRate` varía enormemente por
   par target/draft y por tarea (código vs. prosa vs. instrucciones cortas).
   Un umbral fijo (p.ej. "abandona si acceptanceRate < 0.3 tras N rondas")
   sin medición real es exactamente el tipo de constante fabricada que este
   documento evita proponer. El paso previo real es instrumentar
   `IntegrationTesting` con pares target/draft reales y medir
   acceptanceRate por tarea antes de fijar cualquier umbral — ver la
   sección "Investigación propuesta" de `DOCS/performance-and-generation.md`,
   que ya pide exactamente esto.
3. **Ajuste de longitud de draft** (en vez de abandono binario): más
   sofisticado — reducir `numDraftTokens` dinámicamente en vez de
   abandonar del todo. Requiere que `numDraftTokens` deje de ser `let`
   (`Evaluate.swift:806`) y que `speculateRound()` decida su propio
   `numDraft` por ronda a partir de la telemetría reciente (no solo
   acumulada desde el inicio — probablemente una ventana móvil de las
   últimas K rondas, ya que `acceptanceRate` es acumulado histórico y una
   mala racha reciente puede quedar diluida por buenas rondas tempranas).
4. **Pares target/draft mantenidos oficialmente**: esto es trabajo de
   catalogación/testing (qué draft models funcionan bien con qué targets),
   no de código — encaja mejor como una tabla en
   `DOCS/compatibility-matrix.md` o un nuevo doc, alimentada por
   `IntegrationTesting`, una vez exista una forma repetible de medir
   acceptance rate por par (ítem #1 del backlog, benchmark suite).

## Recomendación

No implementar el abandono/ajuste adaptativo todavía. Orden correcto:

1. Primero, extender el benchmark harness (`DOCS/tech-debt-and-research-backlog.md`
   #1, ya con soporte JSON vía `BenchmarkReport` en esta rama) para medir
   `acceptanceRate` real por par target/draft y por tipo de prompt.
2. Con datos reales, decidir si un umbral fijo simple basta o si hace falta
   una política más rica (ventana móvil, por tarea, etc.).
3 Solo entonces diseñar el modo `passthrough`-equivalente para
   `SpeculativeTokenIterator`, reutilizando el patrón ya probado en
   `MTPSpeculativeTokenIterator` (que sí tiene un modo passthrough real y
   testeado hoy — ver `Tests/MLXLMTests/MTPQuantizationOnsetTests.swift`),
   con tests sintéticos de la transición **y** verificación con
   `IntegrationTesting` sobre un modelo real, exactamente como se verificó
   el fix de este documento hermano (`DOCS/gemma4-chunked-prefill-investigation.md`)
   debía verificarse antes de tocar código compartido de generación.

Implementar esto sin datos de aceptación reales tiene alto riesgo de fijar
un umbral que ni ayuda (demasiado conservador, nunca abandona) ni protege
(demasiado agresivo, abandona speculative decoding útil) — el propio ticket
original ya apunta a esto al pedir "investigar pares target/draft" antes
del umbral, no después.
