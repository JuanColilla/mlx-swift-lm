# Runtime adaptive speculative decoding

Backlog: `DOCS/tech-debt-and-research-backlog.md` #6 — usar
`SpeculativeDecodingTelemetry` para abandonar speculative decoding en caliente
cuando la aceptación medida del par target/draft no justifica seguir usando el
modelo auxiliar.

## Estado

Implementado como API pública aditiva y **opt-in** para speculative decoding
clásico. Si el caller no pasa una política, el iterador conserva exactamente el
comportamiento fijo anterior durante toda la generación.

La implementación no incluye un umbral global ni afirma resultados de ningún
dispositivo. La tasa útil depende del par target/draft, la distribución de
prompts, los parámetros de sampling y el hardware; el caller debe obtener los
valores de su propio benchmark.

## API

`AdaptiveSpeculativeDecodingPolicy` exige cuatro valores explícitos:

- `warmUpRounds`: mínimo de rondas speculative totales antes de evaluar;
- `observationWindowRounds`: tamaño de la ventana móvil de rondas recientes;
- `minimumObservedDraftTokens`: mínimo de propuestas dentro de esa ventana;
- `minimumAcceptanceRate`: tasa mínima medida, inclusiva. Solo una tasa
  estrictamente inferior activa el fallback.

El inicializador es `throws` y rechaza rondas/muestras no positivas y tasas no
finitas o fuera de `0...1`. No existe `default`, `recommended` ni heurística
dependiente del dispositivo.

La política puede pasarse a:

- `SpeculativeTokenIterator(..., adaptivePolicy:)`;
- `generate(..., draftModel:, adaptivePolicy:)`;
- `generateTokens(..., draftModel:, adaptivePolicy:)`.

Ejemplo en el que los valores proceden de un perfil medido por la app:

```swift
let adaptivePolicy = try AdaptiveSpeculativeDecodingPolicy(
    warmUpRounds: measuredProfile.warmUpRounds,
    observationWindowRounds: measuredProfile.windowRounds,
    minimumObservedDraftTokens: measuredProfile.minimumDraftSamples,
    minimumAcceptanceRate: measuredProfile.acceptanceRateFloor
)

let stream = try generateTokens(
    input: input,
    parameters: parameters,
    context: targetContext,
    draftModel: draftModel,
    numDraftTokens: measuredProfile.draftLength,
    adaptivePolicy: adaptivePolicy
)
```

## Semántica de decisión

`AdaptiveSpeculativeDecodingController` registra únicamente rondas realmente
completadas por `SpeculativeTokenIterator`. Para cada ronda conserva
`drafted`/`accepted`, recorta el historial a las últimas
`observationWindowRounds` y sigue este orden:

1. espera a `warmUpRounds`;
2. espera a que la ventana esté completa;
3. espera a que la ventana contenga `minimumObservedDraftTokens` propuestas;
4. calcula `accepted / drafted` sobre esa ventana reciente;
5. continúa en speculative si la tasa es mayor o igual al umbral;
6. cambia una sola vez a autoregresivo si es estrictamente inferior.

El cambio es sticky: no se vuelve a cargar el draft en la misma generación y
no hay oscilación entre modos. Esto mantiene explicable el coste y evita tomar
decisiones repetidas sobre ventanas que ya no reciben nuevas rondas
speculative.

## Equivalencia en la transición

La decisión se toma al terminar una ronda, después de verificar, aceptar y
rebobinar los caches. El iterador primero entrega todos los tokens pendientes
de esa ronda. Después procesa con el target el último token de
corrección/bonus (`y`) contra el `mainCache` ya recortado y continúa a un token
por forward.

Se preservan durante el cambio:

- el `LogitProcessor` canónico, actualizado solo con tokens aceptados/emitidos;
- el `LogitSampler` del caller;
- `mainState` y el KV cache del target;
- cuantización dinámica del KV cache;
- `maxTokens` y el contador de tokens emitidos.

El cache del draft deja de usarse. No se descarga el modelo ni se modifica su
ciclo de vida: esa decisión pertenece al caller que lo comparte o retiene.

## Telemetría explicable

`SpeculativeDecodingTelemetry.adaptive` es `nil` sin política. Cuando está
activa contiene `AdaptiveSpeculativeDecodingTelemetry` con:

- la política exacta usada;
- estado: `warmingUp`, `collectingWindow`, `insufficientDraftTokens`,
  `monitoring` o `autoregressive`;
- número de ventanas realmente evaluadas;
- llamadas al target realizadas después del cambio a autoregresivo;
- rondas, propuestas y aceptaciones de la última ventana;
- `observedAcceptanceRate` calculada sobre esa ventana;
- ronda total del fallback y motivo estructurado
  `acceptanceRateBelowMinimum`, si ocurrió.

La telemetría acumulada previa (`roundCount`, `draftTokenCount`,
`acceptedDraftTokenCount`, calls y tokens emitidos) se mantiene compatible.
Al pasar a autoregresivo, rondas y contadores del draft dejan de crecer;
`targetModelCallCount`/`targetVerifiedTokenCount` incluyen los forwards
target-only y `emittedTokenCount` sigue reflejando el stream completo.

## Cobertura sintética

`Tests/MLXLMTests/AdaptiveSpeculativeDecodingTests.swift` cubre:

- validación de cada entrada de la política, incluidos `NaN` e infinitos;
- independencia entre warm-up y ventana;
- mínimo de muestras dentro de la ventana;
- igualdad exacta con el umbral (no abandona) y caída estricta (abandona);
- ventana móvil reciente frente a la media acumulada histórica;
- transición sticky y señal emitida una sola vez;
- equivalencia token a token con generación target-only, con y sin penalty
  processors, usando logits deterministas de alto margen;
- ausencia de nuevas llamadas al draft después del fallback;
- permanencia en speculative con aceptación alta;
- compatibilidad del camino anterior cuando `adaptivePolicy == nil`;
- propagación del estado final dentro de `GenerateCompletionInfo` mediante
  `SpeculativeDecodingTelemetry`.

Los tests sintéticos prueban el contrato y la transición sin atribuir
rendimiento a hardware. Para adoptar la política en producto sigue siendo
necesario medir target/draft/prompt reales en build Release y comparar tiempo,
energía y memoria además de acceptance rate.

## Fuera de alcance

- Elegir o publicar pares target/draft recomendados.
- Definir un umbral universal.
- Ajustar dinámicamente `numDraftTokens`.
- Volver de autoregresivo a speculative dentro de la misma generación.
- Descargar o liberar automáticamente el draft model al cambiar de modo.
- Activar esta política por defecto en `ChatSession`.
- Inferir mejora de rendimiento únicamente a partir de acceptance rate.
