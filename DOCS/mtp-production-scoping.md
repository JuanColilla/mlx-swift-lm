# MTP de producción en `ChatSession`

Backlog: `DOCS/tech-debt-and-research-backlog.md` #7 — integrar MTP en la API
de chat de alto nivel, con configuración pública, política de memoria,
telemetría y benchmark JSON comparable.

## Estado

Implementado en la rama `feature/complete-improvement-backlog`.

La implementación reutiliza `MTPSpeculativeTokenIterator`; no duplica el
algoritmo draft/verify/accept. El cambio de producción está en la composición:
`ChatSession` puede resolver el drafter, decidir si cabe, conservar el estado
del target entre turnos y exponer el resultado por su stream normal.

## API pública

`MTPSpeculativeDecodingConfig` es deliberadamente independiente de
`SpeculativeDecodingConfig`, porque un drafter MTP no es un `LanguageModel` y
no tiene KV cache propio.

```swift
let config = try MTPSpeculativeDecodingConfig(
    drafter: loadedDrafter,
    blockSize: 4,
    memoryPolicy: .recommendedWorkingSet
)

let session = ChatSession(
    targetContainer,
    mtpSpeculativeDecoding: config,
    generateParameters: GenerateParameters(maxTokens: 512)
)
```

`blockSize` debe ser al menos 2: un bonus/corrección del target y uno o más
tokens propuestos. Los initializers históricos de `ChatSession` y las rutas
sin config o con speculative clásico se conservan; MTP se selecciona mediante
un overload aditivo cuyo label `mtpSpeculativeDecoding:` es obligatorio. Esto
también impide configurar classic y MTP simultáneamente por accidente.

Los overloads MTP cubren `ModelContainer` y `ModelContext`, incluidos los
initializers con historial rehidratado y KV cache preconstruida.

## Carga eager y deferred

Para evitar cargar un auxiliar que no cabe, se puede proporcionar una
estimación y un loader:

```swift
let config = try MTPSpeculativeDecodingConfig(
    drafterBytes: estimatedDrafterBytes,
    blockSize: 4,
    memoryPolicy: SpeculativeDecodingMemoryPolicy(
        limitBytes: deviceBudgetBytes,
        additionalBytes: kvAndWorkspaceBytes,
        action: .fallbackToDefault
    )
) {
    try await MTPDrafterModelFactory.shared.loadContainer(
        configuration: drafterConfiguration
    )
}
```

`drafterBytes` negativo se normaliza a cero, igual que en
`SpeculativeDecodingConfig`. La aplicación es responsable de proporcionar una
estimación conservadora; el framework no inventa un umbral de hardware.

La resolución usa dos gates:

1. Antes del loader, evalúa pesos del target + estimación del drafter +
   `additionalBytes`.
2. Después de cargar, vuelve a evaluar con los bytes reales de parámetros del
   drafter.

Con `.fallbackToDefault`, una denegación ejecuta generación target-only y no
invoca el loader cuando falla el primer gate. Con `.fail`, propaga
`SpeculativeDecodingMemoryError`. Un drafter deferred admitido se cachea en la
sesión y no se vuelve a cargar en cada turno.

## Memoria wired y prefill

`ChatSession.wiredMemoryTicket` se captura al comenzar cada turno. La ruta MTP
lo pasa al mismo `generateTask` que la generación normal y classic speculative,
por lo que start/end mantienen el emparejamiento cancellation-safe.

`ChatSession.prefill(...)` solo prepara el target y su KV cache. No carga el
drafter MTP: este no mantiene una caché propia y se resuelve cuando realmente
empieza una generación MTP. Así se puede construir contexto reutilizable sin
pagar memoria auxiliar ni emitir tokens visibles.

## Coherencia de KV cache y `LMOutput.State`

La sesión conserva una única KV cache del target para MTP; no existe
`draftKVCache` en este modo. El input de cada turno se añade a esa cache y el
iterador recibe también el `LMOutput.State` asociado.

Hay dos salvaguardas adicionales:

- Los iteradores classic y MTP tienen initializers package aditivos que aceptan
  el state de continuación. Las firmas públicas históricas siguen inicializando
  desde `nil`.
- Un state sink de un solo consumidor recoge el state final del iterador tras
  cada token. `ChatSession` espera a `genTask` antes de leerlo y guardarlo junto
  a la KV cache, incluyendo reinicios por tool calls.

En MTP, los snapshots `mtpLastHiddenStatesKey` y `mtpSharedKVStatesKey` de un
turno anterior se eliminan antes de un nuevo prefill. El forward con
`mtpEmitFlagKey = true` produce un snapshot nuevo alineado con el offset actual;
esto evita reutilizar shared K/V stale y disparar la invariante de span.

## Fallbacks observables

MTP no soporta verificar con shared K/V cuantizado. El comportamiento es
explícito y no crashea:

- Si el target deja de emitir hidden/shared K/V —incluido el onset de KV
  cuantizado— `MTPSpeculativeTokenIterator` pasa de forma sticky a target-only.
- El fallback sigue encadenando `LMOutput.State` y la KV cache del target.
- El motivo de cuantización contiene `after KV cache quantization`; un estado
  faltante por otra causa conserva `main model did not emit drafter state`.
- Si el gate de memoria evita entrar en MTP, un wrapper target-only publica
  igualmente `proposedDraftTokens = 0`, `acceptedDraftTokens = 0` y un
  `passthroughReason` que comienza por `MTP skipped by memory policy`.

La app obtiene la telemetría desde el evento `.info` habitual:

```swift
for try await event in session.streamDetails(to: prompt) {
    guard let info = event.info else { continue }
    let proposed = info.proposedDraftTokens
    let accepted = info.acceptedDraftTokens
    let reason = info.passthroughReason
    let detailed = info.speculativeDecodingTelemetry
}
```

## Benchmark JSON reproducible

`BenchmarkHelpers` expone el adaptador
`GenerateCompletionInfo.mtpBenchmarkEntry(...)`. Los integration tests reales
pueden conservar sus mediciones de modelo/dispositivo y producir el mismo
schema JSON que el resto de benchmarks:

```swift
let entry = completionInfo.mtpBenchmarkEntry(
    context: "target=<id>;drafter=<id>;blockSize=4;kv=fp16",
    generationTimesMilliseconds: repeatedDecodeTimes
)

let report = BenchmarkReport(
    label: commitSHA,
    entries: [entry]
)
try report.write(to: outputURL)
```

El entry incluye `tokensPerSecond`, `generationTokenCount`,
`proposedDraftTokens`, `acceptedDraftTokens`, `acceptanceRate` y
`didPassthrough`. El `context` debe fijar target, drafter, block size,
cuantización y preset para que dos commits sean comparables.

Los tests unitarios validan el mapeo y el JSON. Las suites físicas existentes
(`MTPAcceptanceRateTests`, `MTPIteratorEndToEndDiagnosticTests` y
`MTPQuantizationOnsetTests`) siguen siendo la fuente para números reales: MLX
en iOS debe medirse en dispositivo físico, no en Simulator.

## Límites explícitos

- MTP requiere un target que implemente las claves de emisión; actualmente la
  ruta de producción relevante es Gemma 4.
- KV cuantizado usa target-only sticky; no se promete MTP cuantizado.
- La política de memoria es consultiva/explicativa. No cambia `kvBits`,
  `maxKVSize` ni el block size automáticamente.
- `ChatSession` sigue siendo single-consumer y no thread-safe.
- Persistir KV cache con `saveCache` no serializa un `LMOutput.State` arbitrario;
  tras restaurar, el siguiente prefill debe reconstruir el state que el modelo
  necesite.
