# Política pública de memoria — mecanismos entregados y límites

Backlog: `DOCS/tech-debt-and-research-backlog.md` #3 — "Unificar
`GenerateParameters`, wired memory, KV estimate y speculative memory policy.
Producir recomendaciones explicables, no solo errores OOM. Incluir
`ChatSession`."

La entrega mantiene mecanismos explícitos y componibles: estimación pura de KV,
presupuesto wired, gates de memoria para modelos auxiliares y ticket por turno
en `ChatSession`. No introduce un autopilot que cambie `GenerateParameters` o
invente prioridades de producto sin datos físicos.

## Qué existe hoy — mecanismos coordinables, no un único autopilot

### 1. `WiredMemoryPolicy` / tickets (`Libraries/MLXLMCommon/WiredMemoryPolicies.swift`,
`WiredMemoryUtils.swift`)

Sistema ya bastante completo: cuatro políticas (`WiredSumPolicy`, `WiredMaxPolicy`,
`WiredFixedPolicy`, `WiredBudgetPolicy`, todas `WiredMemoryPolicy`), cada una
computando un `limit(baseline:activeSizes:)` y opcionalmente `canAdmit(...)`, más
un mecanismo de "ticket" (`WiredMemoryTicket.withWiredLimit(_:_:)`) que aplica el
límite de memoria wired de MLX durante la ejecución del bloque.

`generateTask` aplica el ticket alrededor de la tarea de generación con cierre
emparejado incluso bajo cancelación. `ChatSession.wiredMemoryTicket` se captura
al empezar cada turno y se propaga a generación normal, speculative clásico y
MTP. La app sigue siendo responsable de crear el ticket con la política y el
presupuesto apropiados para su dispositivo.

### 2. `SpeculativeDecodingMemoryPolicy` (`Libraries/MLXLMCommon/SpeculativeDecoding.swift:147-199`)

Sistema explícito para modelos auxiliares: `limitBytes`,
`additionalBytes`, `action` (`.allow`/`.fallbackToDefault`/`.fail`), con
`evaluate(mainModelBytes:draftModelBytes:)` devolviendo un veredicto explicable
(`SpeculativeDecodingMemoryEvaluation`, con `estimatedBytes`, `isWithinBudget`,
`shouldUseSpeculativeDecoding`). Ya tiene, de hecho, exactamente la propiedad que
el backlog pide para el sistema unificado ("recomendaciones explicables, no solo
errores OOM"). Se aplica tanto al par main/draft clásico como al target/drafter
MTP, con estimación antes de carga y medición de parámetros después de carga.
No muta KV cache ni comparte ciclo de vida con `WiredMemoryPolicy`.

### 3. Estimación de KV cache: API pública pura

`Libraries/MLXLMCommon/KVCacheMemory.swift` expone ahora:

```swift
public func estimateKVCacheBytes(
    numLayers: Int,
    kvHeads: Int,
    headDim: Int,
    maxTokens: Int,
    kvBits: Int? = nil,
    kvGroupSize: Int = 64,
    bytesPerElement: Int = 2
) throws -> Int
```

La ruta sin cuantizar calcula `layers × heads × headDim × tokens × 2
(K+V) × bytesPerElement`. La ruta affine incluye el payload empaquetado y una
escala y bias por grupo; resuelve el group size igual que `QuantizedKVCache`
entre 32/64/128. La función rechaza valores negativos, bytes o bits no soportados,
head dimensions incompatibles y cualquier overflow de `Int` con
`KVCacheMemoryEstimationError`.

Es una estimación de almacenamiento lógico: no incluye spare capacity por el
crecimiento en bloques, alineación del allocator ni workspace temporal. Cuando
hay un modelo/dispositivo disponible, `WiredMemoryUtils.tune(...)` sigue siendo
la fuente de verdad empírica.

`WiredBudgetPolicy` conserva el inicializador histórico y añade una composición
explícita con `kvCacheBytes`. `baseBytes`, `kvCacheBytes` y `totalBaseBytes`
quedan inspeccionables por separado; los cálculos de limit/admission no crashean
si una suma desborda.

```swift
let kvBytes = try estimateKVCacheBytes(
    numLayers: 28, kvHeads: 8, headDim: 128, maxTokens: 8_192, kvBits: 4)
let policy = WiredBudgetPolicy(
    baseBytes: weightsBytes + workspaceBytes,
    kvCacheBytes: kvBytes,
    cap: deviceBudgetBytes
)
```

### 4. `GenerateParameters`: configuración manual y explicable

`maxKVSize`, `kvBits`, `kvGroupSize`, `quantizedKVStart` (`Evaluate.swift:63-79`)
son parámetros de generación puros — ninguno de ellos alimenta ni es alimentado
por `WiredMemoryPolicy` ni por `SpeculativeDecodingMemoryPolicy`. Elegir
`maxKVSize` hoy es una decisión manual del desarrollador de la app, no una
recomendación derivada de un presupuesto de memoria declarado.

## Por qué unificarlos no es trivial

Los tres sistemas resuelven preguntas distintas con distinta forma de resultado:

- `WiredMemoryPolicy` responde "¿qué límite de memoria wired debo pedirle al
  sistema operativo ahora mismo?", como un número que cambia dinámicamente según
  qué tickets están activos concurrentemente.
- `SpeculativeDecodingMemoryPolicy` responde "¿me cabe cargar un segundo modelo
  (draft) además del principal?", como una decisión binaria/de acción tomada
  **antes** de cargar, no un límite dinámico.
- La estimación de KV cache responde "¿cuántos bytes requiere esta
  configuración?". Todavía no recomienda por sí sola qué `maxKVSize`/`kvBits`
  elegir ni muta `GenerateParameters`.

Unificarlos de verdad (no solo ponerlos bajo el mismo namespace) requiere decidir
un modelo de datos común — probablemente algo como un `MemoryBudget` que sepa
convertirse en: un `WiredMemoryPolicy` concreto, una respuesta compatible con
`SpeculativeDecodingMemoryEvaluation`, y una recomendación de `GenerateParameters`
— y decidir el orden de prioridad cuando piden más de lo que cabe (¿se recorta
primero `maxKVSize`, se rechaza el draft model, o se activa `kvBits`
automáticamente?). Esa política de prioridad es una decisión de producto, no solo
de ingeniería, y no debería inventarse sin casos de uso reales — el propio
`DOCS/memory-kv-cache.md` (sección "Mejoras propuestas", punto 1) ya lo enmarca
como "recomendaciones... advertencias... estimación de bytes", es decir, un
sistema consultivo, no uno que decida unilateralmente por la app.

## Decisión implementada en `ChatSession`

`ChatSession` sigue siendo single-consumer y conserva estado mutable público.
La implementación toma snapshots al iniciar el stream y adopta estas reglas:

- El ticket wired es **por turno**: se captura al empezar la generación y se
  cierra con su `generateTask`. No pretende presupuestar durante todo el tiempo
  que la KV cache queda retenida entre turnos.
- La admisión del modelo auxiliar es independiente y explicable. Classic y MTP
  evalúan su `SpeculativeDecodingMemoryPolicy`; MTP puede evitar un loader
  deferred antes de cargar y vuelve a medir después.
- `additionalContext`, `generateParameters`, tools y ticket se capturan por
  stream. Cambiarlos afecta al turno siguiente, no a una tarea ya iniciada.
- La KV cache estimada puede incluirse explícitamente en `WiredBudgetPolicy` o
  en `additionalBytes` del gate auxiliar. No se suma dos veces de forma
  implícita.

## Límites y recomendación de uso

1. Construir el presupuesto desde datos del dispositivo y del modelo; usar
   `estimateKVCacheBytes(...)` como estimación lógica, no como pico RSS exacto.
2. Reservar `additionalBytes` para KV/workspace/headroom cuando se evalúe un
   draft/drafter, y elegir conscientemente entre `.fallbackToDefault` y `.fail`.
3. Medir con Instruments o con los benchmarks JSON antes de publicar umbrales
   adaptativos. El framework entrega mecanismos, no valores universales.
4. Mantener `GenerateParameters` bajo control de la app. Decidir si reducir
   primero contexto, activar KV quantization o rechazar el auxiliar sigue siendo
   una política de producto y no se automatiza en esta versión.
