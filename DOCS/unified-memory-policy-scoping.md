# Política pública de memoria unificada — scoping note

Backlog: `DOCS/tech-debt-and-research-backlog.md` #3 — "Unificar
`GenerateParameters`, wired memory, KV estimate y speculative memory policy.
Producir recomendaciones explicables, no solo errores OOM. Incluir
`ChatSession`, que hoy no expone wired-memory ticket por turno."

Esta nota empezó como documento de alcance. La primera fase de bajo riesgo ya
está implementada: estimación pura de KV y composición aditiva, explícita, con
`WiredBudgetPolicy`. La unificación automática con `GenerateParameters`,
speculative decoding y el ciclo de vida de `ChatSession` sigue fuera de alcance
hasta decidir prioridades de producto y medir sesiones reales.

## Qué existe hoy — tres sistemas de memoria separados, sin unificar

### 1. `WiredMemoryPolicy` / tickets (`Libraries/MLXLMCommon/WiredMemoryPolicies.swift`,
`WiredMemoryUtils.swift`)

Sistema ya bastante completo: cuatro políticas (`WiredSumPolicy`, `WiredMaxPolicy`,
`WiredFixedPolicy`, `WiredBudgetPolicy`, todas `WiredMemoryPolicy`), cada una
computando un `limit(baseline:activeSizes:)` y opcionalmente `canAdmit(...)`, más
un mecanismo de "ticket" (`WiredMemoryTicket.withWiredLimit(_:_:)`) que aplica el
límite de memoria wired de MLX durante la ejecución del bloque.

**Consumido en un único punto de todo el repo**: `Evaluate.swift:1948`, dentro de
una función libre de generación (no dentro de `ChatSession`). Confirmado por
`grep -rln "WiredMemoryPolicy\|\.ticket(\|withWiredLimit" Libraries/` — los únicos
ficheros que aparecen son `ModelFactory.swift`, `WiredMemoryPolicies.swift` (su
propia definición), `Evaluate.swift`, y `ModelContainer.swift`. `ChatSession.swift`
no aparece en absoluto: **confirma literalmente la frase del backlog item**, no es
una descripción aproximada.

### 2. `SpeculativeDecodingMemoryPolicy` (`Libraries/MLXLMCommon/SpeculativeDecoding.swift:147-199`)

Sistema separado, específico de speculative decoding clásico: `limitBytes`,
`additionalBytes`, `action` (`.allow`/`.fallbackToDefault`/`.fail`), con
`evaluate(mainModelBytes:draftModelBytes:)` devolviendo un veredicto explicable
(`SpeculativeDecodingMemoryEvaluation`, con `estimatedBytes`, `isWithinBudget`,
`shouldUseSpeculativeDecoding`). Ya tiene, de hecho, exactamente la propiedad que
el backlog pide para el sistema unificado ("recomendaciones explicables, no solo
errores OOM") — pero **solo para el par de modelos main/draft**, no para KV cache
ni para wired memory. No comparte tipo, ni composición, con `WiredMemoryPolicy`.

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

### 4. `GenerateParameters`: sin conexión a ninguno de los tres sistemas anteriores

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

## Por qué `ChatSession` es la parte más delicada

`ChatSession` ya se documenta a sí mismo como no completamente thread-safe, con
estado mutable público (`instructions`, `additionalContext`, `generateParameters`,
`tools` — ver `DOCS/memory-kv-cache.md`, "Estado actual"). Añadir un
"wired-memory ticket por turno" implica decidir:

- Si el ticket vive por-turno (se adquiere al empezar `respond()`/`streamResponse()`
  y se libera al terminar) o por-sesión (se mantiene mientras la sesión existe,
  cubriendo el KV cache retenido entre turnos).
- Cómo compone con MTP (backlog #7, ver `DOCS/mtp-production-scoping.md`), que
  añade un segundo modelo (drafter) con su propio coste de memoria — el ticket de
  `ChatSession` tendría que saber sumar ese coste también, no solo el del modelo
  principal.
- Qué pasa si `additionalContext`/`generateParameters` cambian a mitad de sesión
  (son `public var` mutables hoy) de forma que el presupuesto estimado ya no es
  válido — ¿se re-evalúa el ticket en cada turno, o solo al construir la sesión?

Ninguna de estas preguntas tiene una respuesta obviamente correcta sin datos de
uso real, y decidirlas mal en el primer intento significa una segunda ronda de
cambios sobre el tipo más usado del paquete.

## Recomendación

1. **Mantener consultiva esta primera fase.** Ya existen la estimación pura y la
   composición explícita con `WiredBudgetPolicy`, sin mutar configuración ni
   adquirir tickets implícitamente.
2. **No implementar el `MemoryBudget` unificado todavía.** Es la pieza de mayor
   superficie de diseño de los cuatro items generation-path de esta rama y
   depende de decisiones de producto (política de prioridad al recortar) que no
   deberían fijarse sin casos de uso reales.
3. **Siguiente orden**: validar estimación frente a mediciones de modelos reales;
   después decidir si `WiredMemoryPolicy` y
   `SpeculativeDecodingMemoryPolicy` se unifican de verdad. Solo al final,
   con esa decisión estable, diseñar el
   ticket por turno de `ChatSession` — que es exactamente donde `DOCS/mtp-production-scoping.md`
   recomienda esperar antes de integrar MTP en `ChatSession`.
4. Esta nota, junto con `DOCS/adaptive-speculative-decoding-scoping.md` y
   `DOCS/mtp-production-scoping.md`, deja los tres items generation-path
   restantes de `DOCS/tech-debt-and-research-backlog.md` documentados con el
   mismo nivel de evidencia que el resto de esta rama, listos para una sesión
   de implementación dedicada con verificación end-to-end sobre modelo real
   (ya probada posible en este entorno vía `IntegrationTesting`, ver el resto
   de esta rama).
