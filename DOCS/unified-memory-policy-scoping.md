# Política pública de memoria unificada — scoping note

Backlog: `DOCS/tech-debt-and-research-backlog.md` #3 — "Unificar
`GenerateParameters`, wired memory, KV estimate y speculative memory policy.
Producir recomendaciones explicables, no solo errores OOM. Incluir
`ChatSession`, que hoy no expone wired-memory ticket por turno."

Nota de alcance, no implementación. De los tres backlog items generation-path
que quedan tras la pasada de release (#3, #6 y #7), este es el que tiene la
superficie de diseño más amplia — no es una decisión técnica puntual sino una
decisión de arquitectura pública que otros dos items (#6 y #7, ver sus propios
scoping notes) ya dependen de que se resuelva primero. Se documenta el estado
real y el hueco, no se implementa.

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

### 3. Estimación de KV cache: no existe como función pública reutilizable

No hay una función `estimateKVCacheBytes(...)` en `KVCache.swift` ni en ningún
otro fichero — verificado por grep de `estimate` sobre `Libraries/MLXLMCommon/`.
Quien quiera saber cuántos bytes va a costar el KV cache de una conversación
tiene que calcularlo a mano a partir de `numLayers × kvHeads × headDim × maxTokens
× bytesPerElement × 2 (K+V)`, sin ninguna función del paquete que lo haga por
ellos, y sin que ese cálculo entre en `WiredSumPolicy`/`WiredBudgetPolicy` de
forma automática.

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
- Una futura estimación de KV cache respondería "¿qué `maxKVSize`/`kvBits` puedo
  permitirme dado el presupuesto restante?", como una recomendación de
  configuración, no un límite de sistema operativo ni una decisión de carga.

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

1. **No implementar el `MemoryBudget` unificado todavía.** Es la pieza de mayor
   superficie de diseño de los cuatro items generation-path de esta rama y
   depende de decisiones de producto (política de prioridad al recortar) que no
   deberían fijarse sin casos de uso reales.
2. **Orden sugerido si se retoma**: primero una función pura y sin estado
   `estimateKVCacheBytes(numLayers:kvHeads:headDim:maxTokens:kvBits:) -> Int`
   (bajo riesgo, testeable con valores conocidos, sin tocar `ChatSession` ni
   políticas existentes) — cierra el hueco #3 del sistema (2) de esta nota.
   Después, decidir si `WiredMemoryPolicy` y `SpeculativeDecodingMemoryPolicy`
   se unifican de verdad o simplemente se hacen componibles (un
   `WiredBudgetPolicy.baseBytes` que incluya la estimación de KV, sin fusionar
   los tipos). Solo al final, con los dos anteriores estables, diseñar el
   ticket por turno de `ChatSession` — que es exactamente donde `DOCS/mtp-production-scoping.md`
   recomienda esperar antes de integrar MTP en `ChatSession`.
3. Esta nota, junto con `DOCS/adaptive-speculative-decoding-scoping.md` y
   `DOCS/mtp-production-scoping.md`, deja los tres items generation-path
   restantes de `DOCS/tech-debt-and-research-backlog.md` documentados con el
   mismo nivel de evidencia que el resto de esta rama, listos para una sesión
   de implementación dedicada con verificación end-to-end sobre modelo real
   (ya probada posible en este entorno vía `IntegrationTesting`, ver el resto
   de esta rama).
