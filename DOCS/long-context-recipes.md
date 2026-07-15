# Recetas de long-context

Recetas concretas y copiables para gestionar contexto largo con `Libraries/MLXLMCommon`. Complementa `DOCS/memory-kv-cache.md` (investigación/riesgos) y `DOCS/implementation-playbook.md` (guía breve de perfiles); aquí el foco es "cómo se escribe esto en Swift real, con qué API existente hoy".

Todos los símbolos citados existen en `Libraries/MLXLMCommon/KVCache.swift`,
`KVCacheMemory.swift`, `WiredMemoryPolicies.swift`, `Evaluate.swift`,
`ChatSession.swift` y `LanguageModel.swift` en esta rama.

## 1. Trim: regenerar o editar sin re-hacer prefill completo

### Cuándo usar esto

`KVCache` expone `isTrimmable` y `trim(_ n: Int) -> Int` (`Libraries/MLXLMCommon/KVCache.swift:65-70`). A nivel de sesión, dos funciones libres operan sobre `[KVCache]`:

```swift
public func canTrimPromptCache(_ cache: [KVCache]) -> Bool
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int
```

(`Libraries/MLXLMCommon/KVCache.swift:1903` y `:1912`). Uso típico: UX de "regenerar última respuesta" (descartar los tokens generados y volver a samplear) o "editar un mensaje anterior" (descartar todo lo posterior al punto de edición). En ambos casos el objetivo es evitar re-tokenizar y re-prefillear la conversación completa.

`ChatSession` no expone hoy un método público de trim: gestiona su caché internamente (`enum Cache` en `ChatSession.swift:150`) y solo permite inspección vía `withCache` (marcado `func withCache` como soporte de test, `ChatSession.swift:878`) y guardado vía `saveCache(to:)`. Para aplicar trim con control fino, hay que trabajar directamente con `[KVCache]` (por ejemplo, un caché creado con `makePromptCache(model:parameters:)` y pasado luego a `TokenIterator`), no vía `ChatSession`.

### Receta

```swift
import MLXLMCommon

// cache: [KVCache] obtenido de makePromptCache(model:parameters:) o de
// una generación previa con TokenIterator.

func regenerateLastResponse(cache: [KVCache], assistantTokenCount: Int) {
    guard canTrimPromptCache(cache) else {
        // Alguna caché en el array no es trimmable (p.ej. RotatingKVCache
        // ya rotada, ver isTrimmable más abajo); hay que reconstruir el caché.
        return
    }
    let trimmed = trimPromptCache(cache, numTokens: assistantTokenCount)
    precondition(trimmed == assistantTokenCount, "trim parcial: recalcular offset")
}
```

Para "editar un mensaje anterior", el número de tokens a recortar es la suma de tokens de todo lo que sigue al punto de edición (respuesta(s) del asistente + turnos posteriores), que hay que llevar contabilizado por turno (el repo no ofrece un mapeo automático mensaje-a-rango-de-tokens).

### Gotchas

- `trimPromptCache` hace `cache.dropFirst().forEach { $0.trim(numTokens) }` y luego `cache.first?.trim(numTokens)` (`KVCache.swift:1913-1915`): todas las cachés del array se recortan por el mismo `numTokens`, asumiendo que todas las capas tienen el mismo offset. No mezclar cachés heterogéneas con offsets distintos.
- `RotatingKVCache.isTrimmable` es `offset < maxCacheSize` (`KVCache.swift:723-725`): una vez que la caché rotatoria alcanzó su tamaño máximo y empezó a rotar, deja de ser trimmable. `canTrimPromptCache` devolverá `false` para el array completo en ese punto — no hay trim parcial disponible, hay que recrear la caché.
- `KVCacheSimple.trim` solo baja `offset` (`KVCache.swift:454-459`): **no libera el backing storage** (`self.keys`/`self.values` siguen teniendo su tamaño `step`-alineado). Esto ya está documentado como riesgo en `DOCS/memory-kv-cache.md` ("Retención de buffers tras trim"). Si el objetivo es liberar memoria (no solo reescribir posiciones), hay que reconstruir el caché desde cero, no solo llamar `trim`.
- `trim(_:)` trata ahora valores no positivos como no-op. En caché rotatoria
  devuelve `0` una vez alcanzado el window, evitando corromper `idx`; estos
  contratos están cubiertos en `LongContextKVCacheTests.swift`.

## 2. Caché rotatoria: dimensionar `maxKVSize` para un presupuesto de dispositivo

### Cuándo usar esto

`GenerateParameters.maxKVSize` (`Evaluate.swift:65`) selecciona automáticamente `RotatingKVCache` en lugar de `KVCacheSimple`: `LanguageModel.newCache(parameters:)` en `LanguageModel.swift:293-297` hace `if let maxKVSize = parameters?.maxKVSize { RotatingKVCache(maxSize: maxKVSize, keep: 4) }`. Usarlo cuando la conversación puede superar el presupuesto de memoria del dispositivo y se acepta perder contexto antiguo a cambio de un techo de memoria fijo.

### Receta

```swift
import MLXLMCommon

let estimatedKVBytes = try estimateKVCacheBytes(
    numLayers: 28,
    kvHeads: 8,
    headDim: 128,
    maxTokens: 4096,
    kvBits: nil
)

let memoryPolicy = WiredBudgetPolicy(
    baseBytes: weightsBytes + workspaceBytes,
    kvCacheBytes: estimatedKVBytes,
    cap: deviceBudgetBytes
)

let parameters = GenerateParameters(
    maxTokens: 512,
    maxKVSize: 4096,   // techo duro de tokens en caché por capa
    temperature: 0.6
)

let session = ChatSession(modelContainer, generateParameters: parameters)
// A partir del turno en que offset > 4096, RotatingKVCache empieza a rotar.
```

La estimación es lógica y pura: no incluye spare capacity de los buffers,
alineación ni el pico temporal de prefill. La policy tampoco se conecta
automáticamente a `ChatSession`; sirve para construir un ticket explícito en las
rutas de generación que ya aceptan `WiredMemoryPolicy`. Para calibración real,
comparar con `WiredMemoryUtils.tune(...)` en el modelo y hardware objetivo.

### Qué significa "la calidad de conversación se degrada al empezar a truncar"

`RotatingKVCache` mantiene `keep` tokens fijos al principio (el `newCache` de arriba usa `keep: 4`, típicamente el system prompt/BOS) y luego opera como un buffer circular sobre el resto: `updateInPlace`/`updateConcat` recortan con `trim(trimSize:_:append:)` (`KVCache.swift:545-559`), que concatena `array[..<keep]` con `array[(trimSize+keep)...]`. En la práctica esto significa:

- Los primeros `keep` tokens (por defecto 4 — normalmente insuficiente para preservar un system prompt real) sobreviven indefinidamente.
- Todo lo demás se descarta en orden FIFO según se supera `maxCacheSize`: los tokens **más antiguos del rango no protegido** desaparecen primero, no los menos relevantes semánticamente.
- `isTrimmable` deja de ser `true` una vez que `offset >= maxCacheSize` (rotación activa), así que el patrón de la Receta 1 (trim manual) ya no aplica a esta caché.
- La máscara de atención (`makeMask`, `KVCache.swift:736-773`) usa un offset capado (`min(maxCacheSize - 1, offset)`) y, para el caso de un solo token con rotación activa, construye una máscara con `roll` para tener en cuenta la rotación del índice circular — es código más delicado que el causal simple, y cualquier regresión ahí puede producir atención sobre posiciones incorrectas sin que el modelo "crashee" (silenciosamente da peor calidad).

En términos de producto: si el system prompt es largo (más de unos pocos tokens), `keep: 4` no lo protege completo. Si se necesita preservar un system prompt largo bajo `maxKVSize`, no hay hoy un parámetro público en `GenerateParameters` para ajustar `keep` — `newCache` lo fija en 4 (`LanguageModel.swift:296`). Ajustarlo requeriría construir el `[KVCache]` manualmente con `RotatingKVCache(maxSize:keep:step:)` en lugar de pasar por `GenerateParameters.maxKVSize`.

### Gotchas

- El comentario de `maxKVSize` en `Evaluate.swift:63-64` dice literalmente: "Old entries (except the first 4 tokens) will be overwritten" — el `4` está hardcodeado en `newCache`, no es configurable vía `GenerateParameters`.
- Durante prefill multi-token, `RotatingKVCache` puede crecer temporalmente por encima de `maxCacheSize` (comentario en `updateConcat`, `KVCache.swift:587-589`: "every token gets at least maxCacheSize context") — no asumir un techo de memoria estrictamente constante durante el prefill de un turno largo.
- No confundir con `ChunkedKVCache` (`KVCache.swift:1122`), que trocea un contexto grande para *procesarlo* (sliding window de procesamiento) pero no impone un límite permanente igual que `RotatingKVCache`.

## 3. Caché cuantizada: `kvBits`, `kvGroupSize`, `quantizedKVStart`, `kvScheme`

### Cuándo usar esto

Cuando el peso en memoria del KV cache (no del modelo) es el cuello de botella — contextos largos con modelos de cabezas KV/head-dim grandes. `GenerateParameters` expone:

```swift
public var kvBits: Int?            // nil = sin cuantización
public var kvGroupSize: Int        // default 64
public var quantizedKVStart: Int   // default 0
public var kvScheme: String?       // "affine4" | "affine8" | nil
```

(`Evaluate.swift:68-79`). `kvScheme`, si resuelve a un esquema conocido, **sobrescribe** `kvBits`/`kvGroupSize` (`resolveAffineScheme`, `KVCache.swift:2021-2027`: `"affine4"` -> `(4, 64)`, `"affine8"` -> `(8, 64)`).

### Receta

```swift
let parameters = GenerateParameters(
    maxTokens: 512,
    kvBits: 4,
    kvGroupSize: 64,
    quantizedKVStart: 512,   // no cuantizar los primeros 512 tokens de contexto
    temperature: 0.6
)
```

La conversión ocurre en caliente, por token, dentro de `TokenIterator.step(previous:)`:

```swift
maybeQuantizeKVCache(
    cache: &cache,
    kvBits: kvBits,
    kvGroupSize: kvGroupSize,
    quantizedKVStart: quantizedKVStart,
    kvScheme: kvScheme
)
```

(`Evaluate.swift:732-739`, también invocado en `MTPSpeculativeTokenIteratorTests`/`WiredMemoryUtils.swift`). `maybeQuantizeKVCache` (`KVCache.swift:2042`) solo convierte entradas donde `cache is KVCacheSimple && !(cache is QuantizedKVCache) && cache.offset > quantizedKVStart` (`isQuantizable`, `KVCache.swift:2065-2072`) — es decir, `quantizedKVStart` es un umbral de *offset actual de esa caché*, no un contador global de turno.

### `RotatingKVCache` + cuantización: gap real confirmado en esta rama

`RotatingKVCache.quantized(groupSize:bits:)` **lanza** incondicionalmente:

```swift
public func quantized(groupSize: Int = 64, bits: Int = 4) throws -> QuantizedKVCache {
    // Future implementation would need to: ...
    throw KVCacheError(
        message: "RotatingKVCache quantization not yet implemented - temporal ordering makes this complex")
}
```

Esta es la ruta recuperable para código nuevo, confirmada por el test `testRotatingKVCacheQuantizedThrowsInsteadOfCrashing`. La firma histórica `toQuantized(...)` sigue disponible y deprecada para conservar compatibilidad fuente con consumidores 3.x; no debe usarse cuando la configuración pueda ser incompatible. Pero eso no es lo que pasa en la práctica al combinar `maxKVSize` + `kvBits`: `maybeQuantizeKVCache.isQuantizable` sólo acepta `KVCacheSimple`, y cuando `maxKVSize` está seteado, `newCache` construye `RotatingKVCache`. Esas cachés se ignoran por completo — **nunca se intenta llamar `quantized` y por tanto nunca se lanza el error**. El resultado observable es: `GenerateParameters(maxKVSize: N, kvBits: 4)` no falla, no lanza, simplemente no cuantiza nada. Este es el gap que hay que documentar de cara a producto: es un silencio, no un error.

`testMaxKVSizeAndKVBitsKeepRotatingCacheUnquantized` fija explícitamente este
contrato para detectar cualquier cambio futuro de comportamiento.

```swift
// Esto NO produce una caché rotatoria cuantizada. Genera sin error,
// pero maxKVSize domina y kvBits se ignora silenciosamente.
let parameters = GenerateParameters(maxKVSize: 4096, kvBits: 4)
```

### Gotchas

- `quantizedKVStart` se compara contra `cache.offset` de cada caché individual (`KVCache.swift:2071`), así que en modelos con cachés heterogéneas (`CacheList`, híbridos Mamba/atención) el umbral se aplica por sub-caché, recorriendo también los hijos de `CacheList` (`mapChildren`, `KVCache.swift:2100-2108`).
- `groupSize` efectivo puede diferir del solicitado: `resolvedKVQuantizationGroupSize` (`KVCache.swift:811-829`) elige el más cercano entre `{32, 64, 128}` que divida ambos head-dims; si ninguno divide, la caché **no se cuantiza** (se devuelve `cache` sin cambios en `quantize(_:)`, `KVCache.swift:2093-2097`) — de nuevo, sin error visible.
- `kvScheme` con un valor desconocido (ni `"affine4"` ni `"affine8"`) hace que `resolveAffineScheme` devuelva `nil` y `maybeQuantizeKVCache` retorne temprano sin cuantizar (comentario explícito: "Unrecognized schemes are left to custom cache implementations", `KVCache.swift:2041`) — no lanza error por typo en el nombre del esquema.

## 4. Prompt cache persistente: cachear un system prompt largo una vez, reusarlo entre sesiones

### Cuándo usar esto

Cuando el mismo prefijo largo (system prompt extenso, documento para RAG) se repite entre sesiones o arranques de la app y no se quiere pagar el prefill cada vez.

### Receta

Basada en el patrón real usado en `Tests/MLXLMTests/ChatSessionTests.swift:464-479` (`testSaveAndRestoreCache`):

```swift
import MLXLMCommon

// 1) Construir el caché una vez, con el system prompt largo como primer turno.
let container: ModelContainer = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")
let warmup = ChatSession(container, instructions: longSystemPrompt, generateParameters: params)
_ = try await warmup.respond(to: "ack")  // fuerza el prefill del system prompt

// 2) Guardar el caché resultante a disco.
let url = cachesDirectory.appendingPathComponent("system-prompt.safetensors")
try await warmup.saveCache(to: url)

// 3) En una sesión futura (mismo proceso o uno nuevo), cargar el caché
//    y arrancar una sesión SIN volver a pasar `instructions` (ya está
//    horneado en el caché; volver a pasarlo lo re-tokenizaría sin
//    coincidir con el estado del KV, produciendo salida incoherente).
let (loadedCache, metadata) = try loadPromptCache(url: url)
let session = ChatSession(container, cache: loadedCache, generateParameters: params)
let answer = try await session.respond(to: "primera pregunta real")
```

`ChatSession` documenta exactamente esta advertencia en su inicializador `init(_:instructions:cache:...)` (`ChatSession.swift:319-322`): *"If the cache was built from a session that already included system instructions, do not pass the same `instructions` here"*.

Para guardar metadatos propios junto al caché (versión del prompt, hash del documento fuente, etc.), `savePromptCache(url:cache:metadata:)` acepta `[String: String]` libre:

```swift
try savePromptCache(url: url, cache: kvCacheArray, metadata: ["promptVersion": "3", "docHash": hash])
let (cache, metadata) = try loadPromptCache(url: url)
let version = metadata["promptVersion"]
```

(patrón de firma verificado en `KVCache.swift:1603` y usado en los tests con `metadata: [:]`, p.ej. `KVCacheTests.swift:77-78`).

### Gotchas

- `ChatSession.saveCache(to:)` lanza `ChatSessionError.noCacheAvailable` si no hubo ninguna generación todavía (`ChatSession.swift:897-908`, cubierto por `testSaveCacheThrowsBeforeGeneration` en `ChatSessionTests.swift:451-462`) — no se puede guardar un caché "vacío a propósito" antes de generar al menos una vez.
- El formato es `.safetensors` (`savePromptCache`/`loadPromptCache` usan `save(arrays:metadata:url:)`/`loadArraysAndMetadata`, `KVCache.swift:1603-1685`): son archivos de arrays MLX, no JSON — tratar el path como binario, no como texto editable.
- El caché cargado debe corresponder al mismo modelo/arquitectura (mismas dimensiones de capas y cabezas KV) que la sesión que lo consume; `loadPromptCache` no valida compatibilidad contra un `ModelContainer` dado, solo reconstruye las clases de caché serializadas (`restoreCacheFromMetaState`, `KVCache.swift:1691-1780`) — un mismatch de modelo produce errores de shape más adelante en `TokenIterator`, no un error temprano y claro en `loadPromptCache`.
- Los initializers `ChatSession(_:cache:...)` toman `cache: consuming [KVCache]` — no reutilizar el mismo array `[KVCache]` en dos sesiones sin copiarlo primero (usar `.map { $0.copy() }` como hace `testInitWithKVCache`, `ChatSessionTests.swift:436-448`, si se necesita el mismo estado de partida en más de una sesión viva).

## 5. Resumen de conversación: NO implementado en este repo — patrón para construirlo

### Estado real (verificado)

Grep de `summar` (case-insensitive) sobre `Libraries/` no encuentra ninguna función, protocolo o tipo relacionado con resumen de conversación. No existe un `ConversationSummarizer`, ni un modo de `ChatSession` que compacte historial automáticamente. `DOCS/implementation-playbook.md` menciona "Resumir historia antes de forzar contexto largo" como recomendación de guía (`implementation-playbook.md:38`), no como API existente. Esta sección es, por tanto, honestamente un patrón de aplicación, no una feature del SDK.

### Patrón: recortar turnos antiguos + inyectar un resumen como mensaje de sistema

La pieza que sí existe y es reutilizable es `Chat.Message` con rol `.system`/`.assistant`/`.user` y el hecho de que `ChatSession` acepta `respond(to: [Chat.Message])` para inyectar mensajes estructurados preservando el caché (`ChatSession.swift:467-475`, pensado originalmente para resultados de tools, pero el mecanismo de "agregar mensajes sin re-prefillear todo el historial previo" es el mismo que se necesita aquí).

```swift
import MLXLMCommon

/// Responsabilidad de la app, no de MLXLMCommon: mantener el texto de
/// resumen actualizado y decidir cuándo dispararlo.
actor ConversationMemory {
    private var summary: String?
    private var turnsSinceLastSummary: [Chat.Message] = []

    func recordTurn(_ messages: [Chat.Message]) {
        turnsSinceLastSummary.append(contentsOf: messages)
    }

    /// Genera (o actualiza) un resumen usando el MISMO modelo, en una
    /// generación aparte y desechable -- no comparte el KVCache de la
    /// sesión principal.
    func maybeSummarize(
        threshold: Int,
        summarize: (_ priorSummary: String?, _ turns: [Chat.Message]) async throws -> String
    ) async throws {
        guard turnsSinceLastSummary.count >= threshold else { return }
        summary = try await summarize(summary, turnsSinceLastSummary)
        turnsSinceLastSummary.removeAll()
    }

    var currentSummary: String? { summary }
}

// Al reconstruir la sesión (o al empezar una nueva) tras resumir:
let memory = ConversationMemory()
// ... tras varios turnos, memory.maybeSummarize(...) produjo un resumen ...
if let summaryText = await memory.currentSummary {
    let session = ChatSession(
        container,
        instructions: "Contexto previo resumido:\n\(summaryText)",
        generateParameters: params
    )
}
```

La generación del resumen en sí (`summarize` closure arriba) es una llamada de generación normal — por ejemplo otro `ChatSession` de corta vida, o un prompt de un solo turno vía `TokenIterator`/`generate` — que pide al modelo un resumen del historial descartado, y cuyo resultado se guarda como texto plano (no como KVCache).

### Gotchas de este patrón (no del repo, del approach)

- Resumir consume tokens y latencia de generación adicional: no ejecutarlo por turno, sino por umbral (cuenta de mensajes, o estimación de tokens vía el tokenizer del `ModelContainer`).
- Al reemplazar `instructions` con el resumen, el caché de la sesión anterior queda inválido para esos turnos (el resumen es texto nuevo, hay que re-prefillearlo) — esto es un trade-off inherente entre "ahorrar contexto" y "ahorrar prefill", no algo que MLXLMCommon resuelva por vos.
- Nada en este repo verifica que el resumen preserve hechos citables exactos (nombres, números, cotas). Si la app depende de precisión factual sobre turnos antiguos, el resumen por LLM es lossy y hay que decidir explícitamente qué se puede permitir perder.

## Cobertura entregada y gaps que requieren medición real

Estado real de la cobertura de tests en `Tests/MLXLMTests/` (verificado por grep):

**Lo que sí está cubierto hoy:**
- Estimación KV BF16 y affine, metadata de cuantización, group-size efectivo,
  configuraciones inválidas y overflow (`KVCacheMemoryTests.swift`).
- Trim y reescritura de `KVCacheSimple`, alineación de offsets entre capas y
  rechazo de trim tras alcanzar el window rotatorio (`LongContextKVCacheTests.swift`).
- Máscara de `RotatingKVCache` tras wraparound para decode de un token y window
  menor que `maxSize` (`testRotatingKVCacheMaskTracksWrappedSingleTokenWindow`).
- Round-trip persistido de una caché rotatoria ya envuelta, seguido de otra
  actualización que verifica que `idx` se restauró correctamente.
- Combinación `maxKVSize + kvBits`: no lanza y mantiene las cachés rotatorias sin
  cuantizar.
- Serialización/round-trip de todos los tipos de caché (`testCacheSerialization`, parametrizado sobre `KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache`, `ChunkedKVCache`, `ArraysCache`, `MambaCache` — `KVCacheTests.swift:56-86`).
- Que `RotatingKVCache.quantized` lanza en vez de crashear (`testRotatingKVCacheQuantizedThrowsInsteadOfCrashing`).
- Que `loadPromptCache` lanza `KVCacheError` (no crashea) ante estado/metaState corrupto para `KVCacheSimple`, `QuantizedKVCache`, `ChunkedKVCache` (`KVCacheTests.swift:717-745`).
- Máscara basada en `leftPadding`/`lengths` para `ArraysCache.makeMask` (`testArraysCacheMaskUsesLeftPaddingAfterStateUpdate`, `testArraysCacheMaskUsesLengthsWhenLeftPaddingIsAbsent`, `KVCacheTests.swift:191-226`).
- Máscara causal compartida (`testAttentionMaskUsesSharedCausalCachePath`, `KVCacheTests.swift:329`) y máscara SSM (`testSSMMaskUsesSharedMambaMetadataPath`, `KVCacheTests.swift:363`).
- `maybeQuantizeKVCache` end-to-end sobre un modelo real pequeño (Gemma4Text, `Gemma4TextTests.swift:18,38`) y sobre híbridos (`FalconH1Tests.swift:145`).
- Guardar/restaurar un `ChatSession` completo vía `saveCache`/`init(cache:)` (`ChatSessionTests.swift:430-479`).

**Gaps que siguen requiriendo modelo o dispositivo real:**
- **Calidad bajo `maxKVSize` (truncación real)**: no hay ningún test que mida degradación de calidad de generación (ni siquiera con heurística simple) cuando la conversación supera `maxKVSize` y la caché rotatoria empieza a descartar tokens.
- **Calidad bajo `kvBits`/`quantizedKVStart`**: los tests de `maybeQuantizeKVCache` verifican que la conversión ocurre y que el modelo sigue generando sin crashear, pero no comparan salida cuantizada vs. no cuantizada para detectar regresión de calidad.
- **Estimación vs. residencia real**: falta contrastar los bytes lógicos con
  `WiredMemoryUtils.tune`, `Memory.activeMemory` y RSS en perfiles 8K/32K/128K.
