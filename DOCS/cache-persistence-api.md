# API de persistencia de KV cache (prompt caching seguro)

> Cubre el backlog item #13 ("Cache persistence API. Prompt caching seguro para
> apps RAG y asistentes con system prompts largos"). La API descrita aquí ya
> existe en el repo — este documento explica cómo usarla de forma segura, con
> ejemplos verificados contra el código y los tests actuales, y señala los
> huecos reales de la API pública en lugar de inventar soluciones.

## 1. La API base

Definida en `Libraries/MLXLMCommon/KVCache.swift`:

```swift
public func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:]
) throws

public func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String])
```

(`Libraries/MLXLMCommon/KVCache.swift:1603` y `:1648`.)

### Qué serializan

- `url` debe tener extensión `.safetensors`. Cualquier otra extensión lanza
  `LoadSaveError.unknownExtension` desde `save(arrays:metadata:url:)` /
  `loadArraysAndMetadata(url:)` de MLX (`.build/checkouts/mlx-swift/Source/MLX/IO.swift:61-176`)
  — estas dos funciones de MLX son el I/O real por debajo de `savePromptCache`/`loadPromptCache`.
- Por cada `KVCache` en el array de entrada, `savePromptCache` toma:
  - `cache.state`: los `[MLXArray]` (keys/values u otro estado según el tipo de caché),
    aplanados en el safetensors con claves `"i.j"` (caché `i`, array `j` dentro de ese caché).
  - `cache.metaState`: `[String]` con metadatos internos de reconstrucción (offset,
    tamaño máximo, bits de cuantización, etc., variable según el tipo concreto de caché),
    guardado bajo claves `"0.i.j"`.
  - El nombre de clase Swift del caché (`KVCacheSimple`, `RotatingKVCache`,
    `QuantizedKVCache`, `ChunkedKVCache`, `MambaCache`, `ArraysCache`, `CacheList`),
    bajo `"2.i"`, usado para reconstruir el tipo correcto al cargar.
  - El `metadata: [String: String]` que pases tú, bajo `"1.key"`.
- El escritor de MLX materializa los `MLXArray` para poder serializarlos, así que no
  hace falta llamar `.eval()` manualmente antes de `savePromptCache` — ningún test de
  este repo lo hace (`Tests/MLXLMTests/KVCacheTests.swift`).

### El parámetro `metadata`

Es un diccionario `[String: String]` de uso libre para el llamador. La API no le da
ningún significado especial — es el sitio correcto para guardar tu propia información
de versionado: qué texto de system prompt/contexto RAG generó este caché, qué modelo
y cuantización se usó, un timestamp, etc. Ver sección 3 (checklist de seguridad):
esto es crítico porque cargar un caché "stale" contra un prompt *distinto* del que lo
generó produce atención silenciosamente incorrecta — no hay ningún chequeo automático
de esto en la API (verificado: no hay ninguna ocurrencia de `hash`, `fingerprint`,
`checksum`, `modelId` ni `modelType` en `KVCache.swift` ni `ChatSession.swift`).

## 2. Ejemplo real: caché de un system prompt largo / contexto RAG

`ChatSession` (`Libraries/MLXLMCommon/ChatSession.swift`) es el único punto de entrada
público que produce un `[KVCache]` ya "prefillado" y lo puede volcar a disco. El flujo
real, verificado contra `Tests/MLXLMTests/ChatSessionTests.swift:464-479`
(`testSaveAndRestoreCache`), es:

```swift
// --- Paso 1: construir el caché una vez, con el system prompt / contexto RAG largo ---

let longSystemPrompt = "... documento RAG largo o instrucciones extensas ..."

let container = ModelContainer(context: modelContext)
let session = ChatSession(
    container,
    instructions: longSystemPrompt,
    generateParameters: GenerateParameters(maxTokens: 1)  // ver nota más abajo
)

// Fuerza el prefill del system prompt. saveCache(to:) exige que haya habido
// al menos una generación (ver ChatSessionError.noCacheAvailable más abajo),
// así que no hay forma de "solo prefillar sin generar" con la API pública actual.
_ = try await session.respond(to: "ok")

let cacheURL = URL(fileURLWithPath: "/path/to/system-prompt.safetensors")
try await session.saveCache(to: cacheURL)

// --- Paso 2 (sesiones futuras): cargar el caché y arrancar sin re-prefillar ---

let (loadedCache, savedMetadata) = try loadPromptCache(url: cacheURL)

// ver sección 3: valida savedMetadata contra tu fingerprint esperado ANTES
// de construir la sesión con este caché.

let restored = ChatSession(
    container,                 // debe ser el MISMO modelo/cuantización que generó el caché
    instructions: nil,         // nil: el system prompt ya está codificado en el caché
    cache: loadedCache,
    generateParameters: GenerateParameters()
)

let answer = try await restored.respond(to: "pregunta real del usuario")
```

### Gap real #1: no hay "prefill-only" público

`ChatSession.saveCache(to:)` (`ChatSession.swift:899`) solo puede leer el caché desde
el estado interno `.kvcache`, que únicamente existe tras `respond()`/`streamResponse()`.
Si se llama antes, lanza `ChatSessionError.noCacheAvailable`
(`"No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."`,
`ChatSession.swift:911-919`, cubierto por `testSaveCacheThrowsBeforeGeneration`).

Bajando un nivel, `TokenIterator` (`Evaluate.swift:556`) tampoco ayuda: su inicializador
llama `prepare(input:)` internamente, y `prepare` ya "ceba la bomba" generando un primer
token como parte del prefill (`Evaluate.swift:690-710`, comentario "evaluate the
remainder of the prompt -- this primes the pump"). Además, la propiedad `cache` de
`TokenIterator` **no es `public`** (`Evaluate.swift:567`), así que ni siquiera se puede
extraer el `[KVCache]` de un `TokenIterator` manualmente desde fuera del módulo.

**Conclusión honesta**: hoy no existe una API pública para "prefillar sin generar
ningún token". El patrón práctico es forzar `GenerateParameters(maxTokens: 1)` para
minimizar el coste del token de descarte y luego llamar `saveCache(to:)`. Esto es un
gap real de la API, no una limitación inventada — si se necesita evitar por completo
la generación de tokens, haría falta una API nueva (p.ej. un `prefill(input:)` público
en `ChatSession` o exponer `TokenIterator.cache`).

### Gap real #2: `saveCache(to:)` no acepta `metadata`

`ChatSession.saveCache(to:)` llama a `savePromptCache(url:cache:)` **sin** pasar
`metadata`, es decir, siempre con `[:]` (`ChatSession.swift:899-908`):

```swift
public func saveCache(to url: URL) async throws {
    try await cache.read { cache in
        switch cache {
        case .kvcache(let cache, _, _):
            try savePromptCache(url: url, cache: cache)   // sin metadata
        default:
            throw ChatSessionError.noCacheAvailable
        }
    }
}
```

Y el método interno que expondría el `[KVCache]` crudo, `withCache(_:)`, no es `public`
(`ChatSession.swift:878`, comentado explícitamente como "meant for test support").

**Consecuencia práctica**: usando solo la API pública de `ChatSession`, no hay forma de
adjuntar tu propio fingerprint (hash del system prompt, id de modelo, versión del
formato de caché) al mismo archivo `.safetensors`. La solución honesta es un **archivo
sidecar**: guarda un JSON/plist junto al `.safetensors` con esa información, y valida
ese sidecar antes de cargar el caché real. Ver sección 3.

Si tu app puede permitirse ir por debajo de `ChatSession` (construyendo el `[KVCache]`
tú mismo con `makePromptCache(model:parameters:)` y llamando `savePromptCache` a mano),
sí tienes acceso directo al parámetro `metadata` — pero entonces pierdes el prefill
automático que da `ChatSession`/`TokenIterator` y tendrías que reimplementar el
prefill llamando a APIs de más bajo nivel del modelo, que quedan fuera del alcance
de este documento porque no hay un patrón público y probado para ello en este repo.

## 3. Checklist de seguridad

Antes de confiar en un caché cargado con `loadPromptCache(url:)`, verifica:

1. **Fingerprint del contenido** (no hay nada automático — hazlo tú):
   - Al guardar: calcula un hash (p.ej. SHA-256) del texto exacto del system
     prompt / contexto RAG y guárdalo en tu sidecar (o en `metadata` si vas por
     la ruta de bajo nivel de la sección 2).
   - Al cargar: recalcula el hash del prompt que la sesión actual va a usar y
     compara contra el guardado **antes** de construir el `ChatSession` con
     `cache:`. Si no coincide, descarta el caché y re-prefillar desde cero.
   - Esto es crítico: no hacerlo produce salidas incoherentes de forma
     silenciosa, no un error — el modelo simplemente atiende sobre un contexto
     que no es el que el resto de la conversación asume.

2. **Archivo corrupto o truncado**: `loadPromptCache` ya no usa `fatalError`
   (cambio reciente en esta rama) — lanza `KVCacheError` con mensajes específicos,
   verificados en `Tests/MLXLMTests/KVCacheTests.swift:695-745`:
   - `KVCacheSimple`/`KVCache` con un número de arrays distinto de 2:
     `"Corrupt prompt cache: KVCacheSimple state must have exactly 2 arrays (keys, values), found \(n)"`.
   - `QuantizedKVCache` con `metaState` distinto de 4 valores:
     `"Corrupt prompt cache: QuantizedKVCache metaState must have exactly 4 values, found \(n)"`,
     o con un número de arrays que no es 4 ni 6.
   - `ChunkedKVCache` con `metaState` distinto de 2 valores:
     `"Corrupt prompt cache: ChunkedKVCache metaState must have exactly 2 values, found \(n)"`.
   - `RotatingKVCache` con `metaState` inválido, `maxSize == "None"`, o un
     `maxSize` no parseable como `Int`.
   - Estructura de metadata del propio archivo inválida (menos de 3 entradas
     top-level): `"Invalid cache metadata format"`.
   - Mismatch entre el número de cachés, de `cacheInfo` y de `cacheClasses`:
     `"Mismatch in cache counts"`.
   - Nombre de clase desconocido: `"Unknown cache class: \(className)"`.

   Todos estos son `throw`, capturables con un simple `do/catch` alrededor de
   `loadPromptCache(url:)`. Un fichero que no exista o no sea un safetensors
   válido no llega siquiera a producir un `KVCacheError` — falla antes, dentro
   de `loadArraysAndMetadata(url:)` de MLX, con un error genérico de I/O de MLX
   (o `LoadSaveError.unknownExtension` si la extensión no es `.safetensors`).
   Trata cualquier error de esta llamada como "caché no utilizable, re-prefillar".

3. **Caché de un modelo/cuantización distinto al que está cargado ahora**: no
   hay ninguna validación en el código actual. Ni `loadPromptCache` ni los
   inicializadores de `ChatSession(_:cache:...)` comprueban que el caché
   corresponda al modelo dado — el doc comment de esos inicializadores dice
   explícitamente que el llamador es responsable ("a non-empty `[KVCache]`
   previously loaded with `loadPromptCache(url:)`, **matching the given
   model**", `ChatSession.swift:328-329` y `:374-375`). Cargar un caché de un
   modelo A en una sesión de un modelo B probablemente no falle limpio: puede
   producir un crash por mismatch de shapes dentro de la atención, o (peor)
   salida con apariencia válida pero semánticamente basura, dependiendo de si
   las dimensiones (número de capas, KV heads, head dim) coinciden por
   casualidad. **Guarda el id/revisión del modelo y el esquema de
   cuantización en tu metadata/sidecar y verifícalo explícitamente antes de
   pasar el caché a `ChatSession`.**

## 4. Limitaciones conocidas

- `KVCacheSimple.trim` baja el offset lógico pero no garantiza liberar
  inmediatamente el backing storage (`DOCS/memory-kv-cache.md`, riesgo 6) — si
  tu app trimea un caché cargado antes de reutilizarlo, no asumas que la
  memoria se libera de inmediato.
- `RotatingKVCache` puede crecer temporalmente por encima de `maxCacheSize`
  durante un prefill multi-token (`DOCS/memory-kv-cache.md`) — ten esto en
  cuenta al dimensionar el presupuesto de memoria para el prefill inicial de
  un system prompt largo, no solo para el estado ya persistido.
  Además, `RotatingKVCache` con `maxSize == nil` (metaState `"None"`) no se
  puede restaurar en absoluto: `restoreCacheFromMetaState` lanza `KVCacheError`
  para ese caso explícitamente (`KVCache.swift:1714-1718`) — no uses
  `RotatingKVCache` sin `maxSize` si planeas persistir el caché.
- No existe ningún mecanismo de versionado del propio formato de serialización
  (la estructura `"i.j"` / `"0.i.j"` / `"1.key"` / `"2.i"`). Si el formato
  cambia en una versión futura del paquete, un caché guardado con una versión
  antigua podría cargar con datos corridos/mal interpretados en lugar de un
  error claro. Recomendación: incluye la versión del paquete o un número de
  versión de formato propio en tu metadata/sidecar también.
- `ArraysCache` y `MambaCache` se reconstruyen vía `restoreFromMetaState`
  (no `throws`) — a diferencia de los demás tipos, no se verificó en el código
  actual que rechacen entradas corruptas de la misma forma explícita que
  `KVCacheSimple`/`RotatingKVCache`/`QuantizedKVCache`/`ChunkedKVCache`; no
  asumas la misma cobertura de errores para esos dos tipos sin revisar
  `KVCache.swift` directamente si tu app los usa.
