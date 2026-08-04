# API de persistencia de KV cache

Backlog #13: prompt caching seguro para apps RAG, asistentes con system prompts
largos y sesiones que necesitan preparar contexto sin emitir una respuesta.

La implementación mantiene dos niveles deliberadamente separados:

- `savePromptCache` / `loadPromptCache`: formato de bajo nivel, sin opinión
  sobre versión ni significado de la metadata;
- `ChatSession`: prefill sin generación visible, metadata de usuario,
  versionado reservado y validación antes de entregar el cache.

## API de bajo nivel: sin cambios

`Libraries/MLXLMCommon/KVCache.swift` conserva las firmas existentes:

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

El archivo safetensors contiene los arrays de cada cache, su `metaState`, el
nombre de la clase concreta y la metadata del llamador. Esta capa sigue siendo
la vía de interoperabilidad para archivos antiguos, herramientas externas y
consumidores que administran directamente `[KVCache]`.

No se añadió una versión obligatoria a estas funciones. Un archivo creado con
`savePromptCache` continúa pudiendo usar cualquier metadata —o ninguna— y
`loadPromptCache` devuelve esa información sin interpretarla.

## Prefill público sin tokens visibles

`ChatSession` expone dos formas aditivas:

```swift
try await session.prefill(
    "Contexto RAG largo",
    role: .system
)

try await session.prefill(messages: [
    .system("Eres un asistente especializado."),
    .user(documento),
])
```

`prefill` usa el mismo processor, media processing, tools y
`additionalContext` que una generación normal. Ejecuta `LanguageModel.prepare`,
evalúa únicamente los tokens restantes que el modelo no haya consumido y
materializa el KV cache. No crea un `TokenIterator`, no muestrea una respuesta y
no abre un stream, por lo que no hay tokens de salida que descartar.

El `LMOutput.State` devuelto por el modelo se guarda junto al cache en la sesión
y se entrega al siguiente turno. Esto es necesario para modelos que mantienen
estado posicional por llamada, como los que usan deltas M-RoPE.

La operación es transaccional respecto a la sesión. Cuando ya existe un cache,
`prefill` trabaja sobre copias; solo sustituye el estado comprometido después de
completar processor, modelo y evaluación. Si hay error o cancelación, el cache
anterior sigue disponible. En sesiones con speculative decoding, el prefill se
replica en el draft model cuando su política de memoria lo admite, para mantener
ambos caches alineados.

## Guardado con metadata y versión

La firma histórica permanece:

```swift
try await session.saveCache(to: cacheURL)
```

La sobrecarga nueva añade metadata de usuario:

```swift
try await session.saveCache(
    to: cacheURL,
    metadata: [
        "model": "org/model@revision",
        "quantization": "4-bit",
        "promptFingerprint": sha256,
    ]
)
```

Ambas firmas escriben automáticamente:

```swift
ChatSession.cacheFormatVersion == 1
ChatSession.cacheFormatVersionMetadataKey
    == "mlx-swift-lm.chat-session-cache.format-version"
```

La clave es reservada. Pasarla dentro de la metadata del llamador produce
`ChatSessionCacheError.reservedCacheMetadataKey` en lugar de permitir que una app
sobrescriba la versión declarada.

El versionado pertenece al contrato de alto nivel de `ChatSession`; no cambia la
estructura interna de `savePromptCache`. Por eso el mismo archivo continúa
siendo legible con `loadPromptCache`, que devolverá tanto las claves de usuario
como la clave reservada.

## Carga y validación

Para caches creados mediante `ChatSession`, la ruta recomendada es:

```swift
let snapshot = try ChatSession.loadCache(
    from: cacheURL,
    validating: [
        "model": "org/model@revision",
        "promptFingerprint": expectedSHA256,
    ]
)

let restored = ChatSession(
    modelContainer,
    instructions: nil,
    cache: snapshot.cache
)
```

`ChatSessionCacheSnapshot` expone:

- `cache`: los `[KVCache]` restaurados;
- `metadata`: solo la metadata del usuario, sin la clave reservada;
- `formatVersion`: la versión ya parseada y validada.

La carga de alto nivel falla con `ChatSessionCacheError` en estos casos:

- falta la clave de versión: `missingCacheFormatVersion`;
- la versión no es un entero: `invalidCacheFormatVersion`;
- la versión no coincide con la soportada: `unsupportedCacheFormatVersion`;
- una clave esperada falta o tiene otro valor: `cacheMetadataMismatch`;
- el llamador intenta validar la propia clave reservada:
  `reservedCacheMetadataKey`.

Los caches anteriores a esta API no tienen la clave reservada. No se interpretan
de forma ambigua: `ChatSession.loadCache` los rechaza como no versionados, pero
`loadPromptCache` sigue cargándolos exactamente como antes. La app puede entonces
aplicar su política de migración o descartarlos y repetir el prefill.

## Flujo recomendado para RAG o system prompts largos

```swift
let session = ChatSession(modelContainer, instructions: longSystemPrompt)
session.wiredMemoryTicket = turnTicket

try await session.prefill("Contexto compartido", role: .system)
try await session.saveCache(
    to: cacheURL,
    metadata: [
        "model": modelRevision,
        "promptFingerprint": fingerprint,
    ]
)

let snapshot = try ChatSession.loadCache(
    from: cacheURL,
    validating: [
        "model": modelRevision,
        "promptFingerprint": fingerprint,
    ]
)

let restored = ChatSession(
    modelContainer,
    instructions: nil,
    cache: snapshot.cache
)
restored.wiredMemoryTicket = nextTurnTicket
```

No se deben repetir en `instructions` los tokens ya incluidos en el cache
restaurado. Hacerlo duplica el contexto lógico y puede producir resultados
incoherentes.

El `wiredMemoryTicket` es mutable porque `ChatSession` ya es single-consumer y no
thread-safe. Se captura al comenzar cada turno y se propaga a `generateTask` tanto
en generación normal como speculative. `prefill` también usa la instantánea del
ticket alrededor de su trabajo y la libera de forma segura ante cancelación.

## Checklist de seguridad

Antes de reutilizar un cache persistido:

1. Guardar y validar un fingerprint del contenido exacto que originó el cache.
2. Guardar y validar id, revisión y cuantización del modelo.
3. Tratar cualquier error de I/O, safetensors, `KVCacheError` o validación como
   cache no utilizable y repetir el prefill.
4. Serializar los guardados: el escritor de safetensors de MLX no debe usarse
   concurrentemente sobre el mismo estado.
5. No guardar mientras otra operación utiliza el mismo `ChatSession`; `saveCache`
   espera acceso exclusivo al contenedor de cache.

## Límites deliberados

- Para preservar el cache comprometido ante error o cancelación, un `prefill`
  incremental copia temporalmente los caches principal y draft. En contextos
  muy largos esto puede aproximarse a duplicar su memoria hasta el commit.
- `LMOutput.State` se conserva entre `prefill` y los turnos posteriores de la
  misma sesión, pero no se serializa. Es un diccionario tipado que puede contener
  valores arbitrarios y no tiene hoy un contrato de persistencia estable. Una
  sesión creada únicamente con `[KVCache]` arranca con state `nil`.
- La metadata permite a la app detectar modelo, revisión, prompt o cuantización
  incompatibles; el framework no puede inferir de forma fiable esos valores a
  partir de los tensores del cache.
- `RotatingKVCache` sin `maxSize` no es restaurable. Los errores estructurales de
  cache siguen correspondiendo a `KVCacheError` en la capa de bajo nivel.
- El número de versión `1` reserva el contrato actual. Cualquier evolución
  incompatible debe incrementar `cacheFormatVersion` y añadir una migración o un
  rechazo explícito; nunca reinterpretar silenciosamente un archivo anterior.

## Cobertura

`Tests/MLXLMTests/ChatSessionTests.swift` cubre:

- prefill sin tokens generados y offset exacto del cache;
- continuidad de `LMOutput.State` al siguiente turno;
- cancelación sin corromper el cache comprometido;
- guardado, carga y restauración tras prefill;
- metadata, clave reservada, versión ausente y versión futura;
- interoperabilidad mediante `loadPromptCache`;
- propagación y reemplazo por turno de wired-memory tickets, incluida la ruta
  speculative.
