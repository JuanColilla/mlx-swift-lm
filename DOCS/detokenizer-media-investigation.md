# Detokenizer y media materialization: investigación

Investigación e implementación del backlog item #15
(`DOCS/tech-debt-and-research-backlog.md`, prioridad baja-pero-estrategica):
"Medir coste O(n^2) de streaming detokenization y picos CPU de imagen/audio".

La primera versión de este documento aisló los hotspots. La implementación posterior añade un
microbenchmark reproducible, acota el buffer para los tokenizers de swift-transformers y reduce
las materializaciones CPU transitorias de vídeo/audio sin romper las APIs públicas 3.x.

Ground truth previo: `DOCS/memory-kv-cache.md`, sección "Riesgos principales", items 7 y 8.

## Detokenizer complexity: what the code actually does

Implementación histórica: `Libraries/MLXLMCommon/Tokenizer.swift`
(`NaiveStreamingDetokenizer`).

```swift
public mutating func append(token: Int) {
    segmentTokens.append(token)
}

mutating func startNewSegment() {
    let lastToken = segmentTokens.last
    segmentTokens.removeAll()
    if let lastToken {
        segmentTokens.append(lastToken)
        segment = tokenizer.decode(tokenIds: segmentTokens)
    } else {
        segment = ""
    }
}

public mutating func next() -> String? {
    let newSegment = tokenizer.decode(tokenIds: segmentTokens)
    let new = newSegment.suffix(newSegment.count - segment.count)
    ...
    if new.hasSuffix("\n") {
        startNewSegment()
    } else {
        self.segment = newSegment
    }
    return String(new)
}
```

Trazando el control de flujo:

- `append(token:)` (línea 81-83) simplemente añade el token al array `segmentTokens`. No decodifica nada.
- `next()` (línea 96-113) es lo que produce el string delta. En la línea 97 llama
  `tokenizer.decode(tokenIds: segmentTokens)` sobre el **array completo acumulado desde el
  último reset**, no sobre el token nuevo únicamente. Esto es una re-decodificación completa
  del buffer en cada token, necesaria porque un solo token puede producir un fragmento UTF-8
  incompleto (de ahí la comprobación de `\u{fffd}` en la línea 102) y hay que recomputar el
  string completo para saber cuál es el sufijo nuevo válido.
- Si el nuevo fragmento termina en `\n` (línea 106), se llama `startNewSegment()` (línea 85-94),
  que vacía `segmentTokens` y lo reinicia solo con el último token (línea 87-89). Esto resetea
  el tamaño del buffer a 1 elemento.
- Si no hay `\n`, el buffer sigue creciendo indefinidamente (línea 109: `self.segment =
  newSegment`, pero `segmentTokens` nunca se trunca fuera de `startNewSegment()`).

### Coste de `decode(tokenIds:)` en sí

`decode(tokenIds:skipSpecialTokens:)` en el bridge de este repo (`Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:95-97`)
delega directamente en `Tokenizers.Tokenizer.decode(tokens:skipSpecialTokens:)` de
swift-transformers. Leí la implementación real en el checkout local
(`.../SourcePackages/checkouts/swift-transformers/Sources/Tokenizers/Tokenizer.swift:660-675`):

```swift
public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
    let tokenStrings: [String] = tokens.compactMap { model.convertIdToToken($0) }  // map, O(k)
    let decoded = decodeTokens(tokenStrings)                                       // ver abajo
    return cleanUp(text: decoded.joined(separator: ""))                            // join + regex fijas, O(k)
}
```

`decodeTokens` (línea 578-581) delega en el `Decoder` configurado (`Sources/Tokenizers/Decoder.swift`),
y `cleanUp` (línea 584+) aplica una serie fija de `replacingOccurrences` sobre el string ya unido.
No inspeccioné cada implementación de `Decoder` individualmente (hay ~8 variantes según el tipo
de tokenizer), pero cada una recibe y devuelve `[String]` sin anidar bucles sobre el mismo dato
dos veces, así que el coste por llamada a `decode(tokens:)` es **al menos lineal en el número de
tokens pasados** (`O(k)`), posiblemente algo más por las transformaciones de string, pero no hay
indicio de que sea cuadrático dentro de una sola llamada.

Además, dentro de `NaiveStreamingDetokenizer.next()` hay dos operaciones adicionales sobre
`String` que también son `O(longitud)`: `newSegment.count` y `segment.count` (línea 97-98) —
`String.count` en Swift recorre grafemas, no es O(1) — aunque su coste está dominado por el de
`decode()` mismo.

### Veredicto de complejidad

Sea **m** el número de tokens acumulados en `segmentTokens` desde el último salto de línea
(no el total de tokens generados, **N**). Cada llamada a `next()` cuesta `O(m)` (por el
`decode()` sobre el buffer completo), y `m` crece en 1 en cada token dentro del mismo segmento.
Sobre un segmento de longitud `m`, el coste acumulado es la suma `1 + 2 + ... + m = O(m²)`.

- **Si el texto generado tiene saltos de línea frecuentes** (cadencia acotada, p. ej. una línea
  cada L tokens), el buffer se resetea regularmente y el coste total sobre N tokens es
  `O(N · L)` — lineal en N si L es una constante acotada.
- **Si no hay saltos de línea** (peor caso: una sola línea larga de JSON, base64, código
  minificado, CSV de una fila, o simplemente prosa sin `\n` durante toda la generación), `m`
  crece hasta `N` y el coste total es `O(N²)` sobre la generación completa.

Es decir: el backlog item tiene razón, pero de forma **condicional**, no incondicional. El
framing de `DOCS/memory-kv-cache.md` item 8 ("potencialmente O(n²) en segmentos largos") ya era
preciso — no es que la premisa fuera errónea, es que depende de la forma del output. Esto **no**
es un caso donde se pueda descartar el item diciendo "ya es O(n) amortizado"; el peor caso es
real y alcanzable con salidas de modelo perfectamente normales (streaming de JSON estructurado,
tool calls, código, listas largas de números).

## Media materialization audit

### Audio: `Libraries/MLXLMCommon/UserInput+Audio.swift`

`asMLXArray(processing:)`, caso `.url(let url)` (líneas 26-64):

```swift
var samples: [Float] = []                                          // línea 48

while let sampleBuffer = output.copyNextSampleBuffer() {           // línea 50
    ...
    var chunk = [Float](repeating: 0, count: byteCount / MemoryLayout<Float>.size)  // línea 54
    CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: byteCount, destination: &chunk)  // línea 55-56
    samples.append(contentsOf: chunk)                               // línea 57
}
...
return MLXArray(samples)                                            // línea 64
```

Todo el audio se lee en un `while` que acumula **todos** los chunks del `AVAssetReader` en el
array `samples` (línea 48, 57) antes de construir el `MLXArray` final (línea 64). Para un audio
largo esto significa: (a) el array `samples` completo vive en memoria antes de que exista el
`MLXArray`, y (b) durante la conversión `MLXArray(samples)` hay un pico transitorio de ~2x
(el array Swift `[Float]` más su copia en el backing store de `MLXArray`). No hay chunking ni
streaming hacia el modelo — el pico de CPU/memoria es proporcional a la duración total del audio.

### Imagen: `Libraries/MLXVLM/MediaProcessing.swift`

`asMLXArray(_:colorSpace:)` (líneas 168-197), el punto de materialización final de toda imagen
antes de llegar al modelo:

```swift
var data = Data(count: w * h * bytesPerPixel)             // línea 180, bytesPerPixel = 16 (RGBAf)
data.withUnsafeMutableBytes { ptr in
    context.render(image, toBitmap: ptr.baseAddress!, ...)  // línea 182-184: render CPU completo
    context.clearCaches()
}
var array = MLXArray(data, [h, w, 4], type: Float32.self)  // línea 188
```

`CIImage` es normalmente un grafo perezoso; esta llamada es el punto donde se fuerza el render
completo a un buffer CPU (`Data`) a resolución completa en float32 (16 bytes/píxel). Esto es
**inevitable** — el modelo necesita el array materializado eventualmente — pero confirma el
framing de `DOCS/memory-kv-cache.md` item 7 sobre `array.asCIImage()`.

`UserInput.Image.asCIImage()`, caso `.array(let array)` (`Libraries/MLXLMCommon/UserInput.swift:127-170`):

- Línea 136: `array.max().item(Float.self)` fuerza una evaluación/reducción completa del
  `MLXArray` (posible sync GPU→CPU) solo para decidir si la entrada está en rango `0...1` o
  `0...255`.
- Línea 162: `array.asData()` materializa el array completo a `Data` en CPU antes de construir
  el `CIImage` de vuelta (línea 166-169).

Ambos son de tamaño acotado por una sola imagen — no escalan con nada más que la resolución de
esa imagen, así que, igual que `asMLXArray`, son materializaciones **necesarias**, no
optimizaciones evidentes.

**Video — el hotspot más claro de "podría hacerse streaming pero no lo hace":**
`_asProcessedSequence` tiene dos variantes (asset: líneas 412-464; frames en memoria: líneas
466-530), y ambas siguen el mismo patrón:

```swift
var ciImages: [CIImage] = []
var timestamps: [CMTime] = []
for await result in generator.images(for: sampledTimes) {   // línea 445 / bucle equivalente en 501
    ...
    ciImages.append(try frame.image.asCIImage())             // línea 451 / línea 519
    ...
}
let framesAsArrays = ciImages.map { $0.asMLXArray() }         // línea 458 / línea 524
```

Se recogen **todos** los frames muestreados de un vídeo en `ciImages` antes de convertir
ninguno a `MLXArray`, y luego se materializan **todos** a la vez en `framesAsArrays` (línea 458 /
524), que queda con N arrays float32 de resolución completa vivos simultáneamente. A diferencia
de imagen/audio de una sola instancia, esto **sí escala sin límite** con la duración del vídeo y
el `samplesPerSecond`/`targetFPS` elegido — nada impide, en principio, procesar frame por frame
y liberar el `CIImage` anterior antes de pedir el siguiente, o batchear en grupos pequeños en
lugar de materializar todo el vídeo de una vez.

### Resumen: evitable vs inevitable

| Lugar | Escala con | ¿Evitable hoy? |
|---|---|---|
| `UserInput+Audio.swift:48-64` (acumulación de samples) | duración del audio | Sí — podría preasignar buffer con `duration` conocida o escribir directo a un backing store, en vez de `append(contentsOf:)` repetido |
| `MediaProcessing.swift:442-458` / `494-524` (todos los frames de vídeo a la vez) | duración del vídeo × fps muestreado | Sí — es el hotspot más claro; podría procesarse por lotes o streaming |
| `MediaProcessing.swift:168-197` (`asMLXArray` de una imagen) | resolución de una imagen | No — el modelo necesita el array materializado |
| `UserInput.swift:136` (`array.max().item()`) | tamaño de una imagen | No, pero es un doble-recorrido sobre el array (uno para el max, otro implícito en `asData()` después) que podría fusionarse |
| `UserInput.swift:162` (`array.asData()`) | tamaño de una imagen | No — requerido para construir el `CIImage` |

## Microbenchmark implementado

`Libraries/BenchmarkHelpers/DetokenizerBenchmarks.swift` expone
`benchmarkStreamingDetokenization(...)`. Usa `BenchmarkStats`, warm-up separado y entradas
sintéticas sin red ni modelo. A diferencia de un benchmark basado en `NoOpTokenizer`, el
tokenizer sintético hace trabajo lineal real por cada ID y cuenta `decodedTokenVisits`. Esta
métrica determinista permite distinguir complejidad algorítmica del ruido del reloj.

**Variable crítica que el benchmark tiene que barrer explícitamente: la cadencia de saltos de
línea.** Es precisamente lo que decide si el resultado medido es O(n) o O(n²) — un benchmark que
solo pruebe una cadencia fija "demostraría" una sola rama del comportamiento y podría, por
accidente, dar la impresión de que el item del backlog está resuelto cuando solo se cubrió el
caso feliz.

**No reusar `NoOpTokenizer`** (`BenchmarkHelpers.swift:25-32`) — su `decode(tokenIds:)` devuelve
siempre `""`, es decir, coste `O(1)` constante, lo cual aplanaría cualquier curva y no mediría
nada real. El stub sintético necesita un `decode` con coste genuinamente `O(k)` (p. ej. mapear
cada id a un string corto fijo y hacer `joined`), para que la medición refleje el mecanismo real
descrito arriba.

Ejemplo:

```swift
let results = benchmarkStreamingDetokenization(
    tokenCounts: [512],
    newlineCadences: [nil],
    strategies: [.unbounded, .bounded(contextTokens: 16)],
    runs: 7
)
```

En el test Debug del 15 de julio de 2026 sobre este Mac, 512 tokens sin saltos de línea dieron:

| Estrategia | Mediana | IDs visitados por `decode` | Visitas/token |
|---|---:|---:|---:|
| Buffer histórico sin límite | 4.963 ms | 131328 | 256.50 |
| Contexto acotado a 16 | 1.204 ms | 16504 | 32.23 |

Los tiempos son una observación local, no un compromiso de rendimiento. Las visitas son
deterministas: el caso histórico sigue la suma triangular `N×(N+1)/2`; el acotado visita como
máximo una constante por token y por tanto es O(N).

Comando reproducible:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test \
  -scheme mlx-swift-lm-Package -destination platform=macOS \
  -only-testing:MLXLMTests/DetokenizerBenchmarkTests
```

### Corrección del path de producción

`BoundedStreamingDecodeTokenizer` es un protocolo de capacidad aditivo y opcional, así que no
añade requisitos a `Tokenizer` ni cambia su witness table. Los tokenizers custom conservan
exactamente el buffer 3.x sin límite salvo que adopten esta capacidad. El adaptador
`#adaptHuggingFaceTokenizer` la adopta con un solapamiento conservador de 16 tokens para los
decoders soportados por swift-transformers. `NaiveStreamingDetokenizer` compacta solo después de
emitir un fragmento Unicode completo y conserva ese solapamiento como contexto para el siguiente
decode. No se comparte ni se retiene ningún `MLXArray`.

El contrato es deliberadamente opt-in: un tokenizer con un decoder custom de contexto no acotado
debe devolver `nil`. Anunciar un límite incorrecto puede cambiar el sufijo decodificado.

## Mitigaciones de materialización

- Vídeo: `MediaProcessing` materializa cada frame procesado como `MLXArray` dentro del bucle de
  muestreo. La API sigue devolviendo todos los arrays, pero ya no mantiene simultáneamente la
  colección completa de `CGImage`/`CIImage` y la colección creciente de arrays MLX.
- Audio: `UserInput.Audio.url` acumula PCM en `Data` y copia cada `CMBlockBuffer` directamente en
  su región final. Se elimina el `[Float]` temporal por chunk y el posterior
  `append(contentsOf:)`. La reserva anticipada se limita a 64 MiB para no hacer una asignación
  eager descontrolada ante audios muy largos.
- Imagen única: no cambia; el buffer RGBAf de `asMLXArray` sigue siendo una materialización
  necesaria y acotada por la resolución de esa imagen.

Los tests de vídeo conservan número de frames, timestamps y shapes; los nuevos tests de audio
validan dtype/shape y el passthrough de `.array`. El pico RSS/CPU exacto depende de CoreImage,
AVFoundation, resolución y hardware, por lo que debe medirse con Instruments Allocations/Time
Profiler en el dispositivo objetivo. No se deriva un porcentaje de ahorro de los tests unitarios.

## Validación

La validación se ejecutó en un clon temporal limpio creado desde `e4296d2`, para excluir los
archivos locales duplicados con sufijo ` 2` del worktree compartido:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test \
  -scheme mlx-swift-lm-Package -destination platform=macOS \
  -skipPackagePluginValidation \
  -only-testing:MLXLMTests/DetokenizerBenchmarkTests \
  -only-testing:MLXLMTests/UserInputAudioTests \
  -only-testing:MLXLMTests/MediaProcesingTests
```

Resultado: 15 tests, 0 fallos. Incluye un tokenizer 3.x que no adopta la capacidad nueva, el
camino acotado, Unicode incompleto, vídeo desde fichero/frames y audio desde fichero/array.
`swift-api-digester -diagnose-sdk -abi` entre los módulos anterior y nuevo no reportó requisitos
de protocolo, declaraciones eliminadas ni otros breakages; el protocolo de capacidad aparece
solo como API aditiva. Un paquete consumidor temporal que importa `Tokenizers` y
`MLXHuggingFace` también compiló la expansión real de `#adaptHuggingFaceTokenizer`.

## Verdict

**El problema O(n²) estaba confirmado y queda corregido para el adaptador oficial de
swift-transformers.** Los tokenizers custom mantienen el comportamiento histórico salvo que
declaren explícitamente un contexto acotado. El benchmark conserva ambos modos para detectar
regresiones futuras.

**En media**, se han eliminado las retenciones transitorias más claras sin alterar el resultado
público. Aun así, `ProcessedFrames.frames` y el audio final continúan siendo colecciones completas:
el cambio reduce el pico intermedio, no convierte la API en streaming end-to-end. Una API que
consuma frames/audio incrementalmente requeriría un diseño nuevo y queda fuera de este bloque
compatible con 3.x.
