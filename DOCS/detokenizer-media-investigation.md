# Detokenizer y media materialization: investigación

Investigación del backlog item #15 (`DOCS/tech-debt-and-research-backlog.md`, prioridad
baja-pero-estrategica): "Medir coste O(n^2) de streaming detokenization y picos CPU de
imagen/audio". Este documento es solo investigación/análisis de código; no implementa
ningún benchmark ni cambia código de producción.

Ground truth previo: `DOCS/memory-kv-cache.md`, sección "Riesgos principales", items 7 y 8.

## Detokenizer complexity: what the code actually does

Implementación: `Libraries/MLXLMCommon/Tokenizer.swift:71-114` (`NaiveStreamingDetokenizer`).

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

## Proposed microbenchmark (not implemented)

Estilo de referencia: `Libraries/BenchmarkHelpers/BenchmarkHelpers.swift` (`BenchmarkStats`,
warm-up + N runs cronometrados, `CFAbsoluteTimeGetCurrent()`) y, más cercano en intención,
`Libraries/BenchmarkHelpers/SamplingBenchmarks.swift` (inputs sintéticos, sin cargar modelo,
`measureSampling`-style helper con warm-up separado de runs cronometrados).

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

Sketch (prosa + pseudo-Swift, no código de producción):

```swift
// Synthetic tokenizer: decode(tokenIds:) cuesta O(k) real (map + join),
// a diferencia de NoOpTokenizer (O(1), no sirve para este benchmark).
private struct SyntheticLinearTokenizer: Tokenizer {
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "tok\($0 % 1000) " }.joined()
    }
    // resto de la conformance con stubs triviales, como NoOpTokenizer
}

public struct DetokenizerBenchmarkResult: Sendable {
    let bufferLength: Int
    let newlineEvery: Int?   // nil = sin saltos de línea (peor caso)
    let stats: BenchmarkStats
}

// Genera una secuencia sintética de `count` "tokens" donde, cada
// `newlineEvery` tokens, el decode producido termina en "\n" (para
// disparar startNewSegment()). `newlineEvery == nil` nunca inserta "\n".
private func syntheticTokenStream(count: Int, newlineEvery: Int?) -> [Int] { ... }

public func benchmarkStreamingDetokenizer(
    bufferLengths: [Int] = [100, 1_000, 10_000],
    newlineCadences: [Int?] = [10, 100, nil],   // nil = peor caso, sin resets
    runs: Int = BenchmarkDefaults.decodingRuns
) -> [DetokenizerBenchmarkResult] {
    var results: [DetokenizerBenchmarkResult] = []
    for cadence in newlineCadences {
        for length in bufferLengths {
            let tokens = syntheticTokenStream(count: length, newlineEvery: cadence)
            var times: [Double] = []
            for _ in 0..<runs {
                var detok = NaiveStreamingDetokenizer(tokenizer: SyntheticLinearTokenizer())
                let start = CFAbsoluteTimeGetCurrent()
                for token in tokens {
                    detok.append(token: token)
                    _ = detok.next()
                }
                times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            }
            results.append(.init(bufferLength: length, newlineEvery: cadence,
                                  stats: BenchmarkStats(times: times)))
        }
    }
    return results
}
```

Lectura esperada del resultado: para `newlineEvery: nil` (peor caso), tiempo vs `bufferLength`
debería trazar una curva cuadrática (tiempo/length creciendo con length, no constante); para
`newlineEvery: 10` o `100` (cadencias acotadas), el tiempo por token debería mantenerse
aproximadamente constante según crece `bufferLength`, confirmando el comportamiento amortizado
lineal. Si ambas curvas resultan planas, la hipótesis de O(n²) estaría refutada empíricamente —
pero dado el análisis estático de arriba, no se espera ese resultado para `newlineEvery: nil`.

Esto es solo una propuesta; implementarlo y validarlo como API pública nueva del paquete necesita
su propia sesión revisada (no se añade ningún archivo Swift nuevo en esta investigación).

## Verdict

**El detokenizer no es un caso cerrado ni una falsa alarma.** El análisis estático confirma que
`NaiveStreamingDetokenizer` es `O(m²)` por segmento entre saltos de línea, con `m` pudiendo
llegar a `N` (la generación completa) si el modelo produce una sola línea larga sin `\n` — un
escenario nada exótico para JSON, tool calls, código o listas. La premisa del backlog item
(potencial O(n²)) es correcta; lo que faltaba era la caracterización condicional, que ahora está
documentada arriba con cita de línea exacta.

**Prioridad:** sigue siendo razonable mantenerlo como "baja-pero-estratégica". No es un blocker
de producción hoy (la mayoría de generación de texto natural tiene saltos de línea razonablemente
frecuentes), pero es barato de confirmar empíricamente (el microbenchmark propuesto no necesita
modelo ni datos reales) y, si se confirma el peor caso, barato de arreglar: la solución obvia es
mantener un puntero de "posición ya emitida" en el tokenizer o decodificar solo el delta de
tokens nuevos en vez de todo `segmentTokens`, cambio acotado a `Tokenizer.swift`.

**En media, el hotspot más accionable es la acumulación de todos los frames de vídeo antes de
convertirlos (`MediaProcessing.swift:442-458` / `494-524`)**, seguido de la acumulación completa
de samples de audio antes de construir el `MLXArray` (`UserInput+Audio.swift:48-64`). Ambos
escalan con la duración del input y no tienen equivalente en el path de imagen única, que ya está
acotado por definición (una imagen, una resolución). Ninguno de los dos requiere romper la API
pública para mitigarse — son cambios internos de estrategia de acumulación/streaming.
