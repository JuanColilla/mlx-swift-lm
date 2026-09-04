# R-56 · Expert streaming para K2-Horizon MoVA (dos familias de expertos)

> Rama `feature/r56-k2-streaming`, tag `v3.31.4-fork.13`. Checkpoint de
> aceptación: `IFM/K2-Horizon-MoVA-36B-A4B` convertido a 4-bit affine g64
> (20 GB, 4 shards, routers sin cuantizar). Mac de 128 GB, 2026-09-05.

## Qué cambia respecto a Qwen 3.5 / LFM2-MoE

K2-Horizon MoVA enruta **dos** grupos de expertos por capa sparse:

| Familia | Clave | Expertos | top-K | Fila por experto | Alineada a 16 KiB |
|---|---|---|---|---|---|
| `.mlp` | `mlp.switch_mlp.{gate,up,down}_proj` | 100 | 8 | peso 983.040 B · escalas/biases **61.440 B** | peso sí · escalas/biases **no** |
| `.value` | `self_attn.switch_v` (una sola proyección, `silu` por experto) | 64 | 4 | peso 1.310.720 B · escalas/biases 81.920 B | sí |

Por capa: 45 capas sparse (3…47; `mlp_only_layers` [0,1,2] densas). MLP 14,93 GB,
value 4,25 GB, `routedBytes` 19,18 GB.

`ExpertFamily` recorre el diseño de extremo a extremo: `ExpertKey`, índice
(`ExpertOffsetIndex.families`, formato v2), store (un batch = una familia),
un `ExpertSlotBank` por familia con su geometría, sesión
(`resolve(layer:family:…)`, prefetch con un batch en vuelo por familia,
predicción por `StreamedExpertStep` = (capa, familia) porque la atención va
antes del MLP), `StreamedSwitchLinear` junto a `StreamedSwitchGLU`. Qwen y
LFM siguen con una familia; sus suites y experimentos no cambian de resultado.

**Fallo silencioso cubierto por test**: si `isRoutedExpertKey` no reconoce
`switch_v`, `loadWeights` materializa 4,25 GB de expertos de valor residentes
sin que nada lo diga (`ExpertFamilyIndexTests.testValueExpertKeysAreRoutedExpertKeys`,
`K2HorizonStreamedTests.testStreamedLoadSkipsBothFamiliesAndPicksTheStreamedTypes`).

## Alineación: el invariante es sobre el staging, no sobre la fila

`newBufferWithBytesNoCopy` exige puntero **y longitud** múltiplos de página, y
`array.cpp` degrada a `malloc` + copia en silencio si no. Las escalas MLP del
36B (61.440 B/experto) no son múltiplo de 16 KiB. Solución: el buffer de
staging se redondea a página (`ExpertReadBatch.paddedByteCount`), se envuelve
como array plano `uint8`/dtype sobre la longitud redondeada, se hace slice al
número exacto de elementos y reshape — ambos vistas, cero copias.
`ExpertPaddedStagingTests.testPaddedStagingIsWrappedWithoutACopy` lo decide por
**observación** (escribe en el buffer tras crear el array y comprueba que el
array lo ve), no leyendo el asignador. `ExpertOffsetIndex` ya no rechaza filas
no alineadas; solo las anota (`ExpertTensorRecord.isPageAligned`).

## El hallazgo: `sortedIndices: true` sobre pools compactos deriva

La primera pasada de aceptación sobre el 36B dio logits distintos al
residente (peor diferencia absoluta 2,16, relativa 0,09, top-1 16/16), igual
con banco de 3 GiB que de 256 MiB, y **solo en prefill** (decode bit a bit).
Bisección por subcapa con la misma entrada
(`K2HorizonStreamedMacExperiment.testBisectSublayersAgainstTheResidentModel`):
router idéntico, experto compartido idéntico, bytes staged idénticos a los
parámetros residentes, y aun así el producto routed difería en 1 ulp de bf16
en capas sueltas (17, 18, 19, 20, 22, 26…), en ambas familias.

Dentro de la capa 17, con las activaciones reales
(`testSortedIndicesFlagDriftsOnCompactPools`):

| Comparación (gate_proj) | max Δ |
|---|---|
| pools completos `[100,…]`, flag on vs off | 0,000000 |
| pools compactos `[39,…]` con flag on vs completos con flag on | **0,031250** |
| pools compactos, permutación expert-major con flag **off** vs completos | 0,000000 |
| pools compactos sin permutar vs completos | 0,000000 |

Es decir: con los pools completos el flag no cambia nada; con un pool
compacto (un batch staged de los 39 expertos que tocó el prompt, o un banco
de slots) el kernel con `sortedIndices: true` cae en otro orden de
acumulación. No es un error de datos, pero rompe "streamed == residente". Fix:
las capas streamed mantienen la permutación (`gatherSort`, la localidad) y
**nunca** pasan el flag al kernel. Coste de prefill medido: ninguno
apreciable (0,43 s → 0,41 s sobre 20 tokens). Un test con índices aleatorios
no lo detectaba; hizo falta la distribución real de expertos por token.

## Aceptación

Teacher forcing sobre los tokens del volcado Python (`k2-reference-logits`),
16 pasos, 36B 4-bit real (`MLX_R56_K2_EXPERIMENT=1 … --filter K2HorizonStreamedMacExperiment`):

| Brazo | top-1 vs residente | peor Δ abs | pico GPU | decode (test, debug) |
|---|---|---|---|---|
| Residente | — | — | 19,72 GiB | 17,6 tok/s |
| Streamed 3 GiB (MLP 755 slots / value 483) | 16/16 | **0,000000** | 5,06 GiB | 7,5 tok/s |
| Streamed 256 MiB (62 / 40, fuerza expulsiones) | 16/16 | **0,000000** | 2,31 GiB | 6,5 tok/s |
| Control: capas 3↔4 de `.value` intercambiadas | 3/4 | 15,97 (rel 0,61) | | |
| Control: capas 3↔4 de `.mlp` intercambiadas | 3/4 | 8,13 (rel 0,40) | | |

Residente y streamed vs Python: 15/16 (el suelo de ruido Swift-vs-Swift ya
medido en el port residente).

Suites (`swift test --filter 'Expert|Streamed|K2Horizon|LFM2'`): 123 tests,
0 fallos, 12 skipped (los opt-in).

## Generación end-to-end con el tokenizer y el `chat_template.jinja` reales

Arnés standalone (paquete ejecutable en release que depende de este fork
por `path:` y de `swift-transformers` 1.3.3, exactamente lo que pinnea
MLXHub; no se añade la dependencia al paquete para no cambiar lo que
resuelve la app). Carga por `LLMModelFactory.shared.loadContainer(from:using:
#huggingFaceTokenizerLoader())`, `ChatSession`, greedy, 64 tokens.

| Modelo | Brazo | carga | prefill 20 tok | decode | pico GPU |
|---|---|---|---|---|---|
| 3.7B 4-bit | residente | 0,7 s | 27,5 tok/s | **146,9 tok/s** | 2,74 GiB |
| 36B-A4B 4-bit | residente | 1,8 s | 16,5 tok/s | **67,1 tok/s** | 19,72 GiB |
| 36B-A4B 4-bit | streamed 3 GiB (hit MLP 54,6 %, value 36,1 %) | 0,8 s | 34,9 tok/s | **16,7 tok/s** | 5,03 GiB |
| 36B-A4B 4-bit | streamed 1 GiB (hit 0 %) | 0,8 s | 39,6 tok/s | **10,7 tok/s** | 3,03 GiB |

La salida de los tres brazos del 36B es **idéntica carácter a carácter**:

```
User asks: "Explain in two sentences why the sky is blue." Need to give two
sentences. Provide concise explanation: Rayleigh scattering, shorter
wavelengths scatter more. Also mention sun's light and atmosphere. Provide
two sentences.
</ifm|think>
The sky appears blue because molecules in Earth's atmosphere scatter sunlight
more efficiently at short (blue
```

Dos observaciones para el host:

1. **El tokenizer de 250k y el template funcionan en Swift** (un solo BOS,
   `<|ifm|begin_of_text|><|ifm|im_start|>user\n…<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n`).
   El template **abre el canal de razonamiento `<ifm|think>` en el prompt**,
   así que el modelo empieza pensando y cierra con `</ifm|think>`: es el
   mismo patrón que Qwen3.5 (sembrar el stream con el tag de apertura para
   que la UI de razonamiento funcione en vivo), con tags distintos de
   `<think>`.
2. **Un banco por debajo del working set de un token no sirve de nada**:
   45 capas × top-8 = 360 expertos MLP (1,14 GiB) y 45 × 4 = 180 de valor
   (0,25 GiB) por token; con 251/161 slots (1 GiB) el hit rate es 0 % y cada
   token relee ~1,4 GB. El reparto proporcional deja al banco de 3 GiB en
   54,6 % / 36,1 %. El suelo útil para K2 está en torno a 1,5 GiB.

## No verificado

- iOS y dispositivo físico: solo Mac. Techo de 6,0 GiB del iPhone 17 Pro Max
  con las 3 capas densas (`intermediate_size` 10240) + embeddings de 250k ×
  2560 + banco: pendiente de medir en el aparato.
- Prefetch temporal por paso (capa, familia): cableado y cubierto por los
  tests de unidad de `ExpertPrefetcher`, sin medida de hit rate en el 36B.
- Prompts largos: la permutación sin flag no se ha cronometrado más allá de
  20 tokens.
