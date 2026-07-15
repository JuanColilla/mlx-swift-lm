# VLM media budgets

Este documento es una referencia por arquitectura de cómo cada VLM registrado en
`Libraries/MLXVLM/` convierte imagen/vídeo de entrada en tokens visuales, y qué
determina el presupuesto de memoria resultante. Está derivado leyendo el código
fuente en esta rama (`VLMModelFactory.swift`, `Libraries/MLXVLM/Models/*.swift`,
`MediaProcessing.swift` y los tests bajo `Tests/MLXLMTests/`), **no de mediciones
en hardware real**. Todas las cifras de "memoria aproximada" son fórmulas
cualitativas, no bytes medidos — cualquier número de memoria real requeriría un
perfilado con Instruments/`os_proc_available_memory` sobre un modelo concreto.

Cubre las 16 arquitecturas distintas detrás de las 18 entradas de
`VLMTypeRegistry.shared` (`Libraries/MLXVLM/VLMModelFactory.swift:89-108`):
PaliGemma, Qwen2VL, Qwen2.5VL, Qwen3VL, Qwen3.5, Qwen3.5-MoE, Idefics3, SmolVLM2
(alias de clase de Idefics3), Gemma3, Gemma4, Gemma4Unified, FastVLM (también
sirve `llava_qwen2`), Pixtral, Mistral3VLM, LFM2VL (registrada dos veces, con y
sin guion: `lfm2_vl` / `lfm2-vl`) y GlmOcr.

## Cómo leer la tabla

- **Resolución**: si el modelo redimensiona siempre a un tamaño fijo de config, o
  si aplica un algoritmo de resize dependiente del tamaño de la imagen de
  entrada (resolución dinámica).
- **Tiling**: si la imagen se divide en varios tiles/parches de tamaño fijo antes
  de pasar por el encoder visual (más allá del patchify estándar de un ViT).
- **Vídeo**: estrategia de muestreo de frames, si existe.
- **Tokens visuales**: la fórmula o el ejemplo concreto tomado de un test.
- **Memoria**: nota cualitativa. La KV cache por token visual usa la misma forma
  `[B, kvHeads, seqLen, headDim]` que un token de texto — ver
  `Libraries/MLXLMCommon/KVCache.swift:401-404` — así que el coste incremental de
  cada token visual es: `2 (K+V) × num_capas × kv_heads × head_dim × bytes_por_elemento`.
  Esto significa que el modelo no distingue coste por modalidad: lo único que
  importa para el presupuesto de memoria de KV es **cuántos tokens visuales**
  produce el paso de preprocesado, que es justo lo que tabula este documento.

## Tabla por arquitectura

| Arquitectura | Resolución | Tiling | Vídeo | Tokens visuales | Memoria |
|---|---|---|---|---|---|
| **PaliGemma** (1/2) | Fija (`config.size`, p.ej. 224/448/896) | No | No | `imageSequenceLength` fijo del `preprocessor_config.json` (p.ej. 256/1024/4096 para 224/448/896 a patch 14) | KV lineal en tokens fijos; predecible por variante |
| **Qwen2VL** | Dinámica (`smart_resize`) | No (patchify + merge, no tiles) | Sí, 2 fps, sin cap de frames | `(gridH×gridW×gridT)/mergeSize²`; por defecto acotado a ≤1280 tokens/imagen | Puede escalar de 4 a miles de tokens por imagen; vídeo sin cap de frames es un riesgo real |
| **Qwen2.5VL** | Dinámica (`smart_resize`, igual que Qwen2VL) + ventaneo en el encoder | No | Sí, 2 fps, sin cap de frames | Igual fórmula que Qwen2VL | Igual que Qwen2VL |
| **Qwen3VL** | Dinámica, rango por defecto mucho más amplio | No | Sí, 2 fps, sin cap de frames | 4 a 16384 tokens/imagen por defecto (`min=4×28²`, `max=16384×28²` px) | El caso peor por imagen es el mayor de toda la tabla |
| **Qwen3.5 / Qwen3.5-MoE** | Reutiliza la vision tower de Qwen3VL (`Qwen3VLConfiguration.VisionConfiguration`) | No | No verificado en el processor (no hay registro de `Qwen35Processor`; usa el mismo `Qwen3VLProcessor`) | Igual que Qwen3VL | Igual que Qwen3VL; MoE solo afecta al FFN de texto |
| **Idefics3** | Fija (384×384) | No — 1 imagen, sin tiling en `Idefics3Processor` | No | Modelo de 1 solo `imageTokenId` insertado en el prompt (ver "Gaps") | Bajo por imagen si el tiling realmente no aplica aquí |
| **SmolVLM2** (clase = Idefics3, processor propio) | Fija por tile (384 modelos grandes / 512 modelos 200-500M) tras tiling dinámico | Sí — grid `nRows×nCols` + 1 tile global | Sí, ~1 fps con multiplicador para vídeos cortos; `maxVideoFrames` hardcodeado a 20 | `(nRows×nCols + 1) × imageSequenceLength(64)`. Ejemplos reales: 512×384→2 tiles→128 tok; 1024×768→5 tiles→320 tok; 4096×3072→13 tiles→832 tok | Escala con el tamaño real de la imagen; el fix de #208 evita upscaling espurio |
| **Gemma3** | Fija (`config.imageSize`, p.ej. 896) | No (config trae `do_pan_and_scan` pero el processor lo ignora — ver "Gaps") | No | 256 tokens fijos por imagen (`config.imageSeqLength`) | El más predecible de la tabla: siempre 256 tok/imagen |
| **Gemma4** (no unificado) | Fija (800×800 por defecto) | No | No leído desde `UserInput.videos` (ver "Gaps") | 280 tokens por defecto (`imageSeqLength`) | Predecible, similar a Gemma3 |
| **Gemma4Unified** | Dinámica, preserva aspect ratio, acotada por `maxSoftTokens` | No (patchify + padding a `maxSoftTokens`, no tiles) | Vía `videoPixelValues`/`videoPositionIds` en el modelo, pero el processor no lee `UserInput.videos` (ver "Gaps") | Hasta `maxSoftTokens` (del config.json, no hardcodeado); patches reales se rellenan/truncan a ese máximo | Presupuesto acotado de forma explícita — el más "seguro" de los dinámicos |
| **FastVLM** (`fastvlm` y `llava_qwen2`) | Fija (`config.cropSize`, pad-to-square primero) | No | No | Placeholder único `-200` en el prompt; el recuento real de tokens lo decide la vision tower (FastViTHD) en tiempo de merge, no el processor — no verificable sin ejecutar el modelo | No calculable desde el processor solo |
| **Pixtral** | Dinámica, `longestEdge` por defecto 1540 | No (patchify puro, sin merge) | No | `numPatchesH × numPatchesW` tras padding a múltiplos de `patchSize` (p.ej. hasta ~110×110=12100 tok en el peor caso a 1540px/patch14) | Sin spatial merge → el más caro por píxel de los dinámicos sin tiling |
| **Mistral3VLM** | Dinámica pero acotada a `patchSize×24` (336px para patch14) | No | No | `(numPatchesH/spatialMergeSize) × (numPatchesW/spatialMergeSize)`, con `spatialMergeSize` por defecto 2 → tope ≈144 tok/imagen | Mucho más barato que Pixtral puro gracias al cap de 336px + merge/2 |
| **LFM2VL** | Dinámica vía tiling | Sí — grid de tiles de `tileSize` (512 por defecto), hasta `maxTiles` (10) **por dimensión** | No (no hay referencia a `input.videos` en el processor) | `(totalPatchesH × totalPatchesW) / downsampleFactor²` tras tiling; con defaults, hasta 10×10 tiles de 512px es el caso extremo permitido por el código | El cap de 10×10 tiles es una cota de código, no necesariamente lo que un config.json real usa — verificar por modelo |
| **GlmOcr** | Dinámica (`smart_resize`, esquema Qwen-like) | No | Sí, vía `videoTokenId` + posiciones 3D estilo Qwen | `size.shortestEdge`/`size.longestEdge` del config, dividido por merge² | Mismo patrón de riesgo que la familia Qwen-VL |

## Notas por arquitectura con comportamiento no trivial

### Familia Qwen-VL (Qwen2VL, Qwen2.5VL, Qwen3VL, Qwen3.5)

El resize dinámico (`smart_resize`) vive en `Libraries/MLXVLM/Models/QwenVL.swift:122-173`
(`QwenVL.targetSize`). Redondea la imagen a múltiplos de `factor = patchSize × mergeSize`
y la reescala para que el total de píxeles quede entre `minPixels` y `maxPixels`.
El patchify + reordenado a formato de merge está en `QwenVL.swift:212-256`.

- Qwen2VL/Qwen2.5VL (`Libraries/MLXVLM/Models/Qwen2VL.swift:1202-1207`): defaults
  `minPixels = 3136` (56×56 px), `maxPixels = 12_845_056`, pero el processor
  acota además a `min(config.maxPixels, 1280 × factor²)` en
  `Qwen2VL.swift:800-817` — es decir, por defecto el tope real es **1280 tokens
  visuales por imagen**, no el máximo teórico del config.
- Qwen3VL (`Qwen3VL.swift:224-231`, confirmado en
  `Tests/MLXLMTests/Qwen3VLProcessorConfigTests.swift:74-78`): defaults
  `minPixels = 4×28×28` (4 tokens), `maxPixels = 16384×28×28` (16384 tokens) —
  un rango mucho más amplio que Qwen2VL/2.5VL. Nota: estos defaults hardcodeados
  usan el factor "legado" 28 (`patchSize=14 × mergeSize=2` de Qwen2VL), aunque el
  config real de un modelo Qwen3VL pueda declarar `patch_size: 16` — el mismo test
  también verifica que un config.json con el formato nuevo (`size.longest_edge` /
  `size.shortest_edge`, visto en modelos PARO Qwen3.5/3.6) se decodifica
  correctamente en vez de caer en el default legado.
- Vídeo: Qwen2.5VL (`Qwen25VL.swift:864-865`) y Qwen3VL (`Qwen3VL.swift:123-124`)
  muestrean a **2 fps fijos**, vía `MediaProcessing.asProcessedSequence(video:
  targetFPS:)` sin pasar `maxFrames` — el default de esa función es `Int.max`
  (`MediaProcessing.swift:355-361`). **No hay tope de frames en el código para
  esta familia**: un vídeo largo puede generar un número de tokens visuales sin
  límite superior explícito.
- Qwen3.5/Qwen3.5-MoE reutilizan el mismo tipo `VisionConfiguration` que Qwen3VL
  vía `typealias` (`Qwen35.swift:179`) y la misma `Qwen3VLVision.VisionModel`
  (`Qwen35.swift:908`). No hay una entrada `Qwen35Processor` en
  `VLMProcessorTypeRegistry` (`VLMModelFactory.swift:111-144`), así que en
  tiempo de ejecución usan el `processor_class` que declare el `config.json` del
  modelo — previsiblemente `Qwen3VLProcessor`, pero esto no está verificado
  contra un checkpoint real.

### Idefics3 vs SmolVLM2 — mismo modelo, distinto processor

`SmolVLM2Configuration`/`SmolVLM2` son alias directos de `Idefics3Configuration`/
`Idefics3` (`SmolVLM2.swift:16-17`) — es el mismo modelo Swift. Lo que cambia es
el processor:

- `Idefics3Processor` (`Idefics3.swift:838-928`) solo acepta una imagen, la
  redimensiona a 384×384 fijo, y en el camino de "single image" inserta **un
  único** `imageTokenId` en el prompt (`Idefics3.swift:883-884`), sin expandirlo
  al recuento de tokens visuales reales. `prepareInputsForMultimodal`
  (`Idefics3.swift:686-730`) sí espera múltiples posiciones de imagen agrupadas
  en chunks de tamaño `imageFeatures.dim(1)` (64) — con un solo token insertado
  esa lógica de chunking nunca se activa igual que en SmolVLM. No pude confirmar
  si esto es intencional (p.ej. si real Idefics3 8B usa un solo tile por
  diseño) o una discrepancia frente al pipeline de tiling de SmolVLM — ver
  "Gaps".
- `SmolVLMProcessor` (`SmolVLM2.swift:79-375`) sí tiliza: `tiles(from:)`
  (`SmolVLM2.swift:182-218`) calcula un grid `nRows × nCols` de tiles de
  `fixedImageSize` (384 o 512 según el modelo,
  comentario en `SmolVLM2.swift:92`), más siempre 1 tile global adicional. El
  fix de la issue #208 (`SmolVLM2.swift:186-191`, verificado en
  `Tests/MLXLMTests/SmolVLM2TilingTests.swift`) evita hacer upscale de imágenes
  pequeñas antes de tilizar — antes del fix, una imagen de 512×384 se subía a
  2048×1536 y generaba 12 tiles (~1140 tokens de prompt) en vez de 1.
  Ejemplos verificados por test: 512×384 → 1 tile (+1 global = 2×64=128 tok);
  1024×768 → 4 tiles (+1 global = 5×64=320 tok); 4096×3072 → 12 tiles (+1
  global = 13×64=832 tok).
- Vídeo en SmolVLM2: `SmolVLM2.swift:322-343`. FPS objetivo:
  `max((10 − 0.9×duration_s) × targetVideoFPS, 1)` (`SmolVLM2.swift:324-327`),
  es decir, más denso para vídeos cortos y ~1 fps para vídeos ≥10s.
  `maxVideoFrames` está **hardcodeado a 20** en `SmolVLM2.swift:94`, con un
  comentario que indica que ignora deliberadamente `config.videoSampling.maxFrames`
  del `preprocessor_config.json`.

### Gemma3 — fijo, y `pan_and_scan` no implementado

`Gemma3Processor.preprocess` (`Gemma3.swift:1062-1084`) fuerza siempre
`targetSize = config.imageSize × config.imageSize` e ignora cualquier resize
pedido por el llamador (comentario explícito en `Gemma3.swift:1066`:
"Always use the vision configuration's imageSize. Ignore UserInput resize
setting."). El número de tokens visuales es fijo:
`config.imageSeqLength` = **256** (`Gemma3.swift:1110,1145`).
`Gemma3ProcessorConfiguration` decodifica los campos `do_pan_and_scan`,
`pan_and_scan_max_num_crops`, etc. (`Gemma3.swift:1152-1155`) pero
`Gemma3Processor.preprocess` nunca los lee — el "pan and scan" multi-crop de
Gemma 3 en Python/transformers **no está portado** en este processor Swift.

### Gemma4 vs Gemma4Unified — dos processors muy distintos

- `Gemma4Processor` (`Gemma4.swift:2627-2701`) es fijo: resize a
  `config.fixedSize` (800×800 por defecto, comentario en `Gemma4.swift:2755-2756`
  explica que ese tamaño mantiene el recuento de patches bajo el presupuesto
  `280 × 3²` de Gemma4), `imageSeqLength` por defecto 280
  (`Gemma4.swift:2735`).
- `Gemma4UnifiedProcessor` (`Gemma4.swift:2954-3051`) es dinámico:
  `aspectRatioPreservingSize(for:)` (`Gemma4.swift:2918-2951`) calcula un tamaño
  objetivo que preserva aspect ratio y queda acotado por `maxSoftTokens`
  (número de "soft tokens" del config.json, sin default hardcodeado visible en
  este archivo — viene de `preprocessor_config.json`/`processor_config.json`).
  `patchify(_:)` (`Gemma4.swift:2963-3000`) trunca el número real de patches a
  `maxSoftTokens` si sobran, y rellena con padding si faltan, así que el
  recuento de tokens visuales por imagen es **siempre exactamente
  `maxSoftTokens`** (parte relleno, parte real) — un presupuesto explícito y
  acotado, a diferencia de la familia Qwen.
- Ambos processors (`Gemma4Processor.prepare` y `Gemma4UnifiedProcessor.prepare`)
  no tienen ninguna referencia a `input.videos` — a pesar de que el modelo
  `Gemma4` sí acepta `videoPixelValues`/`videoPositionIds` y tiene un
  `videoTokenId` de config (`Gemma4.swift:2778`, `2481-2515`) y
  `Gemma4UnifiedProcessorConfiguration` decodifica `videoTokenId` también
  (`Gemma4.swift:2899`). El soporte de vídeo parece existir a nivel de modelo
  pero no está cableado desde `UserInput` en ninguno de los dos processors —
  ver "Gaps".

### FastVLM — el único caso no calculable desde el processor

`FastVLMProcessor.prepare` (`FastVLM.swift:985-1029`) solo acepta 1 imagen,
la rellena a cuadrado (`paddingToSquare()`) y la redimensiona a
`config.cropSize` fijo. A diferencia de todos los demás modelos, **no expande
el placeholder de imagen a N tokens en el texto**: inserta un único token
centinela `-200` (`FastVLM.swift:974-975, 1008-1011`) y dice al tokenizer que
lo trate como un split point. El recuento real de tokens visuales lo decide la
vision tower FastViTHD (una CNN jerárquica con varias etapas de downsample —
`config.downStride`/`downSamples`, `FastVLM.swift:44-49`) en el momento del
merge de embeddings (`FastVLM.swift:1114-1120`), no en el processor. **No pude
derivar una fórmula cerrada de tokens/imagen desde el código sin trazar todas
las etapas de stride de FastViTHD** — marcado como gap.

### Pixtral vs Mistral3VLM — mismo patchify, presupuestos muy distintos

Ambos comparten el mismo patrón de resize-and-pad-to-patch-multiple
(`Pixtral.swift:1069-1121`, `Mistral3.swift:949-1017`), pero:

- Pixtral (`PixtralProcessor`, `Pixtral.swift:1029-1128`) usa `longestEdge`
  por defecto **1540 px** (`Pixtral.swift:1070`) y **no aplica spatial merge**:
  `numImageTokens = numPatchesH × numPatchesW` directo
  (`Pixtral.swift:1119-1121`). A `patchSize=14` eso puede llegar a ~110×110 =
  ~12100 tokens en el caso extremo de una imagen cuadrada grande.
- Mistral3VLM (`Mistral3VLMProcessor`, `Mistral3.swift:925-1018`) acota el
  borde más largo a `patchSize × 24` (336 px para `patchSize=14`,
  comentario explícito en `Mistral3.swift:959`: "Pixtral vision expects 24×24
  patches"), y además divide por `spatialMergeSize` (por defecto 2,
  `Mistral3.swift:1039`). El resultado son como máximo ~144 tokens/imagen —
  casi 100× menos que el peor caso de Pixtral puro, pese a compartir la misma
  vision tower Pixtral.

### LFM2VL — tiling con un cap potencialmente alto

`LFM2VLProcessor.splitIntoPatchesAndPreprocess` (`LFM2VL.swift:683-745`)
calcula `numTilesH`/`numTilesW` como `min(maxTiles, ceil(dim/tileSize))`
**de forma independiente en cada eje** (`LFM2VL.swift:699-700`). Con los
defaults del código (`tileSize=512`, `maxTiles=10`, `encoderPatchSize=16`,
`downsampleFactor=2` — `LFM2VL.swift:1284-1287`), el caso extremo permitido
por el código es un grid de 10×10 tiles de 512px, cada uno con
`(512/16)²/2² = 256` tokens tras el downsample, es decir hasta 25600 tokens
visuales en teoría. No hay evidencia en este archivo de que un checkpoint real
llegue a usar `maxTiles=10` en ambos ejes simultáneamente — es una cota de
código, no un valor observado. `minTiles`/`useThumbnail` también existen en
`LFM2VLVisionConfiguration` (`LFM2VL.swift:1243-1244,1262`) pero no aparecen
usados en `LFM2VLProcessor`, así que puede que sean vestigiales o consumidos en
otro punto no localizado.

### GlmOcr — arquitectura Qwen-like

`GlmOcrProcessor` (`GlmOcr.swift:765-820`) reutiliza el mismo patrón de
`QwenVL.targetSize`/`patchify` que la familia Qwen-VL, con
`minPixels = size.shortestEdge` y `maxPixels = size.longestEdge` tomados
directamente del config, sin default legado hardcodeado
(`GlmOcr.swift:1233-1234`). El modelo tiene `videoTokenId` (default 59281,
`GlmOcr.swift:1177`) y usa el mismo esquema de posiciones 3D estilo Qwen para
vídeo — no se localizó un cap de frames explícito, igual que en la familia
Qwen-VL.

## Gaps / lo que no se pudo determinar desde el código

- **FastVLM**: no hay fórmula cerrada de tokens visuales/imagen sin trazar
  todas las etapas de stride de la vision tower FastViTHD. Necesita una
  ejecución real con un `config.json` concreto (p.ej.
  `mlx-community/FastVLM-0.5B-bf16`) inspeccionando la forma del tensor que
  sale de `visionModel(...)`.
- **Idefics3Processor vs SmolVLMProcessor**: no se pudo confirmar si el
  comportamiento de "un solo `imageTokenId` sin expandir" en
  `Idefics3Processor` (a diferencia del tiling completo de `SmolVLMProcessor`)
  es el comportamiento correcto para los checkpoints Idefics3 puros (no
  SmolVLM), o si es una discrepancia frente al pipeline Python de referencia.
- **Vídeo en Gemma4 / Gemma4Unified**: el modelo acepta `videoPixelValues` y
  tiene `videoTokenId` de config, pero ninguno de los dos processors lee
  `input.videos` en `prepare(input:)`. No está claro si el soporte de vídeo se
  activa por otra vía (construcción manual de `LMInput`) o si simplemente no
  está cableado todavía.
- **Cap de frames de vídeo en la familia Qwen-VL / GlmOcr**: el código no
  aplica ningún `maxFrames` al muestrear vídeo a 2 fps
  (`MediaProcessing.asProcessedSequence` usa `Int.max` por defecto). Esto es
  coherente con lo ya documentado en la skill `mlx-swift` (`references/vlm.md`),
  pero no hay guard-rail en el código de este repo — un vídeo largo puede
  generar un prompt de tamaño no acotado.
- **Memoria real en bytes**: ningún número de esta tabla está medido en
  hardware. La fórmula de KV cache citada (`Libraries/MLXLMCommon/KVCache.swift:401-404`)
  es correcta estructuralmente, pero traducirla a MB/GB reales requiere
  `head_dim`, `num_kv_heads`, `num_layers` y `dtype` de un checkpoint concreto,
  más el peso de los propios pesos del modelo (no cubierto aquí — ver
  `DOCS/tech-debt-and-research-backlog.md` para cuantización).
- **`Qwen35Processor`**: no existe una entrada explícita en
  `VLMProcessorTypeRegistry`; se asume que los checkpoints Qwen3.5/3.5-MoE usan
  el mismo `processor_class` (`Qwen3VLProcessor`) que Qwen3VL vía
  `preprocessor_config.json`, pero esto no está verificado contra un checkpoint
  real descargado.
- **LFM2VL `minTiles`/`useThumbnail`**: campos de configuración parseados en
  `LFM2VLVisionConfiguration` que no aparecen consumidos en
  `LFM2VLProcessor.swift` — no se pudo determinar si están vestigiales o si se
  usan en otro punto no localizado en esta revisión.
