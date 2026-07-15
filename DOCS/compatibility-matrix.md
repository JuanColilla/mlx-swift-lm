# Matriz de compatibilidad (generada)

Backlog: `DOCS/tech-debt-and-research-backlog.md` #2 — "Matriz de compatibilidad
generada. Derivar desde `LLMTypeRegistry`, `VLMTypeRegistry`, registros, tests
y fixtures."

Este documento **no está escrito a mano**: es la salida literal de
`Tests/MLXLMTests/CompatibilityMatrixGeneratorTests.swift`, que lee
`LLMTypeRegistry.shared`, `VLMTypeRegistry.shared`,
`EmbedderTypeRegistry.shared` (arquitecturas que el código puede instanciar)
y `LLMRegistry.shared`/`VLMRegistry.shared`/`EmbedderRegistry.shared`
(modelos concretos de Hugging Face preconfigurados) directamente en tiempo de
ejecución. Regenerar con:

```
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:MLXLMTests/CompatibilityMatrixGeneratorTests
```

y sustituir las listas de abajo por el `stdout` del test. Generado en esta
rama el 2026-07-15 contra el commit tras el merge de `origin/main`
(`f97da4e` + los commits de esta rama).

## Cobertura actual

| Registro | Arquitecturas (`model_type`) | Modelos recomendados concretos |
|---|---|---|
| LLM | 58 | 55 |
| VLM | 18 | 17 |
| Embedder | 10 | 21 |

Nota: el número de arquitecturas VLM (18) no coincide 1:1 con familias
distintas — `lfm2-vl`/`lfm2_vl` son la misma familia registrada dos veces
(con y sin guion), y varias entradas comparten clase de modelo Swift (p.ej.
`gemma4`/`gemma4_unified` son dos `model_type` para variantes del mismo
modelo). Ver `DOCS/vlm-media-budgets.md` para el desglose real por familia
(16 arquitecturas distintas).

## Qué SÍ cubre esta matriz (mecánicamente verificable)

- Qué `model_type` string puede instanciar cada registro (`registeredModelTypes`,
  añadido en `Libraries/MLXLMCommon/Registries/ModelTypeRegistry.swift`).
- Qué modelos concretos de Hugging Face tienen una `ModelConfiguration`
  preconfigurada en este repo (`LLMRegistry`/`VLMRegistry`/`EmbedderRegistry`).
- Para LLM: `toolCallFormat` explícito (cuando el registro lo fija a mano —
  ver `ToolCallFormat.infer(from:configData:)` en
  `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift` para los casos donde se
  infiere en tiempo de carga en vez de declararse) y `extraEOSTokens`
  explícitos.

## Qué NO cubre (gaps conocidos, ver `DOCS/compatibility-and-models.md`)

- **Thinking mode**: no existe ningún campo estructurado equivalente a
  `toolCallFormat` para "este modelo soporta thinking toggle" — se maneja
  puramente vía chat template / `additionalContext`, sin metadata tipada.
  Confirmado por `grep -rln "thinking\|enableThinking" Libraries/` sin
  resultados relevantes.
- **Cuantización, tokenizer, template, VLM/video budgets**: no están en esta
  tabla porque no hay un campo por-modelo que los capture de forma
  homogénea hoy; añadir esas columnas requeriría antes añadir los campos a
  `ModelConfiguration` (o derivarlos descargando `config.json` de cada repo,
  que esta matriz deliberadamente no hace por no requerir red).
- Modelos que "cargan y generan realmente" (conformance real, no solo
  configuración declarada) — eso es responsabilidad de
  `DOCS/conformance-dashboard.md`, no de este documento.

## LLM — `model_type` registrados (58)

acereason, afmoe, apertus, baichuan_m1, bailing_moe, bitnet, cohere,
deepseek_v3, ernie4_5, exaone4, falcon_h1, gemma, gemma2, gemma3,
gemma3_text, gemma3n, gemma4, gemma4_text, gemma4_unified, glm4, glm4_moe,
glm4_moe_lite, gpt_oss, granite, granitemoehybrid, internlm2, jamba, lfm2,
lfm2_moe, lille-130m, llama, mamba2, mimo, mimo_v2_flash, minicpm, minimax,
mistral, mistral3, mixtral, nanochat, nemotron_h, nemotron_labs_diffusion,
olmo2, olmo3, olmoe, openelm, phi, phi3, phimoe, qwen2, qwen3, qwen3_5,
qwen3_5_moe, qwen3_5_text, qwen3_moe, qwen3_next, smollm3, starcoder2

## VLM — `model_type` registrados (18)

fastvlm, gemma3, gemma4, gemma4_unified, glm_ocr, idefics3, lfm2-vl,
lfm2_vl, llava_qwen2, mistral3, paligemma, pixtral, qwen2_5_vl, qwen2_vl,
qwen3_5, qwen3_5_moe, qwen3_vl, smolvlm

## Embedder — `model_type` registrados (10)

bert, distilbert, gemma3, gemma3_text, gemma3n, lfm2, nomic_bert, qwen3,
roberta, xlm-roberta

## LLM — modelos recomendados con `toolCallFormat`/`extraEOSTokens` declarados

Solo se listan las entradas con al menos uno de los dos campos fijado
explícitamente en el registro (el resto usa inferencia en tiempo de carga
vía `ToolCallFormat.infer` o no declara EOS extra):

| Modelo | toolCallFormat | extraEOSTokens |
|---|---|---|
| mlx-community/GLM-4-9B-0414-4bit | glm4 | - |
| mlx-community/LFM2-1.2B-4bit | lfm2 | - |
| mlx-community/LFM2-8B-A1B-3bit-MLX | lfm2 | - |
| mlx-community/Llama-3.2-1B-Instruct-4bit | - | `<\|eot_id\|>` |
| mlx-community/Llama-3.2-3B-Instruct-4bit | - | `<\|eot_id\|>` |
| mlx-community/Meta-Llama-3-8B-Instruct-4bit | - | `<\|eot_id\|>` |
| mlx-community/Meta-Llama-3.1-8B-Instruct-4bit | - | `<\|eot_id\|>` |
| mlx-community/Phi-3.5-MoE-instruct-4bit | - | `<\|end\|>` |
| mlx-community/Phi-3.5-mini-instruct-4bit | - | `<\|end\|>` |
| mlx-community/Qwen1.5-0.5B-Chat-4bit | - | `<\|im_end\|>` |
| mlx-community/Qwen2.5-1.5B/7B-Instruct-4bit | - | `<\|im_end\|>` |
| mlx-community/Qwen3-*-4bit (0.6B/1.7B/4B/8B/30B-A3B) | - | `<\|im_end\|>` |
| mlx-community/Qwen3.5-2B-4bit / Qwen3.6-27B-4bit | - | `<\|im_end\|>` |
| mlx-community/gemma-3-1b-it-qat-4bit | - | `<end_of_turn>` |
| mlx-community/gemma-3n-E2B/E4B-it-lm-* | - | `<end_of_turn>` |
| mlx-community/gemma-4-e2b/e4b-it-4bit | - | `<turn\|>` |

El resto de las 55 entradas LLM recomendadas no fija ninguno de los dos
campos explícitamente (dejando `toolCallFormat` a inferencia por
`model_type` en tiempo de carga, y sin EOS extra más allá del `eos_token`
del tokenizer).

## VLM — modelos recomendados (17)

HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx,
lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit,
mlx-community/FastVLM-0.5B-bf16, mlx-community/Qwen2-VL-2B-Instruct-4bit,
mlx-community/Qwen2.5-VL-3B-Instruct-4bit,
mlx-community/Qwen3-VL-4B-Instruct-8bit, mlx-community/Qwen3.5-27B-4bit,
mlx-community/Qwen3.5-35B-A3B-4bit, mlx-community/SmolVLM-Instruct-4bit,
mlx-community/gemma-3-{4b,12b,27b}-it-qat-4bit,
mlx-community/gemma-4-{e2b,e4b}-it-4bit, mlx-community/gemma-4-26b-a4b-it-4bit,
mlx-community/gemma-4-31b-it-4bit, mlx-community/paligemma-3b-mix-448-8bit

## Embedder — modelos recomendados (21)

BAAI/bge-{base,large,small}-en-v1.5, BAAI/bge-m3,
Snowflake/snowflake-arctic-embed-{l,xs}, TaylorAI/bge-micro-v2,
TaylorAI/gte-tiny, intfloat/multilingual-e5-small,
mixedbread-ai/mxbai-embed-large-v1, mlx-community/LFM2.5-ColBERT-350M-*
(4bit/8bit/bf16), mlx-community/LFM2.5-Embedding-350M-*
(4bit/8bit/bf16), mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ,
nomic-ai/nomic-embed-text-v1{,.5}, sentence-transformers/all-MiniLM-{L6,L12}-v2
