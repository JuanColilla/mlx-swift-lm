# Compatibilidad, modelos y cuantizacion

## Estado actual

El paquete publica `MLXLLM`, `MLXVLM`, `MLXLMCommon`, `MLXEmbedders`, `MLXHuggingFace`, `BenchmarkHelpers` e `IntegrationTestHelpers` (`Package.swift`). La version de Swift requerida es 6.1 y el core depende de la revision exacta `5e27a4cb` de `JuanColilla/mlx-swift`, que anade cuantizacion affine de 1 bit sobre la base oficial actualizada y desactiva NAX en GPU gen-17 por correccion.

La compatibilidad LLM es amplia: `LLMTypeRegistry.shared` registra 58 `model_type`, con familias como Mistral, Mixtral, Llama, Phi/Phi3/PhiMoE, Gemma 1/2/3/3n/4, Qwen2/Qwen3/Qwen3.5/MoE/Next, MiniCPM, Starcoder2, Cohere, OpenELM, InternLM2, DeepSeek V3, Granite, GLM4, Falcon H1, Bitnet, SmolLM3, Ernie, LFM2, Baichuan, Exaone, GPT-OSS, Olmo, Nemotron, Jamba, Mamba2, Mistral3, Apertus y otros (`Libraries/MLXLLM/LLMModelFactory.swift:23`). Hay 54 ficheros Swift bajo `Libraries/MLXLLM/Models`.

La compatibilidad VLM cubre PaliGemma, Qwen2 VL, Qwen2.5 VL, Qwen3 VL, Qwen3.5, Idefics3, Gemma3, Gemma4, SmolVLM, FastVLM/LLaVA-Qwen2, Pixtral, Mistral3, LFM2-VL y GLM OCR (`Libraries/MLXVLM/VLMModelFactory.swift:86`). Hay 17 ficheros Swift bajo `Libraries/MLXVLM/Models`.

`MLXEmbedders` tambien debe entrar en la matriz de compatibilidad: BERT/RoBERTa/XLM-R/DistilBERT, NomicBERT, Qwen3, LFM2 bidireccional y Gemma3/Gemma3n aparecen en factories y tests de embeddings. Sus decisiones relevantes no son solo carga, sino pooling, normalizacion, truncado dimensional y heads tipo ColBERT.

El soporte de cuantizacion aparece en varios niveles:

- Modelos pre-cuantizados en registros `mlx-community/*-4bit`, `*-8bit`, `qat-4bit`.
- `QuantizedLinear` y modulos `Quantizable` a nivel MLXNN/modelo.
- `GenerateParameters.kvBits`, `kvGroupSize`, `quantizedKVStart` y `kvScheme`.
- `QuantizedKVCacheProtocol` con modo, bits y group size (`Libraries/MLXLMCommon/KVCache.swift:145`).
- Conversion de safetensors con cuantizacion global y por capa: `skip`, `bits`, `groupSize`, `mode`.
- Modos de conversion: `affine`, `mxfp4`, `mxfp8`, `nvfp4`; defaults observados: affine 4-bit group 64, MX/NV FP 4/8 group 32.
- El modo `affine` admite pesos de 1 bit con `scale + bias`; el port, las revisiones y la validacion estan documentados en [Compatibilidad de Bonsai 1-bit](bonsai-1bit-compatibility.md).
- ParoQuant como ruta especial para modelos/layouts concretos, con AutoAWQ, rotaciones y proyecciones Mamba fusionadas.
- Tests de paridad y comportamiento, por ejemplo `ParoQuantTests`, `Gemma4TextTests`, `EvalTests`.

## Brechas de compatibilidad

1. Matriz viva de modelos.
   - El registro dice que una arquitectura puede construirse, pero no documenta de forma centralizada que repos concretos funcionan, con que revision, cuantizacion, tokenizer, template, tool format y extra EOS.
   - Se recomienda generar una tabla automatica desde `LLMRegistry`, `VLMRegistry`, tests de integracion y fixtures.
   - El README y los READMEs de librerias van por detras del codigo; la fuente de verdad actual son los registros y tests.

2. Thinking mode y tool calling.
   - Qwen/DeepSeek/GLM/LFM tienen formatos diferentes de tool calls y plantillas.
   - La compatibilidad deberia etiquetarse por capacidades: `supportsThinkingToggle`, `supportsTools`, `toolCallFormat`, `requiresExtraEOS`, `supportsSystemPrompt`, `supportsMultiImage`, `supportsVideo`.

3. Modelos MoE y recurrentes.
   - MoE, Mamba/SSM y modelos hibridos tienen costes y caches distintos de un transformer denso.
   - Hace falta documentar mejores practicas por familia: expertos activos, memoria de router, cache recurrente, compatibilidad con KV quantization, speculative decoding y LoRA.

4. VLM media budgets.
   - VLM no solo consume tokens de texto; imagen, tiles, video frames y resolucion dominan prefill y memoria.
   - Crear una API/documentacion de presupuesto: imagenes maximas, resolucion recomendada, tokens visuales aproximados y degradacion progresiva.

5. Formatos de pesos no cubiertos.
   - El camino fuerte es safetensors. PyTorch `.bin`, GGUF y requantizacion de modelos ya cuantizados no son objetivo actual.
   - Antes de ampliar formatos, el retorno mas alto parece mejorar errores de incompatibilidad y reportes de conversion.

## Estructura de modelos recomendada

Para nuevos ports, mantener esta disciplina:

- `Configuration: Codable, Sendable` con defaults explicitos y validacion via `ModelConfigurationValidating` cuando haya invariantes.
- Modelo top-level conformando a `LanguageModel` y, si aplica, `KVCacheDimensionProvider`.
- `newCache(parameters:)` especifico por arquitectura, no generico, para soportar sliding windows, Mamba/SSM o caches hibridas.
- Uso consistente de `makeAttentionMask` y `MLXFast.scaledDotProductAttention`.
- `sanitize(weights:)` pequeno, probado y documentado cuando haya conversiones de nombres, dequant temporal o formatos especiales.
- Tests con pesos sinteticos para contratos y tests de integracion descargables para paridad real.
- Para VLM, portar procesador y media budget junto con el modelo; una arquitectura registrada sin processor robusto no es compatibilidad real.

## Investigacion propuesta

- Generador de matriz de compatibilidad desde source + tests + Hugging Face metadata.
- Clasificador de modelos por capacidad: chat, tools, thinking, VLM, video, embeddings, LoRA, KV quantization, speculative draft/target.
- Recetas oficiales por cuantizacion: Q4 default, Q8 calidad, bf16 desarrollo/paridad, QAT 4-bit para Gemma, formatos MXFP4/FP4 si aparecen en repos MLX.
- Pipeline de porting asistido desde `mlx-lm` Python: detectar config, pesos, layers, nombres, caches y generar checklist/test skeleton.
- Tests de carga smoke para modelos populares pequenos por familia, con etiqueta "offline/unit" vs "integration/download".
- Tests parametrizados que al menos decodifiquen `config.json` reales por cada arquitectura registrada, sin descargar pesos completos.
