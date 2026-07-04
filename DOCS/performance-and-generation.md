# Rendimiento y generacion

## Estado actual

`GenerateParameters` ya expone los mandos principales de rendimiento: `prefillStepSize`, `maxTokens`, `maxKVSize`, `kvBits`, `kvGroupSize`, `quantizedKVStart`, `kvScheme`, `temperature`, `topP`, `topK`, `minP`, penalizaciones y `seed` determinista (`Libraries/MLXLMCommon/Evaluate.swift:54`). Esto permite construir perfiles de inferencia sin cambiar la API base.

El repo tambien contiene una implementacion de speculative decoding con telemetria (`SpeculativeDecodingTelemetry`) y politica de memoria (`SpeculativeDecodingMemoryPolicy`) basada en `GPU.maxRecommendedWorkingSetBytes()` (`Libraries/MLXLMCommon/SpeculativeDecoding.swift`). Hay tests especificos de speculative decoding y MTP en `Tests/MLXLMTests/SpeculativeDecodingTests.swift`, `MTP*Tests.swift` e integracion en `IntegrationTesting/IntegrationTestingTests/MTP*`.

El `TokenIterator` separa prefill y decode, y la informacion final de generacion expone campos para speculative/MTP como `proposedDraftTokens`, `acceptedDraftTokens`, `passthroughReason` y `speculativeDecodingTelemetry`. `BenchmarkHelpers` ya ofrece benchmark base con warm-up, mediana y `temperature: 0`, pero no hay equivalente oficial para speculative clasico ni MTP.

MTP merece seguimiento propio. `MTPSpeculativeTokenIterator` propone bloques de tokens mediante un drafter asociado al target; los tests de integracion muestran mejoras reales con `blockSize` 4 frente a 6 en escenarios medidos. La brecha principal: MTP con KV cuantizado no esta implementado y entra en passthrough sticky, por lo que `GenerateParameters(maxKVSize: ..., kvBits: 4)` puede anular una de las optimizaciones mas interesantes para dispositivos limitados.

El cambio 3.x documentado en `README.md` es importante para rendimiento real: el core queda desacoplado de tokenizer/downloader, y la ruta comoda se mueve a `MLXHuggingFace` macros. Eso permite comparar implementaciones de tokenizacion/descarga sin contaminar la generacion.

## Oportunidades de mejora

1. Benchmark harness publico y estable.
   - Medir TTFT, tokens/s sostenidos, tokens/s por fase, memoria pico, GPU working set, prefill throughput, decode throughput, acceptance rate y energia/termica cuando el host lo permita.
   - Reutilizar `BenchmarkHelpers` y `IntegrationTesting`, pero publicar una matriz facil de ejecutar por modelo/dispositivo.
   - Guardar resultados como JSON para comparar PRs.

2. Tuning adaptativo de `prefillStepSize`.
   - El default actual es 512. Conviene investigar perfiles por arquitectura: modelos con estado recurrente, MoE, VLM con muchas imagenes y modelos de contexto largo pueden necesitar otros pasos.
   - Proponer una API de recomendacion, por ejemplo `GenerationProfile.recommendedPrefillStepSize(model:device:promptTokens:)`, dejando el valor manual como escape hatch.

3. Speculative decoding gobernado por datos.
   - Ya existe telemetria de rondas, draft tokens, aceptados, llamadas al target y ratio de aceptacion.
   - Falta convertir esa telemetria en decision adaptativa: parar speculative si `acceptanceRate` cae por debajo de un umbral durante N rondas, ajustar longitud de draft, o volver al modo normal si el draft no amortiza memoria/latencia.
   - Investigar pares target/draft por familia: Gemma4/Gemma4 draft, Qwen3/Qwen pequeno, Llama/Llama pequeno, y no solo tamanos genericos.

4. MTP como producto de rendimiento, no solo test de integracion.
   - Integrar MTP en `ChatSession`, o documentar claramente que solo vive en APIs lower-level.
   - Exponer `MTPSpeculativeDecodingConfig` con `blockSize`, politica de memoria, fallback y telemetria.
   - Eliminar `print` directo en passthrough y reemplazarlo por logging configurable o eventos.
   - Convertir `fatalError` de compatibilidad target/drafter en preflight throwing.

5. Separar coste de sampling del coste del modelo.
   - `topP`, `topK` y `minP` aplican filtros sobre vocabulario completo. En vocabularios grandes, el coste no es gratis.
   - Crear benchmarks de sampling con logits sinteticos y medir argmax vs categorical vs top-p/min-p/top-k.

6. Reducir materializacion de mascaras.
   - `makeAttentionMask` devuelve `.causal` cuando basta una mascara simbolica y materializa arrays solo cuando hace falta.
   - Mantener esta regla como invariant de rendimiento y anadir tests de regresion para ventanas, cache rotatoria, long prompt y batch lengths.

## Tech debt

- La medicion de rendimiento esta dispersa entre tests, `BenchmarkHelpers` e integracion. Hay primitives, pero falta una historia de "run this and compare".
- Las politicas de speculative decoding tienen gating de memoria, pero no parecen cerrar el loop con calidad/acceptance en runtime.
- `GenerateCompletionInfo.summary()` todavia se centra en prompt/generation TPS; deberia resumir acceptance rate, emitted/target call y passthrough.
- MTP no cubre KV quantization y no esta en `ChatSession`.
- La documentacion de uso dice como cargar y chatear, pero no da una receta oficial para elegir `prefillStepSize`, `maxKVSize`, `kvBits` o draft model.
- Conviene evitar que futuras optimizaciones dependan solo de tests unitarios con pesos sinteticos; esos tests son buenos para contratos, pero no capturan TTFT ni memoria real.

## Investigacion propuesta

- Matriz M-series/iPhone/iPad: TTFT, tokens/s, memoria pico y termica para Q4/Q8/bf16, con y sin `kvBits`.
- Curvas `prefillStepSize`: 128/256/512/1024/2048 por arquitectura y longitud de prompt.
- Curvas speculative decoding: acceptance rate vs speedup real por par target/draft.
- Sampling microbenchmarks: coste de `topP`, `topK`, `minP`, penalizaciones y seed.
- Long-context soak tests: 8k/16k/32k/64k tokens con `maxKVSize`, `RotatingKVCache` y `QuantizedKVCache`.
- Matriz `baseline vs speculative clasico vs MTP`: prompts corto/2k/8k, `maxTokens` 64/256, `temperature` 0 y 0.7, `blockSize` 2/4/6/8, KV normal vs `kvBits: 4`.
