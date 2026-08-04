# Rendimiento y generacion

## Estado actual

`GenerateParameters` ya expone los mandos principales de rendimiento: `prefillStepSize`, `maxTokens`, `maxKVSize`, `kvBits`, `kvGroupSize`, `quantizedKVStart`, `kvScheme`, `temperature`, `topP`, `topK`, `minP`, penalizaciones y `seed` determinista (`Libraries/MLXLMCommon/Evaluate.swift:54`). Esto permite construir perfiles de inferencia sin cambiar la API base.

El repo tambien contiene una implementacion de speculative decoding con telemetria (`SpeculativeDecodingTelemetry`) y politica de memoria (`SpeculativeDecodingMemoryPolicy`) basada en `GPU.maxRecommendedWorkingSetBytes()` (`Libraries/MLXLMCommon/SpeculativeDecoding.swift`). Hay tests especificos de speculative decoding y MTP en `Tests/MLXLMTests/SpeculativeDecodingTests.swift`, `MTP*Tests.swift` e integracion en `IntegrationTesting/IntegrationTestingTests/MTP*`.

El `TokenIterator` separa prefill y decode, y la informacion final de generacion expone campos para speculative/MTP como `proposedDraftTokens`, `acceptedDraftTokens`, `passthroughReason` y `speculativeDecodingTelemetry`. `BenchmarkHelpers` ofrece benchmark base con warm-up, mediana y `temperature: 0`, microbenchmarks de sampling, un schema JSON generico y un adaptador MTP. Los resultados de speculative clasico, checkpoint y dispositivo siguen perteneciendo a los tests de integracion que los ejecuten.

MTP merece seguimiento propio. `MTPSpeculativeTokenIterator` propone bloques
de tokens mediante un drafter asociado al target y `ChatSession` lo expone con
`MTPSpeculativeDecodingConfig`, incluida admisión de memoria, carga eager o
deferred y telemetría. La verificación MTP con shared K/V cuantizado no está
implementada: cuando la cuantización entra en vigor, el iterador pasa de forma
sticky a target-only. Los tests prueban el contrato, no una mejora universal de
rendimiento; hace falta medir cada target/drafter/dispositivo.

El cambio 3.x documentado en `README.md` es importante para rendimiento real: el core queda desacoplado de tokenizer/downloader, y la ruta comoda se mueve a `MLXHuggingFace` macros. Eso permite comparar implementaciones de tokenizacion/descarga sin contaminar la generacion.

## Oportunidades de mejora

1. Benchmark harness publico y estable.
   - Medir TTFT, tokens/s sostenidos, tokens/s por fase, memoria pico, GPU working set, prefill throughput, decode throughput, acceptance rate y energia/termica cuando el host lo permita.
   - Reutilizar `BenchmarkHelpers` y `IntegrationTesting`, pero publicar una matriz facil de ejecutar por modelo/dispositivo.
   - Guardar resultados como JSON para comparar PRs.

2. Tuning adaptativo de `prefillStepSize`.
   - El default actual es 512. Conviene investigar perfiles por arquitectura: modelos con estado recurrente, MoE, VLM con muchas imagenes y modelos de contexto largo pueden necesitar otros pasos.
   - Proponer una API de recomendacion, por ejemplo `GenerationProfile.recommendedPrefillStepSize(model:device:promptTokens:)`, dejando el valor manual como escape hatch.

3. Calibrar speculative decoding adaptativo con datos reales.
   - `AdaptiveSpeculativeDecodingPolicy` ya permite abandonar speculative de forma opt-in y sticky cuando una ventana medida cae por debajo del umbral configurado.
   - Falta publicar perfiles medidos por par target/draft y dispositivo; la API no inventa un umbral universal.
   - Investigar pares por familia y, como trabajo posterior, si compensa ajustar también la longitud de draft en runtime.

4. Validar MTP de alto nivel en modelos y dispositivos reales.
   - `ChatSession` y `MTPSpeculativeDecodingConfig` ya cubren `blockSize`, política de memoria, fallback target-only y telemetría.
   - Los fallos de configuración y compatibilidad recuperables usan errores; los fallbacks no dependen de `print` directo.
   - Falta ejecutar la matriz de benchmark con pesos reales para elegir target/drafter, bloque y presupuesto por producto.

5. Separar coste de sampling del coste del modelo.
   - `topP`, `topK` y `minP` aplican filtros sobre vocabulario completo. En vocabularios grandes, el coste no es gratis.
   - Crear benchmarks de sampling con logits sinteticos y medir argmax vs categorical vs top-p/min-p/top-k.

6. Reducir materializacion de mascaras.
   - `makeAttentionMask` devuelve `.causal` cuando basta una mascara simbolica y materializa arrays solo cuando hace falta.
   - Mantener esta regla como invariant de rendimiento y anadir tests de regresion para ventanas, cache rotatoria, long prompt y batch lengths.

## Tech debt

- La medicion de rendimiento esta dispersa entre tests, `BenchmarkHelpers` e integracion. Hay primitives, pero falta una historia de "run this and compare".
- El fallback adaptativo ya cierra el loop con acceptance rate en runtime, pero su calibración sigue perteneciendo al producto y requiere benchmarks reales.
- `GenerateCompletionInfo.summary()` expone datos speculative/MTP y MTP tiene adaptador JSON; falta decidir qué subconjunto conviene mostrar en cada UI o informe de producto y estandarizar el adaptador de speculative clasico.
- MTP está integrado en `ChatSession`, pero no verifica con shared K/V cuantizado y pasa a target-only cuando aparece esa condición.
- El playbook y las recetas documentan los controles, pero no publican valores universales para `prefillStepSize`, `maxKVSize`, `kvBits` o draft model porque dependen del modelo y dispositivo.
- Conviene evitar que futuras optimizaciones dependan solo de tests unitarios con pesos sinteticos; esos tests son buenos para contratos, pero no capturan TTFT ni memoria real.

## Investigacion propuesta

- Matriz M-series/iPhone/iPad: TTFT, tokens/s, memoria pico y termica para Q4/Q8/bf16, con y sin `kvBits`.
- Curvas `prefillStepSize`: 128/256/512/1024/2048 por arquitectura y longitud de prompt.
- Curvas speculative decoding: acceptance rate vs speedup real por par target/draft.
- Sampling microbenchmarks: coste de `topP`, `topK`, `minP`, penalizaciones y seed.
- Long-context soak tests: 8k/16k/32k/64k tokens con `maxKVSize`, `RotatingKVCache` y `QuantizedKVCache`.
- Matriz `baseline vs speculative clasico vs MTP`: prompts corto/2k/8k, `maxTokens` 64/256, `temperature` 0 y 0.7, `blockSize` 2/4/6/8, KV normal vs `kvBits: 4`.
