# Tech debt e investigacion

## Prioridad alta

1. Benchmark suite reproducible.
   - Output JSON, comparacion entre commits y presets por modelo.
   - Metricas: TTFT, tokens/s, memoria pico, acceptance rate, prefill/decode, sampling overhead.
   - Rutas relacionadas: `Libraries/BenchmarkHelpers`, `IntegrationTesting`, `Tests/MLXLMTests/SpeculativeDecodingTests.swift`.

2. Matriz de compatibilidad generada.
   - Derivar desde `LLMTypeRegistry`, `VLMTypeRegistry`, registros, tests y fixtures.
   - Campos: arquitectura, modelo recomendado, cuantizacion, tokenizer, template, EOS, tools, thinking, VLM/video, KV quantization, speculative.

3. Politica publica de memoria.
   - Unificar `GenerateParameters`, wired memory, KV estimate y speculative memory policy.
   - Producir recomendaciones explicables, no solo errores OOM.
   - Incluir `ChatSession`, que hoy no expone wired-memory ticket por turno.

4. Recetas oficiales de long-context.
   - Cubrir trim, cache rotatoria, cache cuantizada, prompt cache y resumen de conversacion.
   - Incluir tests de regresion de mascaras y calidad.

5. Reducir `fatalError` en superficies recuperables.
   - Prompt cache corrupto, `metaState` invalido, conversion KV no soportada y combinaciones `maxKVSize + kvBits` deberian poder fallar con errores utiles.

## Prioridad media

6. Runtime adaptive speculative decoding.
   - Usar `SpeculativeDecodingTelemetry` para ajustar o abandonar speculative en caliente.
   - Investigar pares target/draft mantenidos oficialmente.

7. MTP de produccion.
   - Integrar en `ChatSession`, anadir config publica, benchmark oficial y soporte/fallback claro para KV cuantizado.
   - Sustituir prints directos por logging o eventos.

8. Perfilador de sampling.
   - Microbenchmarks para argmax, categorical, top-p, top-k, min-p y penalizaciones.

9. Guia de VLM media budgets.
   - Tabla por arquitectura: resolucion, tiles, frames, tokens visuales, memoria aproximada.

10. Tool/thinking compatibility layer.
   - Metadata por modelo con formato de tools, thinking mode, tags, stop tokens y restricciones.

## Prioridad baja pero estrategica

11. Pipeline de porting desde `mlx-lm`.
   - Scaffold de configuracion, modelo, registry, tests y mapping de pesos desde Python MLX.

12. Investigacion de `kvScheme`.
   - Affine4/8 ya son base. Explorar WHT, product quantization o compresion hibrida si MLX Swift lo permite eficientemente.

13. Cache persistence API.
   - Prompt caching seguro para apps RAG y asistentes con system prompts largos.

14. Conformance dashboards.
   - Reporte en CI/manual de modelos que cargan, generan, usan tools, VLM y long-context por plataforma.

15. Detokenizer y media materialization.
   - Medir coste O(n^2) de streaming detokenization y picos CPU de imagen/audio.

## Preguntas abiertas

- Que modelos pequenos maximizan acceptance rate como draft para cada familia target?
- Cual es el umbral real de `prefillStepSize` antes de perder rendimiento por memoria/materializacion?
- En que tareas `kvBits: 4` degrada calidad de forma visible y donde es practicamente gratis?
- Como debe exponerse el thinking mode: metadata de modelo, parametro de template, o API de alto nivel?
- Que parte del coste de VLM viene de procesamiento media vs transformer prefill?
- Debe `ChatSession` seguir siendo mutable/no thread-safe o moverse hacia configuraciones inmutables por turno?
