# Memoria y KV cache

## Estado actual

`KVCache` define offset, `ropeOffset`, `maxSize`, update, estado serializable, trim, mascara, copia, preparacion por batch y finalizacion (`Libraries/MLXLMCommon/KVCache.swift:46`). La API ya contempla caches trimmables, caches rotatorias, caches cuantizadas y metadatos de batch.

`QuantizedKVCacheProtocol` permite actualizar y leer estados cuantizados sin volver al camino regular (`Libraries/MLXLMCommon/KVCache.swift:145`). `GenerateParameters` expone `kvBits`, `kvGroupSize`, `quantizedKVStart` y `kvScheme`, lo que abre la puerta a esquemas no solo affine 4/8-bit.

El repo contiene `WiredMemoryPolicies.swift` y `WiredMemoryUtils.swift`, ademas de documentacion `Libraries/MLXLMCommon/Documentation.docc/wired-memory.md`. Esto indica una preocupacion explicita por coordinar cargas y generacion bajo limites reales de memoria.

`ModelContainer` usa acceso serializado y documenta la regla de que `MLXArray` no es `Sendable`. `ChatSession` protege su KV cache con `SerialAccessContainer` y evita re-prefill de historial tras la primera vuelta, pero se declara no thread-safe y mantiene estado publico mutable como `instructions`, `processing`, `generateParameters`, `additionalContext` y `tools`.

## Riesgos principales

1. KV cache como coste dominante en contexto largo.
   - En modelos pequenos, el peso Q4 puede ser manejable, pero el KV cache crece linealmente con tokens, capas, KV heads y head dim.
   - `maxKVSize` y `kvBits` deben ser parte de las recetas de produccion, no parametros avanzados escondidos.

2. Presupuesto de memoria por aplicacion.
   - En iOS/iPadOS no basta con "memoria fisica". Hay limites del proceso, estado termico, GPU working set y otras partes de la app.
   - La politica de speculative decoding ya estima main + draft + adicional, pero conviene unificar esa logica con wired memory y con recomendaciones de `GenerateParameters`.

3. Serializacion/copia de caches.
   - `KVCache.copy()` y `state/metaState` son poderosos para prompt caching, branching y sesiones.
   - Tambien son una zona de riesgo: copiar arrays grandes por accidente puede duplicar memoria justo cuando se intenta optimizar.

4. Errores recuperables convertidos en crash.
   - Varias rutas de cache usan `fatalError` ante estado invalido o combinaciones no soportadas.
   - Casos importantes: conversion cuantizada de cache rotatoria no implementada, uso generico de `QuantizedKVCache.update()` y metadatos corruptos.

5. Batch lengths y mascaras.
   - `prepare(lengths:)` y `finalize()` permiten batch variable, pero cualquier regression en mascara puede provocar resultados incorrectos o materializacion cara.

6. Retencion de buffers tras trim.
   - `KVCacheSimple.trim` baja el offset, pero no implica liberar inmediatamente el backing storage.
   - `RotatingKVCache` limita memoria, aunque durante prefill multitoken puede crecer temporalmente por encima de `maxCacheSize`.

7. Media input materialization.
   - `UserInput.Image.array.asCIImage()` puede materializar datos CPU.
   - Audio por URL acumula floats antes de construir `MLXArray`; para entradas largas puede ser un pico evitable.

8. Streaming detokenizer.
   - El detokenizer ingenuo decodifica segmentos acumulados por token para manejar Unicode parcial. Correcto, pero potencialmente O(n^2) en segmentos largos.

## Mejoras propuestas

1. `MemoryBudget` publico y reutilizable.
   - Entradas: modelo, cuantizacion, contexto objetivo, media tokens, draft model, app reserve, dispositivo.
   - Salidas: `GenerateParameters` recomendados, si speculative decoding cabe, advertencias y estimacion de bytes.

2. Telemetria de memoria en generacion.
   - Exponer eventos opcionales con cache tokens, KV bytes estimados, GPU working set recomendado/observado cuando MLX lo permita, trims y fallback decisions.

3. Prompt cache seguro.
   - Documentar y probar patrones de guardar/cargar cache para system prompt largo, RAG prefills y conversaciones ramificadas.
   - Anadir ejemplos que eviten duplicaciones accidentales y que llamen `.eval()` cuando corresponda.

4. Politicas adaptativas de cache.
   - Si el usuario selecciona contexto alto en un dispositivo limitado, recomendar `RotatingKVCache`.
   - Si la calidad lo permite, activar `kvBits: 4` desde cierto token.
   - Si hay VLM/media, reservar presupuesto antes de generar.

5. `ChatSession` con politica de memoria.
   - Exponer `wiredMemoryTicket` o una policy closure por turno.
   - Documentar/implementar fallback cuando se pide `maxKVSize` + `kvBits` en una ruta que no soporta cache rotatoria cuantizada.

6. APIs throwing para cache.
   - Sustituir `fatalError` por errores lanzables en restauracion de prompt cache, conversion de cache y combinaciones no soportadas.
   - Mantener preconditions solo para bugs internos imposibles, no para input externo o estado persistido.

## Investigacion propuesta

- Comparar `KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache`, caches hibridas y Mamba/SSM en memoria y calidad.
- Medir degradacion de calidad por `kvBits` y `quantizedKVStart` en tareas de long-context.
- Estudiar si `kvScheme` puede soportar esquemas WHT/low-rank/product quantization sin romper APIs.
- Crear stress tests de trim/copy/serialize en 8k/32k/128k tokens.
- Unificar wired memory, GPU working set y speculative memory policy bajo una sola capa de decision.
- Medir RSS/`Memory.activeMemory` para `KVCacheSimple.trim` y confirmar si bajar `offset` retiene buffers grandes.
- Probar `GenerateParameters(maxKVSize: ..., kvBits: 4)` en modelos reales para validar fallo/fallback esperado.
- Benchmark del `NaiveStreamingDetokenizer` con respuestas largas sin saltos de linea.
- Validar `GPU.maxRecommendedWorkingSetBytes()` vs `os_proc_available_memory()` vs jetsam observado en dispositivos iOS/iPadOS reales.
