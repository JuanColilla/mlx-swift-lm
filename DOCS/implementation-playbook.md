# Playbook de implementacion

## Perfil base recomendado

Para una app on-device con chat LLM:

```swift
let parameters = GenerateParameters(
    maxTokens: 512,
    maxKVSize: 4096,
    kvBits: 4,
    kvGroupSize: 64,
    quantizedKVStart: 0,
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    minP: 0.0,
    prefillStepSize: 512
)
```

No tratar estos valores como universales. Son un punto de partida para medir. En apps reales, derivarlos de modelo, dispositivo, longitud de conversacion, media adjunta y objetivo de calidad.

Para rutas que usen MTP, no asumir que el perfil anterior aplica sin cambios.
`ChatSession` permite activarlo con `MTPSpeculativeDecodingConfig`, pero la
verificación MTP no consume shared K/V cuantizado: si la cuantización entra en
vigor, el iterador pasa de forma sticky a generación target-only y lo explica en
la telemetría. Medir por separado MTP con KV normal y el perfil cuantizado.

## Seleccion de modelo

- iPhone o iPad base: empezar por 0.5B-3B Q4/QAT 4-bit, contexto moderado y `kvBits: 4`.
- iPad Pro/Mac con margen: 4B-8B Q4 para chat general; Q8/bf16 solo cuando calidad/paridad justifique memoria.
- Razonamiento: preferir Qwen3/Gemma/DeepSeek distilled pequenos con thinking controlado por template. Medir coste de tokens de pensamiento.
- Herramientas: seleccionar familias con `toolCallFormat` probado; no asumir que JSON tools funcionan igual en GLM/Qwen/LFM.
- Embeddings: elegir pooling y normalizacion segun tarea. CLS/mean/max/last/none, truncado dimensional y ColBERT cambian tanto calidad como memoria.
- VLM: presupuestar imagen/video antes del modelo. Un VLM pequeno con tiles bien elegidos puede superar a uno grande mal presupuestado.

## Gestion de conversacion

- Mantener system prompt corto o cachearlo.
- Resumir historia antes de forzar contexto largo.
- Usar `maxKVSize` para conversaciones largas en dispositivos limitados.
- Medir impacto de `kvBits` en tareas que requieren citas exactas o memoria larga.
- Separar "thinking" de respuesta final si la plantilla/modelo lo soporta, y limitar tokens de pensamiento en UI.

## Rendimiento

- Medir por fases: carga, tokenizer, prefill, primer token, decode sostenido, detokenizacion y UI streaming.
- Probar `prefillStepSize` por familia. No asumir que 512 es optimo.
- Usar speculative decoding solo cuando el draft tenga buena acceptance rate y quepa en memoria.
- Para MTP, empezar con `blockSize` 4 y medir; subir bloque sin acceptance rate suficiente puede empeorar rendimiento.
- Desactivar filtros caros de sampling cuando no aporten valor: `temperature: 0` para determinismo, o top-p/top-k/min-p solo si se necesita diversidad.
- Evitar generar en background en apps Apple; pausar y reanudar alrededor del ciclo de vida.

## Memoria

- Reservar memoria de app antes de cargar modelo y cache.
- Estimar KV cache con longitud objetivo, no con longitud inicial.
- Preferir `kvBits: 4` y cache rotatoria cuando el valor de contexto completo sea bajo.
- No asumir que `trim` libera memoria residente; medir y, si hace falta, recrear cache/sesion.
- Si se usa `ChatSession`, tratarlo como no thread-safe salvo la proteccion interna de KV cache.
- Liberar modelos/caches de forma explicita en cambios de tarea si la app alterna LLM, VLM y embeddings.
- Excluir pesos descargados de backups de iCloud en apps iOS.

## Compatibilidad

- Pinnear modelos por revision para productos.
- Guardar junto al modelo: `extraEOSTokens`, template esperado, tool format, thinking toggle y limites media.
- Para thinking mode, preferir metadata/plantilla explicita por modelo. No confiar en que todos los modelos interpreten `additionalContext` igual.
- Ejecutar smoke tests con prompt fijo tras actualizar `mlx-swift`, `mlx-swift-lm`, tokenizer o pesos.
- Separar tests unitarios sin pesos de tests de integracion con descarga.
