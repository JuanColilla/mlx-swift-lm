# Investigación: `Gemma4ChunkedPrefillTests.chunkSizeInvariance` falla

## Estado

**Root cause identificada con alta confianza, no arreglada todavía.** Esto es
investigación (Fase 1-2 de systematic-debugging), no un fix. Tocar
`RotatingKVCache` a ciegas arriesga romper generación en producción para todos
los modelos con sliding window (Gemma3, Gemma3n, Gemma4, GLM4-MoE...), así que
el fix necesita su propia sesión con verificación end-to-end dedicada (test
sintético + al menos un modelo real vía `IntegrationTesting`).

## Reproducción

```
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation
```

Falla determinísticamente (100% reproducible, no flaky) para las 4
combinaciones parametrizadas de `chunkSizeInvariance` (chunkSize 3, 5, 8, 16),
con `Expectation failed: close` en `Gemma4ChunkedPrefillTests.swift:100`.
Confirmado presente en el baseline recién mergeado desde `origin/main` (commit
`f97da4e`), **antes** de cualquier cambio de esta rama — no es una regresión
introducida aquí. El commit `65be34c` ("Qwen3.5/3.6: windowed prefill...") que
llegó en ese merge solo tocó la firma de `Gemma4.prepare` (parámetro `state`
añadido y ambos lados de la llamada actualizados mecánicamente) — no tocó
lógica de máscara ni de cache, así que tampoco es la causa.

## Mecanismo sospechoso

El test compara logits de un prefill "chunked" (`windowSize` pequeño, p.ej. 3)
contra un prefill "single-pass" (`windowSize: 1024`) para el mismo prompt de
37 tokens, con capas sliding-window respaldadas por `RotatingKVCache(maxSize:
config.slidingWindow, keep: 0)` (`Gemma4.swift:1398`, `slidingWindow: 8` en el
config sintético del test).

El bucle de prefill (`gemma4PrepareTextOnly`, `Gemma4.swift:262-286`) siempre
dedica una llamada final de **un solo token** al último elemento del prompt
(`totalPositions - processed > 1` como condición del bucle, dejando
exactamente 1 posición fuera). Esto significa que, para `windowSize` pequeño,
`RotatingKVCache.update(keys:values:)` recibe **varias llamadas multi-token
(`updateConcat`, n>1)** seguidas de **una llamada final de un solo token
(`updateInPlace`, n==1)** — dos representaciones físicas distintas dentro del
mismo cache:

- `updateConcat` (`KVCache.swift:571-593`) mantiene los keys/values en **orden
  temporal ascendente**, recortando (`trim`, `KVCache.swift:539-553`) y
  concatenando: nunca reordena in-place, siempre crece por la derecha.
- `updateInPlace` (`KVCache.swift:595-651`) es un **buffer circular real**:
  escribe en la posición `idx`, que rota a `keep` (0) al llegar a
  `maxCacheSize`, y depende de que `idx` refleje fielmente el punto de
  escritura para que el rolled-mask de `makeMask` (caso `n == 1`,
  `KVCache.swift:744-766`) sea coherente con el layout físico.

La transición de `updateConcat` → `updateInPlace` (que ocurre exactamente en
la última llamada de cada prefill "chunked") pasa por una rama de
`updateInPlace` (`KVCache.swift:604-622`, comprobación
`self.keys!.dim(2) < maxCacheSize`) que **no está pensada** para heredar un
array más grande que `maxCacheSize` en orden temporal puro (producto de
`updateConcat`) — asume que el cache ya viene en modo circular por debajo de
capacidad, o vacío. Verifiqué a mano (trazado aritmético de offsets/índices
para `chunkSize=16`, prompt de 37 tokens) que el **contenido** final (qué
posiciones absolutas sobreviven) parece correcto tras la transición, pero
**no descarté** con evidencia instrumentada que el `idx`/orden físico quede
en un estado que el rolled-mask de pasos posteriores de generación
interprete mal, ni que la combinación con la rama `n == 1` de `makeMask`
(`KVCache.swift:745-751`, que devuelve `.none` — sin máscara — cuando
`maxCacheSize == windowSize`, que es siempre el caso en Gemma4 porque el
cache se construye con `maxSize: config.slidingWindow`) sea válida
exactamente en el primer paso tras la transición.

**No until confirmado con prints/instrumentación real** — esto es una
hipótesis fuerte (aritmética verificada a mano), no una causa raíz probada
por observación directa. Siguiente paso correcto: instrumentar
`RotatingKVCache.update`/`makeMask` con logging temporal (offset, idx,
`self.keys?.dim(2)` antes/después) y comparar chunked vs. single-pass paso a
paso hasta el primer punto de divergencia numérica real, en vez de seguir
razonando en papel.

## Por qué no se arregla en esta sesión

- `RotatingKVCache` es compartida por todos los modelos sliding-window del
  repo (no solo Gemma4) — un fix mal verificado aquí tiene radio de impacto
  amplio.
- El caso que dispara el bug (prefill explícitamente *chunked* con
  `windowSize` pequeño Y sliding-window activo Y prompt que no es múltiplo
  exacto del chunk) es un camino menos común que el prefill normal
  (`ChatSession` usa `windowSize` por defecto ~512, así que la mayoría de
  prompts caben en un solo chunk y nunca disparan la transición
  `updateConcat`→`updateInPlace`) — real, pero no es el camino crítico del
  95% de usos.
- Arreglarlo bien requiere: instrumentación, hipótesis confirmada, fix
  mínimo, test sintético en verde, y al menos una verificación con modelo
  real vía `IntegrationTesting` que ejercite sliding window con prompt largo
  — más de lo que cabe con seguridad en el resto de esta sesión ya muy
  extendida.

## Siguiente acción recomendada

Abrir una sesión dedicada que:
1. Instrumente `RotatingKVCache` con logging temporal de offset/idx/shape.
2. Confirme el punto exacto de divergencia numérica (por capa, por posición).
3. Proponga el fix mínimo (probablemente: hacer que la transición
   `updateConcat`→`updateInPlace` reconstruya explícitamente el estado
   circular, en vez de asumirlo) con test sintético cubriendo la transición.
4. Verifique con `Gemma4ChunkedPrefillTests` en verde + al menos un test de
   `IntegrationTesting` con un modelo Gemma real y prompt >512 tokens.
