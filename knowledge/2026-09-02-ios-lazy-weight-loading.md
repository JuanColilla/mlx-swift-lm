# Carga de pesos en iOS: perezosa por defecto y sin referencias sobrantes (2026-09-02)

Dos parches propios en `Libraries/MLXLMCommon/Load.swift`, marcados con
`FORK(JuanColilla)` en el código para que sobrevivan identificables a las
fusiones del oficial. Ninguno cambia la API pública.

## Por qué

MLXHub cargaba Bonsai 27B 1-bit (5,1 GB) y LFM2.5-8B-A1B (4,8 GB) en un
iPhone 17 Pro Max hasta que dos cosas coincidieron:

1. iOS 27 beta 24A5430a bajó el límite por proceso de 6,5 GiB a **6,0 GiB**
   (`JetsamEvent` real, `per-process-limit`, 393.711 páginas).
2. El fork integró dos cambios del oficial que suben el pico de carga:
   - `#575` (`loadWeightArrays`): realiza cada tensor dentro de su work item,
     así que `sanitize`/`prepare` operan sobre el checkpoint **ya residente** y
     cualquier transformación se paga encima.
   - `#572` (`FusedQuantizedLinearProjectionCache`, Qwen 3.5): concatena las
     cuatro proyecciones GDN de cada capa en una copia física. Las vistas
     comparten almacenamiento con la copia, pero las originales solo se
     liberan cuando **nadie** las referencia — y `loadWeights` las seguía
     referenciando desde `weights` y `parameters` hasta salir de la función.
     En Bonsai eso son ~630 MB transitorios de golpe.

Detalle y aritmética: `MLXHub/knowledge/articles/largeModelLoadRegressionForensics.md`.

## Qué hace cada parche

- **`lazyWeightLoadingPreferred()`**: en iOS/tvOS/visionOS `loadWeights` usa
  el bucle serie y perezoso anterior a `#575`; la realización ocurre en el
  `eval(model)` final, donde MLX libera cada fuente al terminar su consumidor.
  `MLX_CONCURRENT_WEIGHT_LOAD=1` devuelve el cargador concurrente para A/B.
  macOS no cambia. Bajo el `evalLock` global de mlx-swift los `eval`
  concurrentes de `#575` se serializan igual, así que en iOS no se pierde
  tiempo de carga.
- **Soltar `weights`/`parameters` antes de `prepareInferenceState`**: el árbol
  de módulos pasa a ser el único dueño de los arrays, y la fusión GDN libera
  las originales capa a capa en vez de retenerlas todas hasta el `return`.

## Al fusionar el oficial

Si upstream reescribe `loadWeights`, portar los dos bloques `FORK(...)` encima
de la nueva versión; la prueba de que siguen ahí es
`LoadWeightsTests.testLazyWeightLoadingIsPreferredOnlyOnMemoryConstrainedPlatforms`
y el `grep FORK(JuanColilla) Libraries/MLXLMCommon/Load.swift` (debe dar 3).
