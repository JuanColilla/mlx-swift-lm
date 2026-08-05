# Informe de unificación de ramas en `develop`

**Fecha:** 2026-08-04
**Estado:** Integración implementada en el worktree de `develop`
**Repositorio:** `JuanColilla/mlx-swift-lm`
**Objetivo:** Integrar en `develop` el upstream oficial actualizado y todas las ramas del fork, conservando las mejoras propias que sigan aportando valor y descartando implementaciones obsoletas, duplicadas o incompletas.

## 1. Resumen ejecutivo

La integración debe utilizar `origin/main` como base arquitectónica. El upstream ha evolucionado de forma importante las mismas superficies que modificaba el fork: convenciones de chat y reasoning, `ChatSession`, speculative decoding, KV cache, modelos y configuración SwiftPM.

No es correcto resolver los conflictos eligiendo una rama completa. La estrategia recomendada es:

1. Conservar las implementaciones oficiales actuales como base.
2. Portar sobre ellas las capacidades propias que no tengan equivalente oficial.
3. Sustituir parches propios antiguos por sus versiones ya rebasadas sobre upstream cuando existan.
4. Descartar APIs paralelas, prototipos superados y stubs que solo terminan en `fatalError`.
5. Mantener todas las ramas originales; la unificación no implica borrarlas.

Las decisiones principales propuestas son:

- Conservar pipeline parallel, carga selectiva de shards, DistributedGroup, Bonsai 1-bit, adaptive speculative decoding, MTP de alto nivel, prefill/cache versionada, controles de memoria y buffers acotados.
- Adoptar las implementaciones oficiales de `ReasoningConfig`, `ChatConventionsProviding`, continuaciones estructuradas de `ChatSession`, TurboQuant, Foundation Models y nuevos modelos.
- Descartar el runtime WHT propio, el stub de tensor parallel y la API paralela `ThinkingSupport`.

## 2. Estado Git analizado

| Referencia | Commit | Papel |
|---|---|---|
| `origin/main` | `3d292ec1a78a96030928c764115bbbd6e1a084b6` | Upstream oficial actualizado |
| `fork/main` | `e248cc37d94228d22d6a16580c577667396f0913` | Integración anterior del backlog y Bonsai 1-bit |
| `feature/distributed-pipeline-port` | `d362876fe4c13171423672afce6b529fc0e54f0f` | Rama activa; pipeline distribuido y materialización de `allGather` |
| `codex/selective-shard-loader` | `9987ed80965c323a897b113fad4b88379437e59b` | Antecesora de la rama activa distribuida |
| Base común `origin/main` / `fork/main` | `10e0cb7442920d3f67a08e067d6670334e9dadef` | Punto desde el que divergieron upstream y fork |

La rama `develop` se creó desde `origin/main` en un worktree aislado. La primera fusión con `fork/main` quedó detenida con conflictos deliberadamente sin resolver hasta completar este análisis.

El checkout activo continúa en `feature/distributed-pipeline-port`. Sus archivos sin seguimiento, incluidos los duplicados con sufijos ` 2` y ` 3`, quedan fuera del alcance de esta operación y no deben tocarse.

## 3. Principios de resolución

### P-01. Upstream como base

Ante un conflicto estructural, se conserva la arquitectura de `origin/main` y se porta la funcionalidad propia. No se reemplazan archivos oficiales completos por sus versiones antiguas del fork.

### P-02. Una sola API por concepto

No deben coexistir dos modelos públicos que describan lo mismo. En particular, `ReasoningConfig` será la API de reasoning; `ThinkingSupport` no debe sobrevivir como sistema paralelo.

### P-03. La funcionalidad propia debe seguir siendo verificable

Una feature solo se considera preservada si conserva una API utilizable, tests relevantes y compatibilidad con las invariantes actuales del upstream.

### P-04. Evaluación explícita de colectivos MLX

Los colectivos distribuidos son lazy. Todo rango debe materializar el `allGather` mediante `eval()` aunque no vaya a consumir los logits. Esta propiedad es necesaria para evitar bloqueos entre rangos.

### P-05. Sin APIs públicas que solo fallen

Un stub público que termina siempre en `fatalError` no representa una feature integrada. Se elimina o se mantiene fuera de la API hasta que exista una implementación real.

### P-06. Dependencia MLX única y coherente

`mlx-swift-lm` debe resolver una sola copia de `mlx-swift`. Bonsai 1-bit y DistributedGroup deben convivir en una única revisión del fork basada en una versión oficial compatible con el upstream actual.

## 4. Features oficiales que deben conservarse

### 4.1 Convenciones de chat y reasoning

El upstream incorpora:

- `ChatConventionsProviding` en los modelos.
- `ReasoningConfig` con delimitadores y estrategia de prompt.
- `ChatConventionsRegistry` para resolvers externos y casos dependientes del repositorio, como DeepSeek R1 Distill.
- Precedencia explícita: configuración proporcionada por el usuario, resolver registrado y declaración del modelo.

**Decisión:** conservar íntegramente esta arquitectura.

### 4.2 Continuaciones estructuradas de `ChatSession`

El upstream corrige la relación entre:

- Mensajes conversacionales.
- Tokens realmente representados por la KV cache.
- Estado adicional del modelo, como deltas de RoPE.
- Tool calls y continuaciones después de resultados de herramientas.
- Reutilización, trim o reconstrucción de cache.

**Decisión:** usar esta implementación como base. No conservar el flujo interno antiguo de nuestra rama.

### 4.3 TurboQuant KV cache

TurboQuant oficial incluye rotación Walsh-Hadamard, codebooks, kernels Metal, calibración, persistencia, rutas para speculative decoding y protección de capas sensibles.

**Decisión:** conservar TurboQuant y retirar nuestro runtime WHT experimental.

### 4.4 Correcciones de KV cache

Se conservan:

- Dispatch dinámico de `BaseKVCache.ropeOffset`.
- Corrección de la máscara causal cuantizada basada en `finfo.min`.
- Compatibilidad de prompt cache con TurboQuant.

### 4.5 Foundation Models y guided generation

El upstream añade:

- `MLXFoundationModels`.
- `MLXGuidedGeneration`.
- Salida restringida mediante JSON Schema, EBNF y structural tags.
- Traits SwiftPM para habilitar o excluir la integración.
- Tool calling y reasoning conectados con Foundation Models.

**Decisión:** conservar targets, traits, macros y configuración de proyecto oficiales.

### 4.6 Compatibilidad de modelos

Se conservan las arquitecturas y correcciones oficiales recientes, entre ellas:

- Qwen3-VL-MoE.
- Nanbeige.
- Hunyuan dense V1.
- DeepSeek-V2.
- EOS IDs desde configuraciones `text_config` anidadas.
- MTP oficial para Gemma 4.

## 5. Features propias que deben conservarse

### 5.1 Pipeline parallel distribuido

Incluye:

- Sharding por rangos contiguos de capas.
- Wrappers de primera y última capa para `recv`, `send` y `allGather`.
- Adaptaciones de modelos homogéneos e híbridos.
- Ajuste de contadores, offsets, tipos de capa y reconstrucción de caches.
- `eval(gathered)` en todos los rangos para ejecutar realmente el colectivo lazy.

**Decisión:** conservar y portar sobre las definiciones actuales de los modelos oficiales.

### 5.2 Carga selectiva de shards

Evita que cada rango tenga que materializar primero el modelo completo antes de descartar las capas no locales.

**Decisión:** conservar. Es una condición necesaria para que pipeline parallel sirva para modelos que exceden la memoria de un dispositivo individual.

### 5.3 Bonsai affine 1-bit

El soporte se encuentra en el fork de `mlx-swift`; `mlx-swift-lm` aporta pin, documentación y tests de compatibilidad.

**Decisión:** conservar, pero no mantener el pin antiguo de forma literal. Debe coexistir con DistributedGroup en una revisión actualizada y verificable de `JuanColilla/mlx-swift`.

### 5.4 Errores recuperables de KV cache

La mejora reemplaza terminaciones del proceso por `KVCacheError` en cuantización y restauración de cache corrupta.

Existe una versión posterior, `fork/upstream/kvcache-throwing-errors`, que ya adapta TurboQuant mediante:

- `fa6416b` — errores recuperables.
- `075e6f1` — adaptación de las capas protegidas de TurboQuant.

**Decisión:** integrar la versión rebasada y no el parche antiguo de `fork/main`.

### 5.5 Presupuesto de memoria KV

Incluye estimación de memoria, wired-memory policies y regresiones para contexto largo.

**Decisión:** conservar. Complementa TurboQuant y sirve para decidir si una configuración cabe antes de ejecutarla.

### 5.6 Adaptive speculative decoding

Incluye seguimiento de aceptación, decisión de fallback y continuación autoregresiva sin perder los tokens pendientes de una ronda especulativa.

**Decisión:** conservar y adaptar a la telemetría y bucles oficiales actuales.

### 5.7 MTP de alto nivel en `ChatSession`

Nuestra rama añade configuración de MTP, carga del drafter, política de memoria, fallback y estadísticas accesibles desde `ChatSession`.

**Decisión:** conservar estas capacidades de alto nivel, reutilizando los iteradores y modelos MTP oficiales donde proceda.

### 5.8 Prefill y persistencia versionada

El upstream ya sabe reutilizar y guardar una cache, pero nuestra rama añade:

- `ChatSession.prefill` explícito.
- Construcción de caches reutilizables sin generar una respuesta.
- Metadatos de versión.
- Validación y snapshot al cargar.

**Decisión:** portar estas APIs diferenciales; reutilizar el `saveCache` oficial para lo que ya cubra.

### 5.9 Buffers acotados

Incluye:

- Ventana acotada para streaming detokenization.
- Reserva previa y acumulación eficiente de audio.
- Conversión progresiva de frames de vídeo para no retener simultáneamente todas las `CIImage` y `MLXArray`.

Existe una rama rebasada `fork/upstream/bounded-streaming-buffers` con los commits `6ed5c9a` y `86bc793`.

**Decisión:** usar la versión rebasada como fuente y adaptarla al arreglo oficial reciente del detokenizer para guided generation.

### 5.10 Herramientas de calidad y documentación

Se conservan, actualizándolas a las APIs actuales:

- Dashboard de conformidad.
- Guards VLM.
- Benchmarks de sampling y detokenización.
- Informes JSON de benchmarks.
- Matriz de compatibilidad generada desde registries.
- Guías de porting, memoria, contexto largo, VLM y rendimiento.

## 6. Features propias que deben descartarse o sustituirse

### 6.1 `ThinkingSupport`

`ThinkingSupport` duplica parcialmente `ReasoningConfig`, pero este último también modela delimitadores, valores por defecto, errores de supresión y la integración con Foundation Models.

**Decisión:** eliminar `ThinkingSupport` como API y propiedad paralela.

La inferencia conservadora desde `tokenizer_config.json` o `chat_template.jinja` sí aporta valor. Debe portarse como fallback o resolver de `ReasoningConfig`, sin crear una segunda fuente de verdad.

### 6.2 Runtime WHT propio

`WalshHadamardQuantizedKVCache` es un prototipo experimental solapado por TurboQuant oficial.

**Decisión:** retirar implementación y tests runtime. Conservar la nota de investigación marcada como histórica y explicar que TurboQuant la supera.

### 6.3 Stub de tensor parallel

`TensorParallel.swift` no implementa tensor parallel para ningún modelo y termina siempre en `fatalError`.

**Decisión:** no integrarlo en `develop`. Conservar, si interesa, una nota de deuda técnica sin API pública ejecutable.

### 6.4 Pines antiguos de `mlx-swift`

Se han observado tres estados:

- `fork/main` referencia `5e27a4c`, que contiene Bonsai 1-bit.
- La rama distribuida referencia `09b5d16`, no disponible en el checkout local inspeccionado.
- `Package.resolved` contiene `3c58e36`, verificado localmente como descendiente de Bonsai 1-bit y con DistributedGroup.

**Decisión:** no copiar ninguno ciegamente. Antes de cerrar `develop`, seleccionar o construir una revisión única con:

1. Base oficial compatible con `origin/main`.
2. Bonsai affine 1-bit.
3. DistributedGroup y sharding de capas.
4. Headers y submódulos sincronizados.
5. Build macOS e iOS y smoke test distribuido.

### 6.5 Implementaciones internas antiguas

Se descartan las implementaciones antiguas completas de `ChatSession`, factories, configuración y `.pbxproj`. Solo se portan sus capacidades diferenciales.

## 7. Matriz de conflictos actuales

| Archivo | Cambio oficial | Cambio propio | Resolución propuesta |
|---|---|---|---|
| `IntegrationTesting.xcodeproj/project.pbxproj` | Plataformas, simulador y flags Foundation Models | Tests de conformidad y guards VLM | Mantener configuración oficial y añadir nuestros tests |
| `LLMModelFactory.swift` | Modelos nuevos y `ChatConventionsProviding` | `ThinkingSupport` y presets | Mantener factory oficial; portar solo inferencia útil a `ReasoningConfig` |
| `VLMModelFactory.swift` | Qwen3-VL-MoE, EOS anidado y reasoning oficial | `ThinkingSupport` | Igual que LLM factory |
| `ModelConfiguration.swift` | `ReasoningConfig` | `ThinkingSupport` | Una sola propiedad: `reasoningConfig` |
| `Downloader.swift` | Propagación de `ReasoningConfig` | Inferencia desde templates | Mantener API oficial y añadir fallback de inferencia |
| `ChatSession.swift` | Continuaciones estructuradas | Prefill, cache versionada y MTP de alto nivel | Base oficial más port selectivo de nuestras APIs |
| `Evaluate.swift` | Continuaciones y TurboQuant | Adaptive fallback y MTP | Combinar ambos flujos y preservar autorelease pools |
| `KVCache.swift` | TurboQuant, `ropeOffset` y máscaras | Errores recuperables, memoria y WHT | TurboQuant oficial + errores rebased + memoria; descartar WHT |
| `ChatSessionTests.swift` | Regresiones de continuaciones y tools | Prefill/cache | Mantener ambos conjuntos adaptando nuestras APIs |
| `KVCacheTests.swift` | `ropeOffset` y máscara causal | Cache corrupta y cuantización recuperable | Mantener todas las regresiones compatibles; eliminar tests WHT |

## 8. Relación entre ramas del fork

### 8.1 Ramas distribuidas

`codex/selective-shard-loader` (`9987ed8`) es antecesora de `feature/distributed-pipeline-port` (`d362876`). La rama activa contiene además la materialización del `allGather`.

**Integración:** fusionar la rama activa. La rama `codex/selective-shard-loader` quedará incluida por ancestría y no necesita una segunda aplicación de cambios.

`feature/distributed-group-port-bump` representa una etapa anterior del pin de la dependencia y queda funcionalmente superada por la rama distribuida completa.

### 8.2 Ramas `upstream/*` del fork

- `upstream/bounded-streaming-buffers` contiene la versión rebasada de los buffers acotados.
- `upstream/kvcache-throwing-errors` contiene la versión rebasada de errores recuperables y su adaptación a TurboQuant.

**Integración:** usar estas versiones para resolver semánticamente los cambios equivalentes antiguos de `fork/main`. Las fusiones deben conservar la ancestría de las ramas, pero el contenido final no debe duplicar implementaciones.

## 9. Orden recomendado de integración

1. Resolver `fork/main` sobre `origin/main` siguiendo la matriz anterior.
2. Compilar y ejecutar tests antes de introducir distribución.
3. Integrar `upstream/bounded-streaming-buffers` y reconciliar el detokenizer oficial.
4. Integrar `upstream/kvcache-throwing-errors` y ejecutar tests de KV/TurboQuant.
5. Integrar `feature/distributed-pipeline-port`, que ya contiene `codex/selective-shard-loader`.
6. Reconciliar el pin único de `mlx-swift`.
7. Adaptar los modelos distribuidos a las arquitecturas oficiales nuevas.
8. Ejecutar la verificación completa.
9. Confirmar que todas las cabezas de rama están contenidas en `develop` o representadas por un merge explícito cuyo resultado semántico esté documentado.

## 10. Criterios de verificación

### 10.1 Integridad Git

- No quedan marcadores `<<<<<<<`, `=======` o `>>>>>>>`.
- El checkout activo original conserva su rama y archivos sin seguimiento.
- Ninguna rama del fork ha sido eliminada.
- `develop` parte del upstream oficial actualizado.
- El historial permite identificar qué ramas se integraron.

### 10.2 Compilación

- `swift build` completa con Swift 6.
- `swift test` completa sin regresiones.
- Build para destino iOS real con la dependencia personalizada de `mlx-swift`.
- La configuración sin el trait Foundation Models sigue compilando.

### 10.3 Reasoning y tool calling

- Qwen3 recibe `enable_thinking` correctamente.
- DeepSeek R1 no permite desactivar reasoning cuando no es soportado.
- Tool calling conserva el formato declarado por cada modelo.
- La inferencia desde template solo actúa cuando no existe una declaración más precisa.

### 10.4 KV cache

- TurboQuant y affine cache pasan sus tests.
- Cache corrupta produce errores Swift recuperables y no termina el proceso.
- Prompt cache conserva y valida metadatos.
- `ChatSession.prefill` permite guardar y reutilizar cache.
- Los presupuestos de memoria se calculan para caches normales, cuantizadas e híbridas.

### 10.5 Speculative decoding y MTP

- Los tokens pendientes se drenan antes del fallback.
- La secuencia target es determinista frente a la ruta no especulativa.
- La telemetría de aceptación y fallback es coherente.
- MTP puede rechazarse por política de memoria sin romper la generación normal.

### 10.6 Distribución

- Dos rangos producen la misma salida determinista que un solo nodo dentro de la tolerancia definida.
- Todos los rangos ejecutan `eval()` sobre el colectivo.
- La carga selectiva evita materializar pesos no locales.
- Cancelación y pérdida de un peer terminan con error controlado, sin bloqueo indefinido.
- Se prueba prefill y decode, no solo un `allSum` aislado.
- La validación final se realiza en dispositivos físicos; el simulador no demuestra ejecución MLX distribuida.

### 10.7 Bonsai 1-bit

- El checkpoint 1-bit carga sin rechazo de cuantización.
- Se valida round-trip affine y las rutas QMM/QMV relevantes.
- Se confirma compatibilidad con la revisión de `mlx-swift` usada por DistributedGroup.

## 11. Riesgos abiertos

### R-01. Dependencia personalizada desactualizada

El mayor riesgo es que el fork de `mlx-swift` con Bonsai y DistributedGroup no esté basado en la revisión que requiere el upstream actual. Debe resolverse antes de considerar estable `develop`.

### R-02. Port distribuido sobre modelos que han cambiado

Las factories y varios modelos han cambiado desde la base de la rama distribuida. Las adaptaciones de acceso a capas y `rebuildCaches()` deben revisarse modelo por modelo.

### R-03. Combinación MTP + continuaciones estructuradas

El `ChatSession` oficial registra mensajes y tokens de cache con más precisión. El port MTP debe conservar esa contabilidad al generar, cancelar, ejecutar tools o caer a target-only.

### R-04. Documentación histórica

Parte de `DOCS/` describe el estado 3.31.6v como actual. Debe conservarse como contexto histórico o actualizarse, pero no presentarse como el estado de `develop`.

## 12. Decisión recomendada

Se propone aprobar el siguiente alcance:

- **Conservar:** upstream completo, pipeline parallel, carga selectiva, DistributedGroup, Bonsai 1-bit, errores KV recuperables, presupuesto de memoria, adaptive speculative decoding, MTP de alto nivel, prefill/cache versionada, buffers acotados, dashboard y documentación vigente.
- **Portar:** las features anteriores sobre `ReasoningConfig`, `ChatConventionsProviding`, el nuevo `ChatSession`, TurboQuant y los modelos oficiales actuales.
- **Descartar:** `ThinkingSupport` como API paralela, runtime WHT propio, stub de tensor parallel, pins antiguos copiados literalmente e implementaciones internas antiguas de archivos oficiales.

Este criterio permite que `develop` contenga el trabajo del fork sin congelarlo sobre una arquitectura antigua ni perder las mejoras oficiales recientes.

## 13. Resultado de la implementación

La integración se ejecutó en `/private/tmp/mlx-swift-lm-develop`, sin cambiar el checkout
activo de `feature/distributed-pipeline-port` ni eliminar ramas. El resultado aplicado sigue
las decisiones anteriores:

- `develop` parte del upstream oficial `3d292ec` e incorpora el historial de `fork/main`,
  las dos ramas `upstream/*`, la rama distribuida, su antecesora de carga selectiva y la rama
  intermedia `feature/distributed-group-port-bump`. Todas las referencias remotas del fork
  comprobadas son ahora antecesoras de `develop`.
- `ReasoningConfig` y `ChatConventionsProviding` son la única arquitectura de reasoning;
  solo se conserva la inferencia conservadora de templates como fallback.
- `ChatSession` combina el ledger estructurado oficial con prefill transaccional, cache
  versionada, speculative decoding adaptativo y MTP.
- Se eliminan el runtime WHT, `ThinkingSupport` y el stub público de tensor parallel.
- `Package.swift` fija `mlx-swift` en `b00051a9c1bf4d42036d841ef2c803b50f82d115`.
  Su historial incluye `5e27a4c` (Bonsai 1-bit), `3c58e36` (DistributedGroup), `09b5d16`
  (errores recuperables al inicializar grupos) y el diferido de errores GPU asíncronos para
  evitar abortos fuera de una frontera Swift capturable.
- El pipeline ya no invoca los `rebuildCaches()` inexistentes de la rama original. Las capas,
  `kvHeads`, tipos de cache e índices de atención se recortan y recalculan para el shard local.
  Qwen3.5 desactiva su ruta de decode compilada al operar como pipeline, evitando que rangos
  posteriores vuelvan a interpretar hidden states recibidos como IDs de token.
- Los wrappers homogéneos aceptan únicamente `TransformerLayer`; desaparece el fallback que
  terminaba en `fatalError` para firmas no soportadas.

### 13.1 Verificación realizada

- Parse de todos los fuentes Swift: correcto.
- Type-check de `MLXLMCommon`, `MLXLLM` y `MLXVLM` contra un módulo temporal construido desde
  la revisión `09b5d16` de `mlx-swift`, antecesora con la misma API Swift que `b00051a`:
  correcto. SwiftPM resuelve el manifiesto final directamente en `b00051a`.
- Type-check del bloque principal de tests afectados, incluidos ChatSession, MTP, adaptive
  speculative decoding, KV cache, reasoning y el nuevo test de alineación del pipeline:
  correcto.
- Tests Python del dashboard: 5 de 5 correctos.
- `git diff --check` y formato de los archivos reconciliados: correctos.

El build completo de SwiftPM no llegó a compilar los targets de este repositorio porque el
Xcode instalado carece del componente opcional Metal Toolchain. La dependencia falla antes,
al invocar `metal` sobre `steel_attention.metal`, con el diagnóstico explícito
`missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain`. No se instaló
un componente del sistema sin autorización. Por tanto, el type-check aporta evidencia de
compilación Swift, pero siguen pendientes el enlace, la ejecución de la suite y la prueba
distribuida en dispositivos físicos descrita en los criterios de salida.
