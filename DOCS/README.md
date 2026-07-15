# MLX Swift LM Research Reports

Fecha inicial: 2026-07-01. Actualizado: 2026-07-15 (rama
`feature/complete-improvement-backlog`).

Esta carpeta resume una investigación técnica del framework `mlx-swift-lm` y
la implementación posterior de su backlog sobre `main`.

## Informes originales (2026-07-01)

- [Rendimiento y generación](performance-and-generation.md): prefill, decode, speculative decoding, MTP, sampling y medición.
- [Compatibilidad y modelos](compatibility-and-models.md): arquitecturas soportadas, VLM, embeddings, cuantizaciones y portabilidad.
- [Compatibilidad de Bonsai 1-bit](bonsai-1bit-compatibility.md): port affine-only, revisiones exactas, validación y rollback.
- [Memoria y KV cache](memory-kv-cache.md): caché rotatoria, caché cuantizada, wired memory, límites Apple y patrones de uso.
- [Playbook de implementación](implementation-playbook.md): recetas prácticas para exprimir el framework en apps y herramientas.
- [Tech debt e investigación](tech-debt-and-research-backlog.md): backlog priorizado de mejoras, deuda técnica y experimentos.
- [Fuentes](sources.md): fuentes locales y externas consultadas.

## Implementación del backlog (2026-07-15)

> El estado verificable por bloque, commits, tests y límites está en
> [backlog-completion-status.md](backlog-completion-status.md). El
> [plan de continuación](backlog-continuation-plan.md) conserva los comandos
> reproducibles y el criterio de publicación.

Los 11 bloques operativos están implementados y validados:

- benchmark JSON, sampling y matriz de compatibilidad generada desde los
  registros vivos;
- estimación de KV cache, composición con presupuesto wired y regresiones de
  long-context;
- ticket de memoria, prefill sin tokens visibles y persistencia versionada en
  `ChatSession`;
- speculative decoding adaptativo opt-in con telemetría y fallback sticky;
- [MTP de producción](mtp-production-scoping.md) en `ChatSession`, con gates
  de memoria, continuidad multiturno, fallback cuantizado y benchmark JSON;
- metadata tipada y conservadora para thinking/tool compatibility;
- [dashboard de conformance](conformance-dashboard.md) y guardas VLM
  ejecutables sin convertir ausencias en éxitos;
- detokenización streaming acotada y menor materialización transitoria de
  vídeo/audio;
- prototipo experimental WHT para KV cache, con equivalencia y persistencia.

Se conservan como guías o investigación, no como promesas automáticas de
runtime: [long-context-recipes.md](long-context-recipes.md),
[porting-from-mlx-lm-playbook.md](porting-from-mlx-lm-playbook.md),
[kvscheme-research-note.md](kvscheme-research-note.md) y los límites de
[vlm-media-budgets.md](vlm-media-budgets.md).

El hallazgo de [Gemma4 chunked prefill](gemma4-chunked-prefill-investigation.md)
queda cerrado para este árbol: las cuatro variantes pasan con el runtime
M5-safe fijado, sin parchear `RotatingKVCache`.

## Lectura rápida

El framework ya tenía una base muy avanzada: Swift 6.1, arquitectura modular 3.x, desacoplo de tokenizer/downloader, 58 `model_type` LLM registrados, 18 registros VLM, embeddings, `GenerateParameters` con control de KV cache y sampling, caché cuantizada extensible, wired-memory coordination, tool calling y speculative decoding/MTP.

Las mejoras con más retorno no estaban en "añadir un flag más", sino en cerrar tres bucles:

1. Medición reproducible: el schema JSON, los adaptadores y los tests están
   cerrados; las cifras de memoria y rendimiento por dispositivo/modelo deben
   seguir procediendo de medición física.
2. Políticas adaptativas: memoria wired, admisión de auxiliares y speculative
   adaptativo están implementados como mecanismos opt-in. El framework no
   inventa umbrales universales ni cambia `kvBits`/`maxKVSize` automáticamente.
3. Compatibilidad verificable: matriz, metadata tipada, dashboard y guardas de
   media están cerrados. Los checkpoints de red y el estado de hardware iOS
   permanecen explícitamente `not_run` hasta ejecutarlos.

## Hallazgos críticos (original, con estado actualizado)

- MTP está integrado en `ChatSession`; conserva el state del target, no crea
  una KV cache del drafter y cae a target-only si falta shared K/V, incluida la
  cuantización. No se promete aceleración sin benchmark físico.
- `ChatSession` expone ticket wired por turno, prefill y persistencia de KV.
  `LMOutput.State` se conserva dentro de la sesión, pero no se serializa en el
  archivo de prompt cache.
- La cuantización de pesos cubre affine, MXFP4/MXFP8, NVFP4 y rutas especiales como ParoQuant, pero conversión sigue centrada en safetensors; `.bin`, GGUF y requantización de modelos ya cuantizados quedan fuera.
- Varias rutas públicas o semi-públicas de KV cache usaban `fatalError`; la deserialización de prompt cache corrupto y las nuevas APIs `quantized(...)` son recuperables mediante `KVCacheError`. Las firmas no throwing anteriores se mantienen deprecadas para no introducir una ruptura de API en 3.x.
- WHT KV es experimental y dequantiza/aplica la inversa por paso; sus tests
  prueban corrección, no una mejora de rendimiento o RSS.
