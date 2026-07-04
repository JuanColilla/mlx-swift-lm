# MLX Swift LM Research Reports

Fecha: 2026-07-01

Esta carpeta resume una investigacion tecnica del framework `mlx-swift-lm` en la rama `docs/framework-performance-research`, creada desde `origin/main`.

## Informes

- [Rendimiento y generacion](performance-and-generation.md): prefill, decode, speculative decoding, MTP, sampling y medicion.
- [Compatibilidad y modelos](compatibility-and-models.md): arquitecturas soportadas, VLM, embeddings, cuantizaciones y portabilidad.
- [Memoria y KV cache](memory-kv-cache.md): cache rotatoria, cache cuantizada, wired memory, limites Apple y patrones de uso.
- [Playbook de implementacion](implementation-playbook.md): recetas practicas para exprimir el framework en apps y herramientas.
- [Tech debt e investigacion](tech-debt-and-research-backlog.md): backlog priorizado de mejoras, deuda tecnica y experimentos.
- [Fuentes](sources.md): fuentes locales y externas consultadas.

## Lectura rapida

El framework ya tiene una base muy avanzada: Swift 6.1, arquitectura modular 3.x, desacoplo de tokenizer/downloader, 58 `model_type` LLM registrados, 17 familias VLM registradas, embeddings, `GenerateParameters` con control de KV cache y sampling, cache cuantizada extensible, wired-memory coordination, tool calling y speculative decoding/MTP.

Las mejoras con mas retorno no parecen estar en "anadir un flag mas", sino en cerrar tres bucles:

1. Medicion reproducible: benchmarks de TTFT, tokens/s, memoria pico, acceptance rate de speculative decoding y calidad tras KV quantization por dispositivo/modelo.
2. Politicas adaptativas: seleccionar `prefillStepSize`, `maxKVSize`, `kvBits`, draft model y memoria wired a partir del presupuesto real de dispositivo y de la conversacion.
3. Compatibilidad verificable: matriz viva de modelos, cuantizaciones, chat templates, thinking/tool formats y media budgets con tests de integracion etiquetados.

## Hallazgos criticos

- MTP mejora rendimiento en tests de integracion, pero hoy cae a passthrough si se combina con KV cuantizado y no esta integrado en `ChatSession`.
- `ChatSession` reutiliza KV cache y soporta speculative decoding clasico, pero no expone wired-memory ticket por turno ni sincroniza todo su estado mutable publico.
- La cuantizacion de pesos cubre affine, MXFP4/MXFP8, NVFP4 y rutas especiales como ParoQuant, pero conversion sigue centrada en safetensors; `.bin`, GGUF y requantizacion de modelos ya cuantizados quedan fuera.
- Varias rutas publicas o semi-publicas de KV cache usan `fatalError`; para apps de produccion conviene migrar errores recuperables a APIs throwing.
