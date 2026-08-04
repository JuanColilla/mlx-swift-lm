# Fuentes consultadas

## Fuentes locales

- `README.md`: estado 3.x, instalacion, macros `MLXHuggingFace`, desacoplo tokenizer/downloader.
- `Package.swift`: productos, plataformas, Swift 6.1 y revision exacta del fork `JuanColilla/mlx-swift` con soporte affine de 1 bit.
- `Libraries/MLXLMCommon/Evaluate.swift`: `GenerateParameters`, sampling, KV parameters.
- `Libraries/MLXLMCommon/KVCache.swift`: protocolo KV cache, cache cuantizada, mascaras.
- `Libraries/MLXLMCommon/SpeculativeDecoding.swift`: telemetria y politica de memoria para speculative decoding.
- `Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift`: ruta MTP, passthrough y restricciones de KV compartido.
- `Libraries/MLXLMCommon/ChatSession.swift`: reutilizacion de KV cache, speculative clasico, estado de sesion y limites thread-safety.
- `Libraries/MLXLMCommon/WiredMemoryPolicies.swift`: politicas de wired memory.
- `Libraries/MLXLMCommon/WiredMemoryUtils.swift`: medicion/tuning de memoria wired.
- `Libraries/MLXLMCommon/ModelConversion.swift`: conversion safetensors y cuantizacion global/por capa.
- `Libraries/MLXLMCommon/BaseConfiguration.swift`: configuracion de cuantizacion en `config.json`.
- `Libraries/MLXLLM/LLMModelFactory.swift`: registro de arquitecturas LLM.
- `Libraries/MLXVLM/VLMModelFactory.swift`: registro de arquitecturas VLM y procesadores.
- `Libraries/MLXEmbedders/ModelFactory.swift`: registro de modelos de embeddings.
- `Libraries/MLXLMCommon/Documentation.docc/using.md`: integracion downloader/tokenizer y macros.
- `Libraries/MLXLMCommon/Documentation.docc/developing.md`: estrategia de desarrollo y tests.
- `Libraries/MLXLMCommon/Documentation.docc/porting.md`: guia de porting de modelos.
- `Tests/MLXLMTests/*`: tests de generacion, KV, MTP, speculative decoding, herramientas, VLM y cuantizacion.
- `IntegrationTesting/IntegrationTestingTests/*`: tests descargables de coherencia, MTP y herramientas.

## Fuentes externas

- MLX Swift LM GitHub: https://github.com/ml-explore/mlx-swift-lm
- MLX Swift GitHub: https://github.com/ml-explore/mlx-swift
- Fork PrismML de MLX Swift: https://github.com/PrismML-Eng/mlx-swift
- Variante PrismML sobre 0.31.6: https://github.com/PrismML-Eng/mlx-swift/tree/v0.31.6_prism
- Core MLX de PrismML: https://github.com/PrismML-Eng/mlx
- MLX Swift LM documentation en Swift Package Index: https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon
- MLX documentation: https://ml-explore.github.io/mlx/build/html/index.html
- MLX LM Python reference: https://github.com/ml-explore/mlx-lm
- MLX community models en Hugging Face: https://huggingface.co/mlx-community
- Hugging Face Swift Transformers: https://github.com/huggingface/swift-transformers

## Nota de alcance

Esta investigacion combina lectura local del checkout actual con fuentes publicas del ecosistema. Las recomendaciones de rendimiento deben validarse con benchmarks en hardware fisico antes de convertirse en defaults del framework.
