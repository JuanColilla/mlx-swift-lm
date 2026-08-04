# Capa de compatibilidad tool/thinking

Item de backlog #10 (`DOCS/tech-debt-and-research-backlog.md`): metadata por
modelo para formato de tools, thinking mode, tags, stop tokens y restricciones.

La capa es ahora aditiva y declarativa: conserva las APIs existentes de tools y
stop tokens, y añade metadata tipada de thinking sin cambiar cómo se construye
`UserInput` ni cómo se procesa el texto generado.

## 1. Tool calls y stop tokens

`ToolCallFormat` declara los formatos de tools que el runtime sabe parsear:
`json`, `lfm2`, `xmlFunction`, `glm4`, `gemma`, `gemma4`, `kimiK2`,
`minimaxM2`, `mistral` y `llama3`. Cada caso crea su `ToolCallParser` y
`ToolCallFormat.infer(from:configData:)` usa `model_type` y, para Llama, señales
secundarias de `config.json`.

`ModelConfiguration` mantiene las otras señales de comportamiento:

- `extraEOSTokens`: tokens EOS adicionales expresados como strings.
- `stopStrings`: secuencias de parada explícitas; si es `nil`,
  `effectiveStopStrings` usa `extraEOSTokens`.
- `eosTokenIds`: IDs cargados desde `config.json` o
  `generation_config.json`.
- `toolCallFormat`: formato explícito o inferido durante la carga.

El valor explícito de `toolCallFormat` sigue teniendo precedencia. Solo cuando
es `nil`, `LLMModelFactory` y `VLMModelFactory` intentan inferirlo; si no hay una
señal concluyente, los consumidores existentes conservan su fallback a `.json`.

## 2. Metadata tipada de thinking

`Libraries/MLXLMCommon/ThinkingSupport.swift` define:

```swift
public enum ThinkingSupport: Sendable, Codable, Hashable {
    case none
    case toggleableViaTemplate(contextKey: String)
    case alwaysOn(startTag: String, endTag: String)
}
```

`ModelConfiguration.thinkingSupport` es opcional y se propaga también por
`ResolvedModelConfiguration` y por los contextos finales de LLM, VLM y
embedders. La semántica distingue dos estados que no deben confundirse:

- `nil`: soporte desconocido o todavía no declarado.
- `.none`: el checkpoint se ha verificado explícitamente como no compatible.

Los casos con valores asociados conservan la información que necesita una app:
la clave que debe pasar a `additionalContext`, o los tags que debe tratar en
postproceso. La metadata no activa, desactiva ni elimina thinking por sí sola.

## 3. Inferencia conservadora y precedencia

El thinking toggle es una propiedad de la plantilla Jinja concreta, no de
`model_type`. Por eso `ThinkingSupport.infer(...)` no mantiene una tabla de
arquitecturas ni promete detectar comportamiento desde `config.json`.

La inferencia inspecciona únicamente las fuentes reales descargadas con el
tokenizer:

1. `chat_template` dentro de `tokenizer_config.json`, tanto string único como
   colecciones de plantillas nombradas.
2. El fichero independiente `chat_template.jinja`.

Si alguna plantilla contiene la variable exacta `enable_thinking`, devuelve
`.toggleableViaTemplate(contextKey: "enable_thinking")`. Una aparición fuera de
`chat_template` no cuenta. Encontrar solo `<think>`/`</think>` tampoco permite
inferir `.alwaysOn`: esos tokens no demuestran si el razonamiento es opcional,
obligatorio o solo metadata del tokenizer.

La precedencia en carga es:

1. `thinkingSupport` explícito del `ModelConfiguration` o registro.
2. Inferencia sobre la plantilla ya descargada, solo cuando el valor es `nil`.
3. `nil` cuando no hay evidencia suficiente.

El mismo criterio se aplica en `LLMModelFactory`, `VLMModelFactory` y el loader
ParoQuant. Esto permite detectar, por ejemplo, el checkpoint Nemotron usado por
los tests de integración aunque no exista una entrada exacta para él en
`LLMRegistry`, sin generalizar esa capacidad a toda arquitectura que empiece por
`nemotron`.

## 4. Checkpoints declarados explícitamente

Solo se fijó metadata donde hay una señal concreta conocida:

| Checkpoint | `thinkingSupport` | Evidencia/criterio |
|---|---|---|
| `mlx-community/Qwen3.5-2B-4bit` | `.toggleableViaTemplate(contextKey: "enable_thinking")` | Los tests de integración usan esa clave con el checkpoint |
| `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` | `.alwaysOn(startTag: "<think>", endTag: "</think>")` | Contrato de salida always-on de DeepSeek-R1 distill |
| `mlx-community/DeepSeek-R1-4bit` | `.alwaysOn(startTag: "<think>", endTag: "</think>")` | Contrato de salida always-on de DeepSeek-R1 |

No se declara metadata explícita para `Qwen3.6-27B`,
`GLM-4-9B-0414`, `Nemotron-Labs-Diffusion`, ni para los Qwen3.5 VLM. Compartir
familia, arquitectura o formato de tool calls no demuestra que una plantilla
concreta exponga el mismo comportamiento de thinking. Si una de esas plantillas
incluye `enable_thinking`, el fallback de carga sí lo detectará para esa revisión
concreta.

## 5. Límites deliberados

Esta capa describe capacidad; no cambia la generación:

- No inserta automáticamente `additionalContext["enable_thinking"]`.
- No elimina ni separa bloques `<think>...</think>` del stream.
- No infiere `.alwaysOn` por nombre de familia ni por tokens especiales.
- No garantiza que una plantilla remota mantenga el mismo contrato entre
  revisiones; la metadata explícita sigue siendo responsabilidad del registro.
- No equivale a una prueba end-to-end del checkpoint. Esa evidencia pertenece al
  dashboard de conformance.

## 6. Verificación

`ThinkingSupportTests` cubre:

- round-trip `Codable` de los tres casos y valores asociados;
- semántica `Hashable`;
- plantillas inline, nombradas y `chat_template.jinja` independiente;
- rechazo de menciones fuera de `chat_template` y de tags sin toggle;
- precedencia del valor explícito sobre la inferencia;
- propagación por `ResolvedModelConfiguration`;
- metadata de los checkpoints registrados y ausencia deliberada en los no
  verificados.

`CompatibilityMatrixGeneratorTests` incluye ahora `thinkingSupport` junto a
`toolCallFormat` y `extraEOSTokens` en su salida derivada del registro.
