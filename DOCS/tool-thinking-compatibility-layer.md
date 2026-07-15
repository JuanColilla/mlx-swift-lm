# Capa de compatibilidad tool/thinking

Item de backlog #10 (`DOCS/tech-debt-and-research-backlog.md:42-43`): "Tool/thinking
compatibility layer. Metadata por modelo con formato de tools, thinking mode, tags,
stop tokens y restricciones."

Relacionado con `DOCS/compatibility-and-models.md:31-33` ("Thinking mode y tool
calling"), que ya proponía etiquetar capacidades por modelo incluyendo
`supportsThinkingToggle`, `supportsTools`, `toolCallFormat` y `requiresExtraEOS`.

Esta nota separa lo que **ya existe** (la mitad de "tools") de lo que **sigue
faltando** (la mitad de "thinking"), con citas verificadas contra el código actual
de esta rama. No se ha escrito ni modificado código Swift para esta nota.

## 1. Lo que ya existe: formato de tool calls y stop tokens

### 1.1 `ToolCallFormat`

`Libraries/MLXLMCommon/Tool/ToolCallFormat.swift:64-103` define un enum
`ToolCallFormat: String, Sendable, Codable, CaseIterable` con diez formatos:
`json`, `lfm2`, `xmlFunction` (`"xml_function"`), `glm4`, `gemma`, `gemma4`,
`kimiK2` (`"kimi_k2"`), `minimaxM2` (`"minimax_m2"`), `mistral`, `llama3`.

Cada caso sabe construir su propio parser vía `createParser()`
(`ToolCallFormat.swift:109-136`), que instancia `JSONToolCallParser`,
`PythonicToolCallParser`, `XMLFunctionParser`, `GLM4ToolCallParser`,
`GemmaFunctionParser` (reutilizado para `.gemma`/`.gemma4` con tags distintos),
`KimiK2ToolCallParser`, `MiniMaxM2ToolCallParser`, `MistralToolCallParser` y
`Llama3ToolCallParser`. Todos conforman al protocolo `ToolCallParser`
(`ToolCallFormat.swift:13-35`), que expone `startTag`/`endTag` opcionales y
`parse(content:tools:)`.

### 1.2 Inferencia automática: `ToolCallFormat.infer(from:configData:)`

`ToolCallFormat.swift:159-225` mapea `model_type` (de `config.json`) al formato
correspondiente, con una señal secundaria para Llama:

- `model_type == "llama"`: sin señal adicional devuelve `nil`; con
  `vocab_size >= 128000` o `rope_scaling.rope_type == "llama3"` devuelve `.llama3`
  (`:163-182`).
- Prefijo `lfm2*` → `.lfm2` (`:185-187`).
- Prefijo `glm4*` → `.glm4` (`:190-192`).
- Prefijo `gemma4*` → `.gemma4` (`:195-197`); `gemma` exacto → `.gemma` (`:200-202`).
- Prefijo `nemotron*` → `.xmlFunction` (`:205-207`).
- Prefijo `qwen3_5*` → `.xmlFunction` (`:210-212`).
- Prefijo `qwen3_next*` → `.xmlFunction` (`:215-217`).
- Prefijo `mistral3*` → `.mistral` (`:220-222`).
- Cualquier otro `model_type` → `nil`.

### 1.3 Dónde vive el dato: `ModelConfiguration`

`Libraries/MLXLMCommon/ModelConfiguration.swift:98-119` declara los cuatro campos
que cubren exactamente lo que pide el item de backlog para "tools" y "stop
tokens":

- `extraEOSTokens: Set<String>` (`:102`) — tokens EOS adicionales como strings.
- `stopStrings: Set<String>?` (`:104-108`) — secuencias de parada explícitas; si es
  `nil`, `effectiveStopStrings` (`:110-113`) hace fallback a `extraEOSTokens`.
- `eosTokenIds: Set<Int>` (`:116`) — IDs de EOS cargados desde `config.json` /
  `generation_config.json`.
- `toolCallFormat: ToolCallFormat?` (`:119`) — `nil` significa "usar el formato por
  defecto".

### 1.4 Precedencia: registro explícito gana, inferencia es el fallback en carga

En `Libraries/MLXLLM/LLMModelFactory.swift:588-595`, dentro de `_load(...)`:

```swift
var mutableConfiguration = configuration
mutableConfiguration.eosTokenIds = eosTokenIds
mutableConfiguration.stopStrings.formUnion(generationConfig?.stopStrings ?? [])
if mutableConfiguration.toolCallFormat == nil {
    mutableConfiguration.toolCallFormat = ToolCallFormat.infer(
        from: baseConfig.modelType, configData: configData)
}
```

Es decir: si el `ModelConfiguration` registrado en `LLMRegistry` ya trae un
`toolCallFormat` explícito, ese valor se respeta tal cual y `infer` ni se llama
sobre él (el `if` solo entra cuando es `nil`). Si no hay valor explícito, se infiere
en tiempo de carga a partir del `model_type` real leído de `config.json` del
modelo descargado. Si tampoco hay match de inferencia, el valor queda `nil` en el
`ModelConfiguration` final, y son los puntos de consumo los que hacen el fallback a
`.json`: `Libraries/MLXLMCommon/Evaluate.swift:1524`, `:1582` y `:1722` usan todos
`context.configuration.toolCallFormat ?? .json` / `modelConfiguration.toolCallFormat
?? .json`. El mismo patrón de inferencia existe para VLM en
`Libraries/MLXVLM/VLMModelFactory.swift:385-386`, aunque ahí se llama
`ToolCallFormat.infer(from: baseConfig.modelType)` sin `configData`, así que la
señal secundaria de Llama 3 (vocab_size/rope_scaling) no aplica a modelos VLM.

### 1.5 Qué modelos de `LLMRegistry` tienen `toolCallFormat` explícito

`LLMRegistry.all()` (`LLMModelFactory.swift:411-469`) registra 55 modelos. De esos,
solo 3 fijan `toolCallFormat` explícitamente en el literal de registro:

| Modelo | `toolCallFormat` explícito | Ubicación |
|---|---|---|
| `glm4_9b_4bit` (`mlx-community/GLM-4-9B-0414-4bit`) | `.glm4` | `LLMModelFactory.swift:318-322` |
| `lfm2_1_2b_4bit` (`mlx-community/LFM2-1.2B-4bit`) | `.lfm2` | `LLMModelFactory.swift:349-353` |
| `lfm2_8b_a1b_3bit_mlx` (`mlx-community/LFM2-8B-A1B-3bit-MLX`) | `.lfm2` | `LLMModelFactory.swift:385-389` |

Los otros 52 modelos del registro (familias Llama 3.x, Qwen 2/3/3.5/3.6, Gemma
2/3/3n/4, Mistral/Mixtral, Phi/Phi3/PhiMoE, DeepSeek R1, Granite, MiMo, GLM4-MoE
(no registrado explícitamente aún vía `LLMRegistry`, solo `glm4_9b_4bit` lo está),
AceReason, Bitnet, Baichuan-M1, SmolLM3, ERNIE-4.5, Exaone4, Lille-130m,
OLMoE/OLMo2, Ling-mini, Granite-4.0-H, nanochat, GPT-OSS, Jamba, Nemotron-Labs
Diffusion) no fijan `toolCallFormat` en el registro y quedan enteramente a merced
de `ToolCallFormat.infer(from:configData:)` en tiempo de carga, usando el
`model_type` real que traiga el `config.json` descargado de cada repo de Hugging
Face — dato que esta nota no ha verificado modelo por modelo porque requeriría
descargar cada `config.json`, fuera del alcance de una revisión de código estática.

## 2. Lo que falta: no existe metadata estructurada de "thinking mode"

Se buscó en todo `Libraries/` (excluyendo tests) cualquier rastro de metadata de
thinking: `thinking`, `enableThinking`, `think_mode`, `ThinkingMode`,
`reasoning_content`, tags `<think>`/`</think>`. El único hallazgo real es:

`Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift:634`, `:667` y
`:741` — tres tests de integración pasan
`additionalContext: ["enable_thinking": false]` (Nemotron, dos veces) o
`additionalContext: ["enable_thinking": true]` (Qwen3.5) al construir un
`UserInput`.

Eso es todo. No hay ningún enum, struct, campo de `ModelConfiguration`, ni
constante en ningún modelo (`Qwen3.swift`, `Qwen35.swift`, `GLM4.swift`,
`GLM4MOE.swift`, `DeepseekV3.swift`, etc.) que declare si un modelo soporta
thinking, qué tags usa, o qué clave de `additionalContext` espera. Confirmado con
grep dirigido sobre esos archivos: cero coincidencias.

### 2.1 Por qué no hay nada: el mecanismo vive en la plantilla Jinja, no en Swift

`additionalContext` es un `[String: any Sendable]?` opaco
(`Libraries/MLXLMCommon/Tokenizer.swift:16-20`) que viaja sin validación hasta
`Tokenizers.Tokenizer.applyChatTemplate(messages:tools:additionalContext:)`, la
implementación real de swift-transformers, expuesta a este repo vía el macro
`TokenizerAdaptorMacro`
(`Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:68-125`): el
bridge `TokenizerBridge.applyChatTemplate` simplemente reenvía
`messages`/`tools`/`additionalContext` a `upstream.applyChatTemplate(...)`
(`:117-121`) sin tocarlos.

La plantilla Jinja en sí (`chat_template` de `tokenizer_config.json`) se descarga
de Hugging Face en tiempo de ejecución junto con el resto del tokenizer — no está
empaquetada en este repo (no hay ningún `.jinja` ni `chat_template*` bajo
`Libraries/` o `DOCS/`). Es la plantilla la que decide si la clave
`enable_thinking` existe, qué hace con ella, y qué tags de salida
(`<think>...</think>` u otros) produce el modelo. Swift no tiene visibilidad de
eso hasta que la plantilla ya se resolvió: `enable_thinking` en
`IntegrationTestHelpers.swift` es una convención de nombre que coincide con el
kwarg que usan las plantillas de Qwen/Nemotron en Hugging Face, pero no hay ningún
tipo Swift que la valide, documente por modelo, o impida pasarla a un modelo que no
la entiende (en ese caso la plantilla Jinja simplemente la ignora o falla en
runtime, dependiendo de cómo esté escrita).

## 3. Diseño propuesto (no implementado) para cerrar el hueco

Propuesta mínima y aditiva, análoga a `toolCallFormat`, para no romper nada
existente:

```swift
/// Declara cómo un modelo expone (si acaso) un modo de razonamiento extendido.
/// nil en ModelConfiguration.thinkingSupport significa "desconocido/no declarado",
/// igual que toolCallFormat == nil hoy.
public enum ThinkingSupport: Sendable, Codable {
    /// El modelo no tiene modo thinking, o se desconoce.
    case none

    /// El modelo soporta activar/desactivar thinking vía una clave de
    /// additionalContext pasada a la plantilla Jinja (p. ej. Qwen3.5, Nemotron).
    case toggleableViaTemplate(contextKey: String)

    /// El modelo siempre razona y envuelve la salida en tags fijos
    /// (p. ej. <think>...</think>) que hay que parsear/filtrar en post-proceso.
    case alwaysOn(startTag: String, endTag: String)
}
```

Y en `ModelConfiguration`, junto a `toolCallFormat`:

```swift
public var thinkingSupport: ThinkingSupport?
```

Con la misma precedencia que ya usa `toolCallFormat`: valor explícito de registro
gana, y a falta de él se podría intentar un `ThinkingSupport.infer(from:
modelType:)` estático — pero, a diferencia de `ToolCallFormat.infer`, esa
inferencia **no puede** basarse solo en `config.json`, porque `config.json` no
declara si la plantilla soporta `enable_thinking` ni qué tags usa; esa información
solo existe dentro del propio `chat_template` de `tokenizer_config.json` (un
string Jinja arbitrario). En la práctica, `infer` para thinking tendría que
mantenerse como una tabla estática por `model_type` curada a mano (igual que hoy
`lfm2_1_2b_4bit`/`glm4_9b_4bit` fijan `toolCallFormat` a mano porque la inferencia
automática no los cubre), no una inspección real del `config.json`.

Los tres casos del enum están respaldados por evidencia concreta encontrada en
este repo: Nemotron y Qwen3.5 usan hoy `enable_thinking` como
`toggleableViaTemplate` (`IntegrationTestHelpers.swift:634,667,741`); el caso
`alwaysOn` con tags fijos es la forma habitual en que modelos tipo DeepSeek-R1
exponen razonamiento (sin toggle, siempre envuelto en tags), aunque esta nota no
encontró en este repo Swift ningún manejo explícito de tags `<think>` para
`deepseek_r1_4bit` (`LLMModelFactory.swift:303-306`) — ese modelo está registrado
sin ninguna configuración especial de thinking ni tool call format, así que hoy su
salida con tags de razonamiento llega sin filtrar a quien consuma el stream.

## 4. Recomendación

Documentar el hueco es el alcance correcto para este item **hoy**; construir la
capa completa ahora tiene valor limitado, por una razón concreta y verificada, no
especulativa: el toggle y los tags de thinking no son una propiedad del modelo que
Swift pueda leer de forma fiable — son una propiedad de la plantilla Jinja
concreta que Hugging Face sirve para ese repo, y esa plantilla puede cambiar entre
revisiones del mismo modelo sin que cambie `model_type`. Una tabla Swift
`model_type → ThinkingSupport` mantenida a mano (como se propone arriba) puede
quedar desincronizada de la plantilla real descargada en runtime exactamente de
la misma forma en que ya puede pasar con `toolCallFormat` inferido — con la
diferencia de que para `toolCallFormat` sí existe una señal fiable en
`config.json` (`model_type`, `vocab_size`, `rope_scaling`), mientras que para
thinking no existe ninguna señal estructurada equivalente en `config.json` hoy: es
puro contenido de plantilla.

Dicho esto, el `enum ThinkingSupport` de la sección 3 sí vale la pena escribirlo
como **metadata declarativa de mejor esfuerzo para los ~3-5 modelos del registro
donde ya se sabe la respuesta a mano** (Qwen3.5, Nemotron, DeepSeek-R1, quizá
GLM4-MoE), del mismo modo en que hoy `toolCallFormat` se fija a mano para
LFM2/GLM4 porque la inferencia no los cubre. Eso resolvería el caso de uso real
que ya existe en `IntegrationTestHelpers` (dejar de tener el string mágico
`"enable_thinking"` repetido sin ningún tipo que lo respalde) sin prometer una
inferencia automática que, a diferencia de tools, no tiene base fiable en
`config.json`. Una alternativa más barata y quizá más honesta a medio plazo:
introspección de la plantilla Jinja ya descargada (buscar `enable_thinking` como
substring del `chat_template` string, o tags `<think>` en `additionalSpecialTokens`
del tokenizer) en vez de una tabla Swift mantenida a mano — pero eso es
investigación adicional no cubierta por esta nota y no tiene aún ningún prototipo
en este repo.
