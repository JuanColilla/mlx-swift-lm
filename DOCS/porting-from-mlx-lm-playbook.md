# Playbook de porting desde `mlx-lm` (Python) a mlx-swift-lm

> Backlog: `DOCS/tech-debt-and-research-backlog.md` #11 — "Pipeline de porting desde
> `mlx-lm`. Scaffold de configuración, modelo, registry, tests y mapping de pesos
> desde Python MLX."

## Qué es esto y qué no es

Esto es un **checklist manual/semi-manual**, no un generador de código. Parsear
`model.py` de Python arbitrariamente y emitir Swift correcto no es un problema
resuelto aquí ni se pretende resolver — las arquitecturas varían demasiado
(atención densa, MoE, SSM/Mamba, híbridas) para una traducción mecánica fiable.
Lo que sí ofrece este documento:

- Un orden de pasos verificado contra el código real del repo (no inventado).
- Convenciones concretas de este repo, citadas con `archivo:línea`.
- Tres ejemplos reales de `sanitize(weights:)` que cubren los tres tipos de
  transformación de pesos que vas a necesitar: eliminar claves no usadas,
  transponer/reshapear un tensor, y apilar pesos por-experto en un tensor
  batched.
- Plantillas `.swift.template` en `scripts/porting/templates/` como punto de
  partida copy-paste (no se compilan como parte del paquete — ver más abajo).

Modelos de referencia usados en todo el documento (elegidos porque son ports
recientes, completos y de tamaño moderado):

- **Mixtral** (`Libraries/MLXLLM/Models/Mixtral.swift`) — atención densa +
  MoE disperso vía `SwitchGLU`. Port de
  [`mixtral.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mixtral.py)
  (citado en `Mixtral.swift:6`).
- **Mamba2** (`Libraries/MLXLLM/Models/Mamba2.swift`) — arquitectura SSM pura,
  sin atención, con `MambaCache` propia. Port de
  [`mamba2.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba2.py)
  (citado en `Mamba2.swift:1`).

Ambos entraron en el repo el mismo día (commit `ccdc386` y `84a0cb0`,
2026-06-29) y son representativos de los dos extremos que vas a encontrar:
transformer-con-atención-estándar vs. arquitectura-sin-atención-con-cache-propia.

Ver también `Libraries/MLXLLM/Documentation.docc/adding-model.md` (guía
genérica de estructura, no específica de porting) y
`DOCS/compatibility-and-models.md` sección "Estructura de modelos
recomendada" (convenciones de alto nivel).

---

## Checklist

### 0. Prerequisitos

- [ ] Localiza el `model.py` y el `config.json` de referencia en el repo
  Python `mlx-lm` (https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models)
  o en un checkpoint real de `mlx-community/*` en Hugging Face.
- [ ] Confirma el `model_type` exacto tal como aparece en `config.json`
  (clave `"model_type"`) — es el identificador que usarás en el registry
  (paso 5).
- [ ] Revisa si la arquitectura reutiliza bloques ya existentes en el repo
  (`SwitchGLU` para MoE, `RoPE`/`SuScaledRoPE` para posición, `MambaCache`/
  `SSM.swift` para SSM) antes de escribir nada desde cero — busca en
  `Libraries/MLXLLM/Models/` y `Libraries/MLXLMCommon/` primero.

### 1. Extraer el esquema de configuración del Python

- [ ] Lee la clase `ModelArgs`/`@dataclass` en `model.py` y el `config.json`
  real correspondiente. Anota: nombre snake_case, tipo, si tiene default en
  Python, y si el valor puede faltar en `config.json` (HF a veces omite
  campos con default).
- [ ] Identifica campos derivados que Python calcula en `__post_init__`
  (ej. `ssm_state_size` cae a `state_size` si no está presente — ver
  `Mamba2.swift:65-66`) o relaciones como `head_dim = hidden_size /
  num_attention_heads` (`Mixtral.swift:23`).

### 2. Escribir el `Configuration: Codable, Sendable`

Convención de este repo (no la inventes distinta): **campos en camelCase en
Swift, `CodingKeys` explícito mapeando a snake_case de Python**, con un
`init(from decoder:)` manual que aplica `decodeIfPresent(...) ?? default`
para cualquier campo opcional en `config.json`.

Referencia real — `MixtralConfiguration` (`Mixtral.swift:8-61`):

```swift
public struct MixtralConfiguration: Codable, Sendable {
    var modelType: String = "mixtral"
    var vocabularySize: Int = 32000
    ...
    var headDim: Int { hiddenSize / attentionHeads }   // campo derivado, no en CodingKeys

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        ...
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "mixtral"
        self.vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32000
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)   // sin default -> requerido
        ...
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ...
    }
}
```

Puntos a replicar:

- [ ] Campos **sin** default en Python → `container.decode(...)` (falla si
  falta, correcto: es un bug de config real).
- [ ] Campos **con** default en Python → `container.decodeIfPresent(...) ??
  default`.
- [ ] Defaults que dependen de otro campo ya decodificado (ej.
  `kvHeads ?? attentionHeads`, `Mixtral.swift:50`; `ssmStateSize ??
  stateSize`, `Mamba2.swift:66`) — decodifícalos en orden, el `init(from:)`
  manual te da control total sobre el orden.
- [ ] Campos derivados puros (no vienen del JSON) como `headDim` —
  computed property, **no** en `CodingKeys` (`Mixtral.swift:23`).
- [ ] Si hay invariantes que deben validarse tras decodificar (dimensiones
  incompatibles, listas de tamaño esperado, etc.), conforma a
  `ModelConfigurationValidating`
  (`Libraries/MLXLMCommon/ModelFactory.swift:59-61`) e implementa
  `validateModelConfiguration() throws` — se invoca automáticamente desde el
  `create(...)` del factory (ver paso 5, `LLMModelFactory.swift:8-18`).
  Ejemplos reales: `Qwen3.swift`, `Qwen3Next.swift`, `Qwen3MoE.swift`.

Segundo ejemplo real, para el caso SSM sin atención — `Mamba2Configuration`
(`Mamba2.swift:8-71`), nota el comentario explicando el fallback de
`ssmStateSize` línea 65-66 y de `timeStepLimit` línea 67-69.

### 3. Escribir la estructura de clases del modelo

Patrón estándar del repo (varía en las piezas intermedias según la
arquitectura, pero el esqueleto exterior es constante):

```
<Nombre>Attention / <Nombre>Mixer   — módulo de mezcla de secuencia (atención o SSM)
<Nombre>MLP / <Nombre>SparseMoeBlock — feed-forward, denso o MoE
<Nombre>DecoderLayer / <Nombre>ResidualBlock — un bloque (norm + mixer + norm + ffn, con residuales)
<Nombre>ModelInner / <Nombre>Backbone — embedding + pila de capas + norm final
<Nombre>Model            — top-level: conforma a Module, LLMModel, [KVCacheDimensionProvider]
```

Referencia con atención (`Mixtral.swift`):

- `MixtralAttention` (línea 64-119) — proyecciones `q_proj/k_proj/v_proj/o_proj`
  con `@ModuleInfo(key:)` mapeando al nombre HF, `RoPE`, y
  `attentionWithCacheUpdate(...)` para la integración con `KVCache`.
- `MixtralSparseMoeBlock` (línea 121-149) — gate + `SwitchGLU` (el módulo MoE
  ya existente en `MLXNN`, reutilízalo en vez de escribir MoE desde cero).
- `MixtralDecoderLayer` (línea 151-174) — pre-norm + residual, patrón
  estándar `x + attn(norm(x))`, luego `h + moe(norm(h))`.
- `MixtralModelInner` (línea 176-204) — `embed_tokens`, array de capas,
  `norm` final; usa `createAttentionMask(h:cache:)`
  (`Libraries/MLXLMCommon/KVCache.swift:288`) para construir la máscara una
  vez por forward pass, no por capa.
- `MixtralModel` (línea 206-263) — conforma a `Module, LLMModel,
  KVCacheDimensionProvider`; `kvHeads: [Int]` es un array de tamaño
  `hiddenLayers` (una entrada por capa, línea 218); maneja `tieWordEmbeddings`
  con `lm_head` opcional (líneas 213, 221-223, 228-231).

Referencia sin atención, arquitectura SSM (`Mamba2.swift`):

- `Mamba2Mixer` (línea 93-214) reemplaza a la atención — sin `RoPE`, sin
  `KVCache` de tipo KV; en su lugar usa `MambaCache` (estado convolucional +
  estado SSM, indexado como `cache?[0]` / `cache?[1]`, línea 154, 201).
- `Mamba2Model` (línea 256-297) conforma solo a `Module, LLMModel` — **no**
  a `KVCacheDimensionProvider` (no tiene `kvHeads` en el sentido de atención).
  En su lugar sobrescribe `newCache(parameters:)` explícitamente (línea
  278-280) devolviendo `MambaCache()` por capa. Esto es exactamente lo que
  `DOCS/compatibility-and-models.md` pide: *"`newCache(parameters:)`
  específico por arquitectura, no genérico, para soportar (...) Mamba/SSM o
  caches híbridas"*.
- Usa `createSSMMask(h:cache:)` (`Libraries/MLXLMCommon/KVCache.swift:363`)
  en vez de `createAttentionMask` (línea 248 de `Mamba2.swift`).

Checklist:

- [ ] Cada submódulo con nombre de propiedad HF usa `@ModuleInfo(key:
  "nombre_hf")` o `@ParameterInfo(key: "nombre_hf")` para que la carga de
  pesos funcione sin sanitización adicional cuando el nombre ya coincide.
- [ ] Decide si la arquitectura necesita `KVCacheDimensionProvider`
  (atención con KV cache estándar) o un `newCache(parameters:)` manual
  (SSM, híbridas, sliding window no uniforme).
- [ ] El top-level model implementa `callAsFunction(_:cache:)` devolviendo
  logits `[batch, seq, vocab]`, con el patrón `lmHead ?? embedTokens.asLinear(...)`
  para tied embeddings si aplica.
- [ ] LoRA: conforma a `LoRAModel`
  (`Libraries/MLXLMCommon/Adapters/LoRA/LoRAModel.swift:12-25`) exponiendo
  `var loraLayers: [Module]` devolviendo el array de capas del backbone —
  así lo hacen los 52 modelos actuales del repo, ej. `Mixtral.swift:267-271`
  y `Mamba2.swift:294-296`.
  > **Nota de discrepancia de documentación**: `adding-model.md` (línea
  > 43-46) todavía muestra un método antiguo `loraLinearLayers()` que ya no
  > es el patrón real — ese nombre solo sobrevive como comentario obsoleto
  > en `Libraries/MLXLLM/LoraTrain.swift:79`. Usa `loraLayers: [Module]`
  > como en los ejemplos reales de arriba, no el ejemplo de ese `.md`.

### 4. Mapeo de nombres de pesos — `sanitize(weights:)`

Este es el paso donde más se rompen los ports. El `state_dict` de PyTorch
(claves tipo `model.layers.0.self_attn.q_proj.weight`) normalmente **ya
coincide** con los nombres que `@ModuleInfo(key:)` espera, porque los
autores de mlx-lm nombran los submódulos igual que HF/PyTorch. `sanitize`
solo hace falta para las **diferencias estructurales**, no para renombrados
triviales. Hay tres patrones reales en este repo — no inventes un cuarto
sin evidencia de que lo necesitas:

**Patrón A — eliminar claves no usadas.** El caso más común: buffers
precomputados en Python (típicamente `rotary_emb.inv_freq`) que MLX Swift
recalcula en el propio `RoPE` y que no deben cargarse como parámetro.
Ejemplo real completo, `LlamaModel.sanitize` (`Llama.swift:180-185`):

```swift
public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    // Remove unused precomputed rotary frequencies
    weights.filter {
        !$0.key.contains("self_attn.rotary_emb.inv_freq")
    }
}
```

El mismo patrón aparece en `AfMoE.swift:531`, `Bitnet.swift:469`,
`Apertus.swift:365`, `Internlm2.swift`, entre otros — es el default seguro
si tu arquitectura usa `RoPE` estándar y el checkpoint Python trae ese
buffer.

**Patrón B — transponer/reshapear un tensor.** Cuando MLX Swift espera un
layout distinto al de PyTorch para el mismo peso (típicamente convoluciones,
donde PyTorch usa `[out, in/groups, kernel]` y MLX `Conv1d` espera el kernel
en otro eje). Ejemplo real, `Mamba2Model.sanitize` (`Mamba2.swift:282-292`):

```swift
public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized = [String: MLXArray]()
    for (key, value) in weights {
        if key.contains("conv1d.weight"), value.dim(-1) != 1 {
            sanitized[key] = value.swappedAxes(1, 2)
        } else {
            sanitized[key] = value
        }
    }
    return sanitized
}
```

Nota el guard `value.dim(-1) != 1`: hace la transposición **idempotente**,
para que un checkpoint ya convertido (o re-guardado tras un primer
`sanitize`) no se transponga dos veces. Replica esa idempotencia siempre
que transpongas/reshapees en `sanitize`.

**Patrón C — apilar pesos por-experto en un tensor batched (MoE).** HF
guarda cada experto MoE como un submódulo separado
(`block_sparse_moe.experts.{i}.w1.weight`), pero el módulo `SwitchGLU` de
este repo espera un único tensor con eje de experto al frente. Ejemplo real,
`MixtralModel.sanitize` (`Mixtral.swift:234-262`):

```swift
public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitizedWeights = weights

    if configuration.tieWordEmbeddings {
        sanitizedWeights["lm_head.weight"] = nil
    }

    // Ya convertido (checkpoint MLX-nativo) -> nada que hacer, idempotente.
    if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
        return sanitizedWeights
    }

    for l in 0 ..< configuration.hiddenLayers {
        let prefix = "model.layers.\(l)"
        for (n, m) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
            for k in ["weight", "scales", "biases"] {
                if sanitizedWeights["\(prefix).block_sparse_moe.experts.0.\(n).\(k)"] != nil {
                    let toJoin = (0 ..< configuration.numLocalExperts).map { e in
                        sanitizedWeights.removeValue(
                            forKey: "\(prefix).block_sparse_moe.experts.\(e).\(n).\(k)")!
                    }
                    sanitizedWeights["\(prefix).block_sparse_moe.switch_mlp.\(m).\(k)"] =
                        MLX.stacked(toJoin)
                }
            }
        }
    }

    return sanitizedWeights
}
```

Tres cosas a replicar de este ejemplo si tu arquitectura es MoE:

- El guard de idempotencia (líneas 241-243): si la clave por-experto ya no
  existe, el checkpoint ya está en formato MLX — no hagas nada.
- Itera también sobre `["weight", "scales", "biases"]`, no solo `"weight"`
  — así el `sanitize` funciona igual para checkpoints cuantizados
  (`QuantizedLinear` guarda `scales`/`biases` además de `weight`) sin código
  adicional.
- `MLX.stacked(toJoin)` apila en un nuevo eje 0 (eje de experto) — confirma
  que el orden de `toJoin` (`0 ..< numLocalExperts`) coincide con el orden
  que `SwitchGLU`/`weightedExpertSum` espera.
- `tieWordEmbeddings` (líneas 237-239) es otro sub-patrón útil: si el
  checkpoint Python siempre incluye `lm_head.weight` aunque el modelo ate
  los embeddings, bórralo aquí para que no se intente cargar en un
  `lmHead` que no existe (`self._lmHead.wrappedValue` no se crea cuando
  `tieWordEmbeddings == true`, `Mixtral.swift:221-223`).

Checklist para este paso:

- [ ] Compara las claves del `state_dict` Python real (usa
  `safetensors.safe_open(...).keys()` en Python, o inspecciona el
  `.safetensors` con cualquier visor) contra los nombres `@ModuleInfo(key:)`
  de tu modelo Swift. Si coinciden 1:1, **no escribas `sanitize` — no lo
  necesitas**. Los tres modelos de referencia lo confirman: `Llama` solo
  filtra una clave, muchos modelos del repo no tienen `sanitize` en
  absoluto.
- [ ] Si hay discrepancia, identifica cuál de los tres patrones (A/B/C)
  aplica, o si es una combinación.
- [ ] Haz que cualquier transformación de `sanitize` sea **idempotente**
  (guard antes de transformar) — un usuario puede re-convertir un
  checkpoint ya convertido.
- [ ] Si conviertes/cuantizas pesos, no olvides `scales`/`biases` además de
  `weight` (patrón C).
- [ ] Escribe el test de `sanitize` **antes** de asumir que funciona — ver
  paso 6.

### 5. Registrar en `LLMTypeRegistry` y en el registry de modelos

Dos registros separados, uno obligatorio y uno opcional:

**5a. `LLMTypeRegistry.shared` (obligatorio)** — mapea el string
`model_type` de `config.json` a una función que decodifica la config e
instancia el modelo. Vive en
`Libraries/MLXLLM/LLMModelFactory.swift:23-88`. El helper `create(...)`
(línea 8-18) ya envuelve `JSONDecoder.json5().decode(...)` y, si tu
`Configuration` conforma a `ModelConfigurationValidating`, invoca
`validateModelConfiguration()` automáticamente.

Entradas reales de los dos modelos de referencia:

```swift
public static let shared: ModelTypeRegistry<LanguageModel> = .init(creators: [
    ...
    "mixtral": create(MixtralConfiguration.self, MixtralModel.init),   // línea 28
    ...
    "mamba2": create(Mamba2Configuration.self, Mamba2Model.init),      // línea 82
    ...
])
```

Añade tu propia línea siguiendo el mismo formato: `"<model_type_de_python>":
create(<TuConfiguration>.self, <TuModel>.init),`.

**5b. `ModelRegistry` / constante `ModelConfiguration` (opcional, solo si
quieres que un checkpoint concreto sea localizable por nombre desde
código)**. Esto **no es obligatorio para que el model_type funcione** —
tanto `mixtral` como `mamba2` están registrados en `LLMTypeRegistry.shared`
pero **no** tienen ninguna constante `ModelConfiguration` ni entrada en el
`all()` de `LLMRegistry` (confirmado: `grep -n "mixtral\|mamba2"
Libraries/MLXLLM/LLMModelFactory.swift` solo devuelve las dos líneas del
paso 5a). Añádelo solo si vas a escribir un test de integración descargable
(paso 7) o quieres que el modelo aparezca en apps/ejemplos por nombre.
Formato, si lo haces (`LLMModelFactory.swift:401-409`):

```swift
static public let jamba_3b_4bit = ModelConfiguration(
    id: "mlx-community/AI21-Jamba-Reasoning-3B-4bit",
    defaultPrompt: ""
)
```

y añádela al array de `private static func all()` (línea 411 en adelante)
si quieres que aparezca en el listado general.

Checklist:

- [ ] Línea añadida a `LLMTypeRegistry.shared` con el `model_type` exacto
  del `config.json` Python (no lo adivines, cópialo literal).
- [ ] Decide si necesitas una constante `ModelConfiguration` — solo si vas
  a escribir un test de integración descargable o exponerlo por nombre.
- [ ] Si añades la constante, decide si también entra en `all()`.

### 6. Tests con pesos sintéticos

Patrón real completo — `Tests/MLXLMTests/MixtralTests.swift` (68 líneas,
archivo completo citado porque es corto y es la plantilla a seguir):

```swift
@testable import MLXLLM

final class MixtralTests: XCTestCase {

    private func makeConfig() throws -> MixtralConfiguration {
        let json = """
            {
                "model_type": "mixtral",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_local_experts": 2,
                "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000000.0
            }
            """
        return try JSONDecoder().decode(MixtralConfiguration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = MixtralModel(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 32])   // [batch, sequence, vocab]
    }

    func testSanitizeStacksPerExpertWeightsIntoSwitchMLP() throws {
        // construye pesos por-experto sintéticos con las formas HF reales,
        // llama a model.sanitize(weights:), verifica claves consumidas y
        // forma del tensor apilado resultante.
        ...
    }
}
```

Los dos tests que debes escribir como mínimo, calcados de este ejemplo:

- [ ] **Forward pass produce la forma esperada de logits.** Config
  minúscula (vocab/hidden/capas de juguete, no el tamaño real del modelo)
  para que el test corra en CI sin GPU potente ni pesos descargados;
  pesos aleatorios/cero inicializados por el propio `init` del modelo — no
  hace falta cargar nada externo.
- [ ] **`sanitize(weights:)` transforma lo que dice transformar**, si tu
  modelo tiene `sanitize`. Construye un diccionario `[String: MLXArray]`
  sintético con las claves y formas exactas que produciría el checkpoint
  Python (mira el patrón A/B/C del paso 4 para saber qué construir), llama
  a `sanitize`, y comprueba tanto que las claves viejas desaparecieron como
  que las nuevas tienen la forma correcta.
- [ ] Si tu modelo usa un `Cache` no estándar (`MambaCache`, cache híbrida),
  añade también un test que corra dos forward passes consecutivos con el
  mismo cache y confirme que la segunda salida depende del estado
  acumulado (no solo del shape) — ver `Tests/MLXLMTests/Mamba2Tests.swift`
  para un ejemplo con `MambaCache`.
- [ ] Corre estos tests con:
  ```bash
  xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
  ```
  (`swift test` no funciona en este repo todavía — ver `CONTRIBUTING.md:25`).

### 7. Test de integración descargable (smoke check)

Estos tests **sí** descargan pesos reales de Hugging Face y corren
generación end-to-end; no corren en CI y viven en un proyecto Xcode
separado, `IntegrationTesting/IntegrationTesting.xcodeproj`, no en el
paquete SwiftPM (`CONTRIBUTING.md:31-53`).

Requisito previo: necesitas una constante `ModelConfiguration` con un `id`
de Hugging Face real (paso 5b) — sin eso no hay nada que descargar.

Patrón real, `IntegrationTesting/IntegrationTestingTests/CoherenceIntegrationTests.swift`:

```swift
private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct CoherenceIntegrationTests {
    @Test func gemma3n_E2B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.gemma3n_E2B_it_lm_4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }
    ...
}
```

`ChatSessionTests.planetsCoherence` (helper compartido en
`Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift:287-305`)
manda un prompt fijo ("List all the planets...") y verifica que la
respuesta contenga los ocho planetas — es un smoke check de coherencia, no
de exactitud numérica.

Checklist:

- [ ] Añade tu propio `@Test func <tuModelo>()` al `struct` de
  `CoherenceIntegrationTests.swift` (o crea un `@Suite` nuevo si el modelo
  necesita un flujo distinto — tool calling, thinking mode, etc., ver
  `ToolCallIntegrationTests.swift` como ejemplo de eso), apuntando a tu
  constante `ModelConfiguration`.
- [ ] Corre solo tu test nuevo primero (descarga puede tardar minutos):
  ```bash
  xcodebuild test \
    -project IntegrationTesting/IntegrationTesting.xcodeproj \
    -scheme IntegrationTesting \
    -destination 'platform=macOS' \
    -only-testing:IntegrationTestingTests/CoherenceIntegrationTests/<tuTest>
  ```
- [ ] Si la generación produce texto degenerado/repetitivo/vacío, el
  problema casi siempre está en `sanitize(weights:)` (paso 4) o en un
  desajuste de `RoPE`/normalización — no en el registry ni en los tests
  sintéticos, que ya habrían fallado antes.

---

## Plantillas

`scripts/porting/templates/` contiene esqueletos `.swift.template` (la
extensión `.template` es intencional — el paquete SwiftPM solo compila
`*.swift`, así que estos stubs nunca entran en el build). Cópialos, renombra
quitando `.template`, y sustituye los marcadores `<<PLACEHOLDER>>`:

- `Configuration.swift.template` — struct de configuración con el patrón
  `CodingKeys` + `init(from decoder:)` del paso 2.
- `Model.swift.template` — esqueleto de atención + decoder layer + inner +
  top-level model + `sanitize` + conformidad `LoRAModel`, siguiendo la forma
  de `Mixtral.swift`.
- `ModelTests.swift.template` — test de forward-pass-shape y test de
  `sanitize`, siguiendo la forma de `MixtralTests.swift`.

Estas plantillas son puntos de partida, no scaffolding automático: sigue
completándolas con la información real extraída en los pasos 1-4, no las
uses como sustituto de leer el `model.py` de Python.
