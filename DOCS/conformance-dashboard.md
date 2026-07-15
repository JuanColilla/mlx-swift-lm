# Conformance dashboard (scaffold)

> Estado: **diseño + generador de reporte, sin datos vivos**. Este documento
> responde al item 14 del backlog (`DOCS/tech-debt-and-research-backlog.md`):
> "Reporte en CI/manual de modelos que cargan, generan, usan tools, VLM y
> long-context por plataforma". No incluye resultados pass/fail — ejecutar la
> suite de integración descarga varios GB de checkpoints por modelo y no se
> ha corrido como parte de este trabajo.

## 1. Estructura de tests existente (dos capas)

El repo ya separa "tests rápidos con pesos sintéticos" de "tests de
integración con checkpoints reales". El dashboard de conformidad reutiliza
la segunda capa; no añade una tercera.

### Capa 1 — `Tests/MLXLMTests/`
Tests unitarios rápidos, pesos sintéticos, sin descarga de red. Se ejecutan
con `swift test` o `xcodebuild test -scheme mlx-swift-lm-Package`. Verifican
lógica (máscaras, KV cache, sampling, parsing de tool calls) pero **no**
prueban que un checkpoint real de Hugging Face cargue o genere texto
coherente.

### Capa 2 — `IntegrationTesting/IntegrationTestingTests/`
Proyecto Xcode independiente (`IntegrationTesting/IntegrationTesting.xcodeproj`,
target de test `IntegrationTestingTests`, scheme `IntegrationTesting`) con
tests que descargan pesos reales desde Hugging Face y ejercen capacidad real
del modelo. `xcodebuild -list -project IntegrationTesting/IntegrationTesting.xcodeproj`
solo lista schemes de librería de paquete (`BenchmarkHelpers`,
`IntegrationTestHelpers`, `MLXLLM`, `MLXLMCommon`, `MLXVLM`, `MLXEmbedders`,
`MLXHuggingFace`, `IntegrationTesting`); no hay un scheme "solo tests"
separado — el target ejecutable de xctest (`IntegrationTestingTests`) se
dispara vía el scheme `IntegrationTesting`.

Archivos reales en esa capa (ninguno inventado, todos confirmados por
inspección directa):

- `CoherenceIntegrationTests.swift` — `@Suite(.serialized) struct
  CoherenceIntegrationTests`, 18 `@Test` (uno por arquitectura: `bitnet_b1_58_2B`,
  `exaone_4_0_1_2B`, `gemma3_1B_qat`, `gemma3n_E2B`, `gemma4_e2b`, `glm4_9B`,
  `granite3_3_2B`, `granite4_0_H_tiny`, `jamba_3B_4bit`, `lfm2_1_2B`,
  `llama3_2_1B`, `mistral_7B`, `olmo2_7B`, `olmoe_1B_7B`, `phi3_5`,
  `qwen3_1_7B`, `qwen3_5_2B`, `smollm3_3B`). Cada test carga el `ModelContainer`
  vía `IntegrationTestModels.llmContainer(for:)` y llama a
  `ChatSessionTests.planetsCoherence(container:)` — prueba **carga** y
  **generación coherente** en una sola pasada.
- `ToolCallIntegrationTests.swift` — `@Suite(.serialized) struct
  ToolCallIntegrationTests`, tests por familia (`lfm2FormatAutoDetection`,
  `lfm2EndToEnd`, `glm4FormatAutoDetection`, `glm4EndToEnd`,
  `mistral3FormatAutoDetection`, `mistral3EndToEnd`, `mistral3MultiTool`,
  `nemotronFormatAutoDetection`, `nemotronEndToEnd`, `nemotronMultiTool`,
  `qwen35FormatAutoDetection`, `qwen35EndToEnd`, `qwen35MultiTool`). Prueba
  **detección de formato de tool call** y **generación end-to-end con tool
  calling**, incluyendo multi-tool para mistral3/nemotron/qwen35.
- `BidirectionalMasksTests.swift` — 8 `@Test` sueltos (no agrupados en
  `@Suite`) que comparan `createBidirectionalMask(...)` contra fixtures
  `.safetensors` descargados del dataset HF `angelsbrood/gemma4-mtp-fixtures`
  (revisión fijada). Cubre máscaras bidireccionales y con sliding-window
  attention (SWA) para distintas combinaciones de `queryLen`/`kvLen`/`window`.
  Es la única cobertura de integración relacionada con máscaras — no hay
  suite de "long-context" dedicada (ver §4, gap explícito).
- `Gemma4AssistantDraftModelIntegrationTests.swift` — `@Suite(.serialized)
  struct Gemma4AssistantIntegrationTests`, 3 `@Test`
  (`testGemma4AssistantConfigurationDecodesRealCheckpoint`,
  `testRung1WeightsLoadFrom31BCheckpoint`,
  `testRung2And3ForwardMatchesFixture31BCase01`). Prueba carga y forward-pass
  del checkpoint drafter real de 31B para el pipeline MTP/speculative de
  Gemma4. Importa `MLXVLM` porque el drafter usa esa arquitectura interna,
  **no** ejercita capacidad VLM (sin imagen/vídeo de entrada).
- `MTPAcceptanceRateTests.swift` — 1 `@Test`
  (`testAcceptanceRateFloor64TokenBlock4Temp0`), **`.disabled(...)`**: mide
  acceptance rate ≥ 0.30 en generación target+drafter de 64 tokens. Cuerpo
  retenido pero diferido a un PR de seguimiento; hoy no aporta señal.
- `MTPQuantizationOnsetTests.swift` — 1 `@Test`
  (`testMTPMidGenerationKVQuantizationCompletesWithoutCrash`), también
  **`.disabled(...)`**: verificaría el fallback a modo passthrough cuando el
  KV cache se cuantiza a mitad de generación (R13). Diferido igualmente.
- `MTPDrafterModelFactoryIntegrationTests.swift`,
  `MTPIteratorEndToEndDiagnosticTests.swift`, `MTPRung4TokenParityTests.swift`
  — suites adicionales de paridad/diagnóstico para el pipeline MTP
  (`Rung4TokenParityTests`, `MTPIteratorEndToEndDiagnosticTests`, tests
  sueltos de factory) que comparan generación con/sin drafter, determinismo
  y paridad de tokens contra fixtures.

Ninguno de estos archivos contiene un test que cargue una imagen o vídeo de
entrada real contra un `VLMModelFactory`/`ModelContainer` de VLM — la
cobertura VLM real (carga + generación multimodal) no existe todavía en esta
capa (gap explícito, ver §4).

## 2. Esquema propuesto del reporte

Una fila por combinación `(modelo, capacidad, plataforma)`:

| Columna | Tipo | Descripción |
|---|---|---|
| `model_id` | string | ID de Hugging Face del checkpoint (p. ej. `mlx-community/Llama-3.2-1B-Instruct-4bit`). |
| `architecture` | string | Familia de arquitectura tal como la registra `LLMTypeRegistry`/`VLMTypeRegistry` (p. ej. `llama`, `gemma3n`, `qwen3`). |
| `capability` | enum | Una de: `loads`, `generates`, `tools`, `vlm`, `long_context`, `speculative`. |
| `platform` | enum | `macOS`, `iOS-simulator`, `iOS-device` (las dos últimas no ejecutables hoy, ver §4). |
| `status` | enum | `pass`, `fail`, `not_run`. `not_run` es el valor por defecto para toda fila sin ejecución registrada — nunca se infiere `pass`. |
| `last_checked` | date (ISO 8601) | Fecha de la última ejecución que produjo este `status`. |
| `source_test` | string | Identificador `Suite/testName` o `testName` (sin suite) que produjo el resultado, para trazabilidad directa al archivo `.swift`. |

Notas de diseño:
- `capability` está acotado a las categorías que el backlog pide explícitamente
  (loads/generates/tools/VLM/long-context), más `speculative` porque ya existe
  cobertura real (MTP) que no encaja en las otras cinco y sería una pérdida de
  información omitirla.
- Un mismo `source_test` puede alimentar dos filas de `capability` distintas
  cuando un test combina responsabilidades (p. ej. `CoherenceIntegrationTests`
  prueba `loads` y `generates` en la misma ejecución) — el generador debe
  emitir una fila por capacidad, no una fila por test.
- `status = not_run` es el estado inicial de cualquier combinación
  `(modelo, capacidad, plataforma)` para la que no exista todavía un test de
  integración — por ejemplo, toda fila con `capability = vlm` hoy, y toda fila
  con `platform != macOS`.

## 3. Mapeo esquema → suites reales

| `capability` | `source_test` (real) | Notas |
|---|---|---|
| `loads` + `generates` | `CoherenceIntegrationTests/<modelName>` (18 tests, §1) | Un test cubre ambas capacidades; el generador debe emitir 2 filas por test. |
| `tools` | `ToolCallIntegrationTests/<family><Variant>` (13 tests, §1) | Incluye variantes `FormatAutoDetection`, `EndToEnd` y `MultiTool` según familia. |
| `speculative` | `Gemma4AssistantIntegrationTests/*`, `Rung4TokenParityTests/*`, `MTPIteratorEndToEndDiagnosticTests/*`, tests sueltos en `MTPDrafterModelFactoryIntegrationTests.swift` | Cobertura real de carga de drafter + paridad de tokens. `MTPAcceptanceRateTests` y `MTPQuantizationOnsetTests` existen pero están `.disabled` — deben mapear a `status = not_run`, nunca inferirse como `pass`. |
| `long_context` (parcial) | `BidirectionalMasksTests.swift` (8 tests sueltos) | Cubre corrección de máscaras (incluida SWA), no rendimiento ni comportamiento end-to-end con contextos largos reales. Tratar como cobertura parcial, no como sustituto de una suite de long-context. |
| `vlm` | — (sin test de integración hoy) | Fila explícitamente `not_run` para todo modelo VLM registrado en `VLMTypeRegistry` hasta que exista una suite equivalente a `CoherenceIntegrationTests` pero con entrada de imagen/vídeo. |

## 4. Fuera de alcance de esta primera versión (trabajo futuro)

Explícitamente **no implementado** en este scaffold:

- **Ejecución automática en CI.** El script de la sección siguiente es
  disparo manual; conectarlo a un workflow (con caché de checkpoints entre
  runs, dado el tamaño en GB) es trabajo futuro.
- **Cobertura real de dispositivo/simulador iOS.** Hoy `platform` solo tiene
  datos reales para `macOS`; correr la misma suite contra un simulador o
  device farm de iOS requiere una configuración de destino distinta en
  `xcodebuild` y no se ha validado aquí.
- **Suite de integración VLM.** No existe todavía un equivalente de
  `CoherenceIntegrationTests` que cargue imagen/vídeo real contra
  `VLMModelFactory`. Crearla es prerequisito para que la columna `vlm` del
  esquema tenga alguna fila en `pass`/`fail`.
- **Suite dedicada de long-context.** `BidirectionalMasksTests.swift` prueba
  corrección de máscara, no comportamiento end-to-end (calidad de
  generación, uso de memoria) con prompts largos reales. Ver también el
  item 4 del backlog ("Recetas oficiales de long-context").
- **Publicación/visualización del reporte** (dashboard web, badge, etc.). El
  script solo emite una tabla Markdown a stdout/archivo; renderizarla en un
  sitio o pegarla en un PR es trabajo futuro.
- **Habilitar `MTPAcceptanceRateTests` y `MTPQuantizationOnsetTests`.** Ambos
  tests están `.disabled` con el trabajo diferido documentado en el propio
  archivo; el dashboard debe reflejar su estado real (`not_run`) hasta que se
  activen, no ocultarlos ni inventarles un resultado.

## 5. Cómo generar un reporte

Ver `scripts/conformance-dashboard/generate-report.sh`. Ejecutarlo dispara
la suite `IntegrationTesting` completa, lo que descarga checkpoints reales
(varios GB) — leer el header del script antes de correrlo.
