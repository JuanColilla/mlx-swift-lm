# Conformance dashboard

> Estado: **implementado y verificable offline**. El dashboard genera JSON
> estable y Markdown desde resultados reales de `xcresult`; los checkpoints
> con red siguen siendo una ejecución manual y explícita.

## Componentes

- `scripts/conformance-dashboard/conformance-manifest.json`: mapeo revisable de
  `source_test` a `model_id`, revisión, arquitectura, capacidad y plataformas.
- `scripts/conformance-dashboard/report.py`: parser del JSON producido por
  `xcresulttool`, agregación y render JSON/Markdown.
- `scripts/conformance-dashboard/generate-report.sh`: ejecución opcional de
  Xcode y extracción del result bundle.
- `scripts/conformance-dashboard/fixtures/xcresult-tests.json`: fixture offline
  con casos pass/fail/disabled y un fallo no mapeado.
- `scripts/conformance-dashboard/tests/test_report.py`: pruebas del parser,
  validación del manifiesto y estabilidad de ambos formatos.

El JSON de salida tiene `schema_version = 1`. Cada check conserva exactamente
uno de cuatro estados: `pass`, `fail`, `not_run` o `disabled`. No se convierte
una ausencia en éxito: un check habilitado que no aparece o queda skipped en
el result bundle se mantiene `not_run`; solo un check desactivado expresamente
en el manifiesto conserva `disabled` y su razón.

Un test puede mapear a varias capacidades. Por ejemplo, la prueba real de
SmolVLM2 produce filas distintas para `loads` y `vlm`, pero ambas apuntan al
mismo `source_test` y a la misma revisión fijada.

## Fallos y trazabilidad

El proceso termina con código distinto de cero si ocurre cualquiera de estas
condiciones:

- Xcode devuelve error;
- un check mapeado tiene estado `fail`;
- cualquier test no mapeado falla.

Los fallos no mapeados se incluyen en `unmapped_failures` y en una sección
propia del Markdown. Por tanto, ampliar el target de integración sin actualizar
el manifiesto no puede ocultar una regresión.

## Cobertura añadida

`ConformanceIntegrationTests.swift` contiene dos pruebas de red, deshabilitadas
por defecto mediante `MLX_RUN_CONFORMANCE_NETWORK`:

| Suite | Checkpoint fijado | Capacidad |
|---|---|---|
| `VLMConformanceIntegrationTests/smolVLM2ImageGeneration` | `mlx-community/SmolVLM2-500M-Video-Instruct-mlx@fa57db46815177fbdfd65cc85a2b3416a8332268` | carga + generación desde imagen sintética |
| `LongContextConformanceIntegrationTests/qwen35LongContextSentinel` | `mlx-community/Qwen3.5-0.8B-4bit@da28692b5f139cb0ec58a356b437486b7dac7462` | carga + recuperación de sentinel tras un prompt largo |

Los checks FastVLM, Idefics3 multi-image y Gemma4 vídeo son tests
explícitamente `disabled`, no cuerpos que retornan silenciosamente. Siguen en el
manifiesto para que el reporte muestre la deuda sin atribuir un `pass`.

## Plataformas y destinos

El proyecto de integración declara únicamente `macosx` e `iphoneos`. No declara
`iphonesimulator`, porque MLX no ofrece un backend de ejecución válido en iOS
Simulator.

macOS, sin descargar checkpoints:

```sh
scripts/conformance-dashboard/generate-report.sh
```

macOS, habilitando las dos pruebas de checkpoints fijados:

```sh
scripts/conformance-dashboard/generate-report.sh --include-network
```

iOS físico:

```sh
scripts/conformance-dashboard/generate-report.sh \
  --platform iOS-device \
  --destination 'platform=iOS,id=<DEVICE_UDID>' \
  --development-team <TEAM_ID> \
  --include-network
```

La ejecución iOS necesita dispositivo conectado/confiable, firma válida, equipo
de desarrollo y los entitlements de memoria que requiera el modelo. Que el
proyecto resuelva para `iphoneos` no demuestra conformidad en hardware: solo un
result bundle generado en el dispositivo puede producir `pass`.

También se puede regenerar el reporte desde un result bundle existente, sin
volver a ejecutar Xcode:

```sh
scripts/conformance-dashboard/generate-report.sh \
  --result-bundle /path/to/results.xcresult \
  --platform macOS
```

Los artefactos se escriben por defecto en `.conformance/` y no se versionan.

## Validación offline

```sh
python3 -m unittest discover -s scripts/conformance-dashboard/tests -v
bash -n scripts/conformance-dashboard/generate-report.sh
```

El fixture contiene intencionadamente fallos; invocar el CLI contra él debe
generar ambos formatos y devolver `1`.

## Pendiente de checkpoint y dispositivo físico

- FastVLM: medir tokens visuales reales tras la vision tower y memoria pico.
- Idefics3 puro: confirmar la semántica multi-image frente a SmolVLM.
- Gemma4/Gemma4Unified: verificar si el vídeo es alcanzable desde
  `UserInput.videos` o requiere cableado adicional.
- Qwen3.5 VLM: confirmar el `processor_class` del checkpoint concreto.
- iOS: ejecutar los checks habilitados en hardware real y conservar el
  result bundle; hasta entonces el estado correcto es `not_run`.

CI con caché de checkpoints y publicación web del Markdown siguen fuera de
este bloque. El formato estable ya permite añadirlos sin cambiar el contrato de
datos.
