# Compatibilidad de Bonsai 1-bit

Fecha: 2026-07-15

## Resultado

`mlx-swift-lm` resuelve una revision exacta del fork `JuanColilla/mlx-swift`
que admite cuantizacion affine de 1 bit. El camino de carga existente puede
usar configuraciones con `bits: 1` y `group_size: 128` sin que el core rechace
la operacion en `mlx_quantize`.

## Revisiones reproducibles

| Componente | Base | Revision integrada |
| --- | --- | --- |
| `mlx-swift-lm` | `main` actualizada en `bd02744` | rama `feature/bonsai-1bit-compatibility` |
| `mlx-swift` | `ml-explore/mlx-swift@09051ed` | `JuanColilla/mlx-swift@15e907bf8222c6401659dce5e26a9cc5fd45abc4` |
| MLX core | `ml-explore/mlx@ce45c525` | `JuanColilla/mlx@52ad4c570302d89c84ccb6623eeadbf2e74221f2` |
| `mlx-c` | oficial | `0726ca922fc902c4c61ef9c27d94132be418e945` |

El `Package.swift` de este repositorio usa el SHA completo de `mlx-swift`.
No depende de una rama mutable de Prism.

## Alcance del port

- cuantizacion y dequantizacion affine de 1 bit en CPU y Metal;
- QMM, QMV, producto exterior y kernels NAX y no NAX;
- proteccion de rutas QMV rapidas incompatibles;
- manejo de la cola QMV cuando `K` no esta alineado con el bloque;
- menor presion de registros en el QMV rapido de 1 bit;
- artefactos C/Metal regenerados mediante `tools/update-mlx.sh`;
- pruebas Swift para la API `MLX.quantized`, `QuantizedLinear`, QMM/QMV y
  regresion de 2, 3, 4, 5, 6 y 8 bits.

El port toma como referencia `ml-explore/mlx#3161` y los commits affine-only
de Prism sobre la misma base. No incluye la extension simetrica de 1/2 bits
sin tensor `bias`, ni workarounds de compilacion independientes.

## Validacion realizada

La regeneracion se ejecuto repetidamente y produjo el mismo diff binario
(`20104d5429af18aae104fb5155dfaf10a9f0f1a9e5e32ad35f15cd4b15c71a9e`).

Comando de pruebas:

```shell
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild -quiet test \
  -scheme mlx-swift-Package \
  -destination 'platform=macOS' \
  -derivedDataPath /private/tmp/mlx-swift-bonsai-derived \
  -only-testing:MLXTests/QuantizationTests
```

Resultado: 12 pruebas aprobadas, 0 fallos y 0 omisiones en Mac arm64 con
Xcode 27.0. La compilacion incluyo el core C++, los shaders Metal y el wrapper
Swift.

Este paquete incluye ademas `BonsaiOneBitCompatibilityTests`, que verifica el
round-trip affine y `quantizedMM` con `bits: 1` y `groupSize: 128`. La prueba
dirigida aprobo con 1 prueba, 0 fallos y 0 omisiones:

```shell
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild -quiet test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -derivedDataPath /private/tmp/mlx-swift-lm-bonsai-derived \
  -skipPackagePluginValidation \
  -only-testing:MLXLMTests/BonsaiOneBitCompatibilityTests
```

## Validacion pendiente antes de distribuir

- cargar y generar con Bonsai 27B en el iPhone del incidente;
- cubrir hardware NAX y no NAX fisico;
- registrar pico de memoria, tiempo de carga y tokens por segundo;
- probar varios turnos, descarga/recarga y cancelacion;
- fijar esta misma revision de `mlx-swift` en la dependencia directa de
  MLXHub para que SwiftPM resuelva una unica identidad.

## Rollback

El rollback consiste en restaurar la dependencia oficial anterior de
`mlx-swift` y su resolucion exacta en el consumidor. El modelo debe volver a
bloquearse antes de llamar a MLX mientras el runtime activo no admita 1 bit.
