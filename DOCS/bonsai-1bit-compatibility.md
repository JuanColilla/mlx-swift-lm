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
| `mlx-swift` | `ml-explore/mlx-swift@09051ed` | `JuanColilla/mlx-swift@5e27a4cb2604599c72615cf058e09801c123b831` |
| MLX core | `ml-explore/mlx@ce45c525` | `JuanColilla/mlx@583322c8edefd9c2fdee45619fc6bc6ef949e962` |
| `mlx-c` | oficial | `0726ca922fc902c4c61ef9c27d94132be418e945` |

El `Package.swift` de este repositorio usa el SHA completo de `mlx-swift`.
No depende de una rama mutable de Prism.

## Alcance del port

- cuantizacion y dequantizacion affine de 1 bit en CPU y Metal;
- QMM, QMV, producto exterior y kernels NAX y no NAX;
- proteccion de rutas QMV rapidas incompatibles;
- manejo de la cola QMV cuando `K` no esta alineado con el bloque;
- menor presion de registros en el QMV rapido de 1 bit;
- desactivacion de NAX en GPU gen-17, donde GEMM/QMM puede producir resultados
  incorrectos, con una prueba de regresion ejecutada en M5 Max;
- artefactos C/Metal regenerados mediante `tools/update-mlx.sh`;
- pruebas Swift para la API `MLX.quantized`, `QuantizedLinear`, QMM/QMV y
  regresion de 2, 3, 4, 5, 6 y 8 bits.

El port toma como referencia `ml-explore/mlx#3161` y los commits affine-only
de Prism sobre la misma base. No incluye la extension simetrica de 1/2 bits
sin tensor `bias`, ni workarounds de compilacion independientes.

## Contraste con el fork de PrismML

La rama por defecto `PrismML-Eng/mlx-swift:prism@e40e0a57` y su variante
`v0.31.6_prism@563961df` apuntan ambas al core `d90771c8`. Nuestra solucion
parte de los mismos dos cambios affine imprescindibles de Prism, pero no copia
su rama completa:

| Aspecto | PrismML | Fork propio |
| --- | --- | --- |
| Soporte affine 1-bit | Incluido en `4efbcacb` y `b9effaf6` | Portado como `14a1aa3a` y `1e8cb398` |
| Colas QMV no alineadas | No esta en el core `d90771c8` fijado por el wrapper | Incluye las correcciones posteriores de `mlx#3161` |
| Ocupacion de QMV 1-bit | Disponible en ramas posteriores del core Prism, pero no en el submodulo fijado por el wrapper | Incluida en `52ad4c57` |
| Formato simetrico sin `bias` | Incluido para checkpoints adicionales de Prism | Excluido; Bonsai 27B publica `scale + bias` |
| NAX en M5/gen-17 | La variante `v0.31.6_prism` parchea el header generado y anade una regresion | Portado al core en `583322c8`, regenerado y cubierto por una prueba Swift |
| Integracion SwiftPM | El README propone `branch: "prism"` y conserva parches auxiliares | Revision propia inmutable, sin parches manuales |
| Pruebas Swift 1-bit | No anade casos 1-bit a `QuantizationTests` | Cubre round-trip, `QuantizedLinear`, QMM/QMV, colas y bits existentes |

Se mantiene fuera el workaround `FMT_CONSTEVAL`, porque es independiente del
soporte 1-bit y la compilacion con Xcode 27 funciona sin el. Tambien quedan
fuera los cambios de CI y las extensiones de kernels que el wrapper Prism
publicado no consume mediante su submodulo fijado.

## Validacion realizada

La regeneracion se ejecuto repetidamente y produjo el mismo diff binario
(`22ad17070b971ecf18c57d5993ad2f63176c680d86db7a778c0d6dc9918efa6f`).

Comando de pruebas:

```shell
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild -quiet test \
  -scheme mlx-swift-Package \
  -destination 'platform=macOS' \
  -derivedDataPath /private/tmp/mlx-swift-bonsai-derived \
  -only-testing:MLXTests/QuantizationTests \
  -only-testing:MLXTests/NAXGateRegressionTests
```

Resultado: 13 pruebas aprobadas, 0 fallos y 0 omisiones en un MacBook Pro
M5 Max arm64 con Xcode 27.0. La compilacion incluyo el core C++, los shaders
Metal y el wrapper Swift.

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

La compilacion completa del esquema `mlx-swift-lm-Package` tambien finalizo
correctamente con la revision fijada del fork.

## Validacion fisica confirmada y mediciones pendientes

El usuario confirmo el 2026-07-15 la validacion fisica de Bonsai en el iPhone
objetivo. Esta confirmacion prueba el camino de carga/generacion que motivaba el
port, pero no se usan cifras no registradas para inferir memoria o rendimiento.

Queda como trabajo de caracterizacion, no como bloqueo de compatibilidad:

- cubrir hardware fisico gen-18 con NAX y hardware no NAX; el M5 Max gen-17
  queda correctamente en la ruta no NAX;
- registrar pico de memoria, tiempo de carga y tokens por segundo;
- probar varios turnos, descarga/recarga y cancelacion;
- fijar esta misma revision de `mlx-swift` en la dependencia directa de
  MLXHub para que SwiftPM resuelva una unica identidad.

## Rollback

El rollback consiste en restaurar la dependencia oficial anterior de
`mlx-swift` y su resolucion exacta en el consumidor. El modelo debe volver a
bloquearse antes de llamar a MLX mientras el runtime activo no admita 1 bit.
