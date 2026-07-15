# Investigación cerrada — `Gemma4ChunkedPrefillTests.chunkSizeInvariance` en M5

Fecha de actualización: 2026-07-15.

## Resultado actualizado

El fallo observado durante la investigación inicial ya no se reproduce con el
árbol destinado a release. Al integrar `main` (`f08d903`), el paquete pasa a
resolver `JuanColilla/mlx-swift@5e27a4cb2604599c72615cf058e09801c123b831`,
que desactiva NAX en GPU gen-17/M5 por resultados incorrectos conocidos.

Con esa revisión:

- `Gemma4ChunkedPrefillTests.chunkSizeInvariance` pasa para sus cuatro valores
  parametrizados (`3`, `5`, `8` y `16`);
- la suite completa de `mlx-swift-lm-Package` pasa sin fallos;
- no se aplicó ningún cambio a `RotatingKVCache` para obtener ese resultado.

Por tanto, **no hay evidencia para corregir `RotatingKVCache` como parte de
este release**. La hipótesis anterior sobre la transición
`updateConcat` → `updateInPlace` no quedó confirmada mediante instrumentación y
el test deja de fallar al cambiar únicamente al runtime M5-safe.

La inferencia más fuerte es que la divergencia numérica provenía de la ruta NAX
en gen-17. Esto no se presenta como causalidad demostrada: confirmarla de forma
aislada requeriría una comparación A/B entre ambos runtimes manteniendo idéntico
el resto del árbol.

## Evidencia de verificación

Árbol verificado:

- `mlx-swift-lm`: integración de `docs/improvement-backlog-review` con
  `main@f08d903`;
- `mlx-swift`: `5e27a4cb2604599c72615cf058e09801c123b831`;
- plataforma: MacBook Pro M5 Max arm64, macOS/Xcode 27.

Comando:

```bash
xcodebuild test \
  -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation
```

Resultado estructurado del `.xcresult`: 385 tests lógicos aprobados, 0 fallos
y 0 omisiones. Xcode contabiliza además 417 ejecuciones al expandir parámetros.

## Contexto histórico

En el baseline anterior (`f97da4e` con la dependencia oficial de `mlx-swift`)
el test fallaba de forma determinista para los cuatro tamaños de chunk. La
investigación localizó una transición potencialmente delicada entre las rutas
multi-token y single-token de `RotatingKVCache`, pero no llegó a observar una
divergencia interna que probase esa hipótesis.

Ese análisis se conserva como contexto útil, pero no como causa raíz. Haber
modificado la caché basándose sólo en él habría introducido riesgo en todos los
modelos sliding-window sin resolver el origen numérico demostrado por el cambio
de runtime.

## Criterio para futuras regresiones

Si `chunkSizeInvariance` vuelve a fallar:

1. confirmar primero la revisión efectiva de `mlx-swift` y si NAX está activo;
2. reproducir A/B con runtime gen-17-safe y runtime oficial sobre el mismo
   commit de `mlx-swift-lm`;
3. instrumentar `RotatingKVCache` sólo si el fallo persiste con NAX desactivado;
4. no aceptar una corrección de caché sin el test parametrizado en verde y una
   verificación end-to-end con un modelo sliding-window real.
