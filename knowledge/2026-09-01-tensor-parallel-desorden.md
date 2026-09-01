# Estado disperso del tensor-parallel y punto de retoma

**Fecha:** 2026-09-01
**Estado:** Aparcado deliberadamente hasta después de la 2.1.0
**Repositorios implicados:** `JuanColilla/mlx-swift-lm` (este) y `JuanColilla/mlx-swift`
**Objetivo de este documento:** dejar en un único sitio localizable todo lo que se investigó
sobre tensor-parallel el 2026-09-01, para no tener que repetir la arqueología cuando se
retome. No hay código nuevo en esta rama — es aparcamiento, no desarrollo.

## 1. Resumen ejecutivo

Tensor-parallel para MLXHub no está implementado en `develop`. Hubo un intento real, pero
quedó repartido sin conexión entre tres sitios:

1. Un stub en este repo (`TensorParallel.swift`, añadido 2026-08-03 en `66c1e2c`, descartado
   explícitamente el 2026-08-04 según
   [`2026-08-04-develop-branch-unification-report.md`](2026-08-04-develop-branch-unification-report.md)
   línea 24: "Descartar... el stub de tensor parallel". Correcto en su momento: sin las piezas
   de abajo era código muerto con `fatalError`.
2. Una integración a nivel de modelo en el fork de terceros `N1k1tung/mlx-swift-lm` (remote
   `inferring`), ramas `ios-distrib-tensor`/`ane-backend` — implementación completa por
   arquitectura (Llama, DeepseekV3, Qwen3, Qwen3MoE, LFM2, GPTOSS, GLM4MoELite) con sharding
   estilo Megatron (Q/K/V y gate/up column-parallel, O-proj y down-proj row-parallel con
   all-reduce). **No compila tal cual, aislada**: llama a `shardLinearLeafs`/
   `AllToShardedLinear`/`ShardedToAllLinear`/`QuantizedAllToShardedLinear`/
   `QuantizedShardedToAllLinear`, símbolos que no vienen en su propio `Package.swift` (esa
   rama no depende de nuestro `mlx-swift`) — pero sí están disponibles hoy mismo si se porta
   sobre `develop`, ver el punto 3.
3. **El puerto Swift que falta ya existe, y ya es una dependencia activa de este repo** — no
   solo de otro repo sin conectar. `Package.swift` fija `mlx-swift` por `revision:
   8f74acc...`, y ese commit vive en `JuanColilla/mlx-swift`, rama
   `feature/distributed-group-port`, e incluye `Source/MLXNN/Distributed.swift` (739 líneas):
   `AllToShardedLinear`, `ShardedToAllLinear`, sus variantes cuantizadas y
   `shardLinearLeafs`, exactamente lo que el punto 2 necesita. **Corrección sobre la primera
   versión de este documento:** al comprobar si esas clases existían en algún sitio se usó
   `grep ... | head -5` sobre el checkout ya resuelto en `/tmp/mlx-swift-lm-build`, y el
   recorte cortó la salida antes de llegar a `Source/MLXNN/`, dejando solo los aciertos en el
   `.py` vendorizado de referencia. El archivo Swift estaba ahí desde el principio. Hoy ese
   código ya se compila en cada build de `mlx-swift-lm` — nadie lo llama, pero no es un
   `fatalError` como el stub del punto 1, es una librería completa sin usar.

   El comentario junto al pin en `Package.swift` (línea 64) confirma que `8f74acc` se eligió
   por Bonsai 1-bit + `DistributedGroup` + el diferido de errores Metal — el sharding vino de
   propina, no fue el motivo del pin, y por eso pasó desapercibido.

Ninguna de las tres piezas se pierde con este aparcamiento: quedan referenciadas de forma
permanente (tags propios, no remote-tracking refs de terceros que puedan desaparecer).

## 2. Referencias permanentes creadas hoy

| Referencia | Repo | Apunta a | Contenido |
|---|---|---|---|
| `archive/inferring-ios-distrib-tensor` | mlx-swift-lm | `7756fe4` | Base `ios-distrib`/`update` + 84 commits de tensor-parallel por modelo |
| `archive/inferring-ane-backend` | mlx-swift-lm | `16eb6d5` | Lo anterior + 2 commits sueltos (apuntar a rama ANE de mlx-swift, extender Llama para Nanbeige) sin relación con tensor-parallel |

Estos tags están empujados a `fork` (no a `inferring`, que es de solo lectura y no
controlamos). `inferring/ios-distrib` e `inferring/update` son la misma punta
(`0`/`0` de diferencia) y quedan ya cubiertas como ancestro de los dos tags de arriba, así
que no se archivaron aparte.

`mlx-swift`, rama `feature/distributed-group-port`, ya es nuestra (empujada a `fork`,
al día con `fork/main`) — no necesitó tag de archivo, solo quedó documentada como candidata
única en su propio `AGENTS.md`.

## 3. Por qué no se integra ahora

- El usuario quiere retomarlo después de publicar la 2.1.0, no en paralelo.
- No es cuestión de "traer una dependencia que falta": la primitiva de sharding ya está
  resuelta y compilada en cada build actual de `mlx-swift-lm`. Lo que falta es exclusivamente
  el trabajo de modelo: rebasar el diff de `archive/inferring-ios-distrib-tensor` sobre el
  `develop` actual (que ha divergido mucho desde el punto de fork de esa rama, 2026-02-12 —
  ver la fusión oficial de hoy mismo en `f078bf4` como ejemplo de cuánto ha cambiado
  Qwen3.5/Qwen3Next desde entonces) y verificar el sharding de pesos cuantizados con datos
  reales, no solo que compile.
- Sigue siendo trabajo nuevo de alcance medio, pero más pequeño de lo que parecía: sin puerto
  de primitivas pendiente, es directamente portar ~500 líneas de diff de modelo y probarlas.

## 4. Plan de retoma (checklist para cuando se reanude)

1. ~~Decidir el pin~~ **Hecho el 2026-09-02** (fuera de esta rama, en `develop`): el pin subió
   de `8f74acc` a `6589376` (`v0.31.6-fork.3`) solo por el fix oficial de
   `StreamOrDevice.stream(_:)` (#463); `mlx-swift-lm` se retagueó como `v3.31.4-fork.3`.
   `MLXHub` se actualiza aparte (fuera del alcance de este repo). Sigue sin haber nada de
   tensor-parallel en el pin: sigue viajando sin usarse, ahora desde `6589376` en vez de
   `8f74acc`. Si `develop` avanza más antes de retomar esto, revisa de nuevo si el pin sigue al
   día — este documento no se actualiza solo.
2. Sobre `develop` actual (no sobre el `wip/tensor-parallel` de esta fecha, que solo sirve de
   ancla), portar el diff de `archive/inferring-ios-distrib-tensor` contra su base
   (`718877e`) modelo a modelo, arquitectura por arquitectura, llamando directamente a
   `AllToShardedLinear`/`ShardedToAllLinear`/`shardLinearLeafs` — ya están disponibles vía
   `import MLXNN`, no hace falta escribirlas. `git diff
   718877e29a267ee152d24e27a75721b102adf4b8 archive/inferring-ios-distrib-tensor` da el punto
   de partida exacto.
3. Verificar que `TransformerLayer` (ya existe en `develop`, `LanguageModel.swift:446`) sigue
   siendo el protocolo que ambos mecanismos —pipeline-parallel actual y este tensor-parallel—
   comparten; es la pieza que permite componer los dos (2D parallelism) en vez de que
   compitan.
4. Medir antes de dar por bueno: tensor-parallel exige un all-reduce por cada bloque de
   atención y cada MLP, no solo en los límites de pipeline. Sobre Bonjour/TCP entre iPhones
   puede ser network-bound. Prototipar con 2 nodos reales antes de portar las 7 arquitecturas.

## 5. Tags de variante publicados el 2026-09-01/02

Además de las referencias de archivo (§2), se publicó un tag `vX.X.X-fork.N` normal en cada
repo con el estado del día, siguiendo la convención de `AGENTS.md`:

| Repo | Tag | Commit | Base oficial |
|---|---|---|---|
| `mlx-swift-lm` | `v3.31.4-fork.2` | `c25a192` (`develop`) | `3.31.4` |
| `mlx-swift` | `v0.31.6-fork.3` | `6589376` (`feature/distributed-group-port`) | `0.31.6` |
| `mlx-swift-lm` | `v3.31.4-fork.3` | `01d4745` (`develop`, 2026-09-02) | `3.31.4` |

`v3.31.4-fork.3` supersede a `.fork.2`: sube el pin de `mlx-swift` de `8f74acc` a `6589376`
(`v0.31.6-fork.3`), así que `mlx-swift-lm` y `mlx-swift` vuelven a estar en sus respectivos
últimos tags mutuamente consistentes. `MLXHub` necesita el mismo bump por su cuenta (lo hace
el usuario, fuera de este repo).

Ninguno de los tres incluye trabajo de tensor-parallel nuevo: son la fusión oficial del día más
la documentación de ramas. `v0.31.6-fork.3` es la punta actual de la rama candidata a
tensor-parallel — es el commit al que apuntaría el pin si se decide subirlo (§4.1).

## 6. Fuera de alcance de esta limpieza (no tocado, no relacionado)

- `feature/bonsai-1bit-support` (mlx-swift) — cuantización afín de 1 bit, 2 commits propios
  sobre `fork/main`. Trabajo propio legítimo, sin relación con tensor-parallel.
- `fork/materialized-array` (mlx-swift) — 11 commits propios divergidos, 25 detrás de
  `origin/materialized-array`. Tema distinto (materialización de arrays), no revisado a fondo
  hoy; no se ha tocado ni borrado.
- El worktree `fix/mesh-distributed-inference` de este repo — trabajo activo de otro tema
  (tolerancia de `safetensorWeightURLs`/`loadArraysAndMetadata` a recortes, TD-032), no forma
  parte del desorden de tensor-parallel.
