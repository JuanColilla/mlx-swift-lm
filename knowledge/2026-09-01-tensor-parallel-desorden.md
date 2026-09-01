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
   all-reduce). **No compila**: llama a `shardLinearLeafs`/`AllToShardedLinear`/
   `ShardedToAllLinear`/`QuantizedAllToShardedLinear`/`QuantizedShardedToAllLinear`, símbolos
   que no existen en ningún sitio de `mlx-swift-lm` ni en ninguna rama de `mlx-swift` conocida
   hasta el punto 3.
3. **El puerto Swift que sí falta ya existe**, en el otro repo: `JuanColilla/mlx-swift`, rama
   `feature/distributed-group-port`, archivo `Source/MLXNN/Distributed.swift` (739 líneas) —
   exactamente las clases que el punto 2 necesita. Nadie conectó los dos lados.

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
- Aunque las piezas ya casadas (modelo en `inferring`, primitiva en `mlx-swift`), conectarlas
  implica: traer `feature/distributed-group-port` como dependencia de `mlx-swift-lm`, rebasar
  el diff de `archive/inferring-ios-distrib-tensor` sobre el `develop` actual (que ha
  divergido mucho desde el punto de fork de esa rama, 2026-02-12 — ver la fusión oficial de
  hoy mismo en `f078bf4` como ejemplo de cuánto ha cambiado Qwen3.5/Qwen3Next desde entonces),
  y verificar el sharding de pesos cuantizados con datos reales, no solo que compile.
- Es trabajo nuevo de alcance medio, no una fusión mecánica.

## 4. Plan de retoma (checklist para cuando se reanude)

1. En `mlx-swift`: revisar si `feature/distributed-group-port` sigue al día con `origin/main`
   (llevaba solo 2 commits de diferencia el 2026-09-01) y con `fork/main`; fusionar si hace
   falta.
2. Traer `mlx-swift` en esa rama como dependencia de `mlx-swift-lm` (`Package.swift`, por
   `revision:`, nunca por rango — ver la sección de tags de `AGENTS.md`).
3. Sobre `develop` actual (no sobre el `wip/tensor-parallel` de esta fecha, que solo sirve de
   ancla), portar el diff de `archive/inferring-ios-distrib-tensor` contra su base
   (`718877e`) modelo a modelo, arquitectura por arquitectura. `git diff
   718877e29a267ee152d24e27a75721b102adf4b8 archive/inferring-ios-distrib-tensor` da el punto
   de partida exacto.
4. Verificar que `TransformerLayer` (ya existe en `develop`, `LanguageModel.swift:446`) sigue
   siendo el protocolo que ambos mecanismos —pipeline-parallel actual y este tensor-parallel—
   comparten; es la pieza que permite componer los dos (2D parallelism) en vez de que
   compitan.
5. Medir antes de dar por bueno: tensor-parallel exige un all-reduce por cada bloque de
   atención y cada MLP, no solo en los límites de pipeline. Sobre Bonjour/TCP entre iPhones
   puede ser network-bound. Prototipar con 2 nodos reales antes de portar las 7 arquitecturas.

## 5. Fuera de alcance de esta limpieza (no tocado, no relacionado)

- `feature/bonsai-1bit-support` (mlx-swift) — cuantización afín de 1 bit, 2 commits propios
  sobre `fork/main`. Trabajo propio legítimo, sin relación con tensor-parallel.
- `fork/materialized-array` (mlx-swift) — 11 commits propios divergidos, 25 detrás de
  `origin/materialized-array`. Tema distinto (materialización de arrays), no revisado a fondo
  hoy; no se ha tocado ni borrado.
- El worktree `fix/mesh-distributed-inference` de este repo — trabajo activo de otro tema
  (tolerancia de `safetensorWeightURLs`/`loadArraysAndMetadata` a recortes, TD-032), no forma
  parte del desorden de tensor-parallel.
