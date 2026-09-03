# AGENTS.md — mlx-swift-lm (fork)

> Este `AGENTS.md` es específico de este repositorio y complementa (no
> reemplaza) el `AGENTS.md` global del usuario. Ante conflicto, el más
> cercano al archivo trabajado gana — este archivo tiene prioridad sobre
> el global para todo lo que ocurra dentro de este repo.

## Project Context
- Type: personal
- Architecture: swift-package (librería, no app)

## Naturaleza del repo

Este repositorio es un **fork personal** de
[`ml-explore/mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm), el
repo oficial de MLX Swift para modelos de lenguaje. El fork vive en
`JuanColilla/mlx-swift-lm` y existe para:

1. Recibir las mejoras del oficial según se publican.
2. Añadir variaciones propias (investigación, docs, parches puntuales) que
   no están en upstream.

## Configuración de remotes (IMPORTANTE)

El remote `origin` tiene URLs **distintas para fetch y push**, a propósito:

```
origin  fetch: https://github.com/ml-explore/mlx-swift-lm.git   (oficial/upstream)
origin  push:  https://github.com/JuanColilla/mlx-swift-lm.git  (mi fork)
```

Esto significa:
- `git fetch origin` / `git pull` → trae cambios del **oficial**.
- `git push origin <rama>` → sube al **fork**, nunca al oficial.

**No añadir un remote `upstream` separado** — el propio `origin` ya cumple
ese rol vía fetch URL. No cambiar esta configuración sin confirmarlo con el
usuario primero (romper esto puede hacer que un push termine en el
repo oficial por error).

### Remote `fork` (nuestro fork, lectura y escritura)

`origin` sabe empujar al fork pero **no sabe leerlo**: su fetch URL apunta al
oficial, así que `origin/main` es el oficial y las ramas del fork no aparecen
por ningún lado. Por eso existe además:

```
fork  fetch/push: https://github.com/JuanColilla/mlx-swift-lm.git
```

Úsalo siempre que necesites **ver** el estado real del fork
(`git ls-remote --heads fork`, `git fetch fork`). Sin él es fácil creer que
una rama se perdió cuando lo que pasa es que solo vive en remoto — ocurrió:
en agosto de 2026 `develop` y `main` locales apuntaban a un commit del
oficial, sin nada del fork, mientras la integración completa estaba a salvo
en `fork/develop`.

### Remote `inferring` (solo lectura)

Existe además un remote `inferring` →
`https://github.com/N1k1tung/mlx-swift-lm.git`, otro fork de terceros del que
se traen referencias puntualmente. **Nunca hacer push ahí.**

De él proceden los tags `ios-distrib-*` presentes en local; no son tags de
este fork y por tanto la convención de sufijo `v` descrita abajo no les
aplica.

## Modelo de ramas: solo `develop` y `main`

Desde agosto de 2026 el fork tiene **exactamente dos ramas**, local y
remotamente:

- **`develop`** — rama de integración y única en la que se trabaja. Contiene
  el oficial fusionado más todo lo propio.
- **`main`** — espejo de `develop` (mismo árbol), actualizado con un merge
  explícito cuando `develop` está estable.

Las 14 ramas que existían antes (`feature/*`, `upstream/*`, `wip/pre-*`,
`codex/*`) se consolidaron y se borraron; sus puntas quedaron como tags
`archive/<rama>` **solo en local**, no en el fork. Todo commit único de
aquellas ramas es antecesor de `develop`, así que no hay contenido que
rescatar de ellas.

El criterio de integración vive en
[`knowledge/2026-08-04-develop-branch-unification-report.md`](knowledge/2026-08-04-develop-branch-unification-report.md):
el upstream es la base arquitectónica y lo propio se porta encima, nunca al
revés. Léelo antes de resolver conflictos con el oficial.

**Única excepción vigente:** `wip/tensor-parallel`, empujada a `fork` (no solo
tag local como las antiguas `archive/<rama>`). Aparca la investigación de
tensor-parallel hasta después de la 2.1.0 — ver
[`knowledge/2026-09-01-tensor-parallel-desorden.md`](knowledge/2026-09-01-tensor-parallel-desorden.md)
(el documento solo existe en esa rama, no en `develop`). No fusionar a
`develop` sin retomar el trabajo primero; no borrarla sin confirmar con el
usuario.

## Flujo de trabajo habitual

1. `git fetch origin` para comprobar si hay novedades del oficial.
2. Fusionar `origin/main` en `develop` resolviendo según el informe anterior.
3. Compilar y pasar tests (ver los avisos de entorno más abajo).
4. `git push fork develop:develop`.
5. Cuando `develop` esté estable, actualizar `main` con un merge de `develop`
   y empujar. Nunca se abren PRs al oficial salvo que el usuario lo pida.

Para trabajar en algo grande, crea un worktree en
`../mlx-swift-lm-worktrees/<nombre>` y bórralo al terminar, en vez de dejar
una rama viva.

## Avisos de entorno (agosto 2026)

**El runner de `swift-testing` no encuentra el metallib de MLX.** Cualquier test
escrito con Swift Testing que toque MLX muere con *"Failed to load the default
metallib"* antes de ejecutar nada; reproducido en el árbol sin modificar con
`MixedPrecisionQuantLoadTests`. Los tests con XCTest en el mismo target pasan
sin problema. Hasta que se resuelva, **escribe los tests que toquen MLX con
XCTest**, no con `@Test`.

**Un worktree bajo `~/Documents` rompe la firma de los bundles de test.** Esa
carpeta la sincroniza un fileprovider que deja `com.apple.FinderInfo` en los
directorios recién creados, y `codesign` lo rechaza: *"resource fork, Finder
information, or similar detritus not allowed"*. `xattr -c` no basta porque el
atributo vuelve durante la build. `swift build` (sin tests) no se ve afectado.
Solución: `swift test --scratch-path <ruta fuera de ~/Documents>`.


**`MLXFoundationModelsTests` no puede ejecutarse con Xcode 27 beta.** El
`.swiftinterface` de FoundationModels declara inicializadores con
`metadata: [String: any ConvertibleToGeneratedContent]` que el dylib real no
exporta (`Transcript.Reasoning.init`,
`LanguageModelExecutorGenerationRequest.init`). Referenciarlos compila y mata
el proceso al cargar la imagen — SIGSEGV en 0x0, sin traza útil. No es
arreglable desde aquí: solo afecta a ese target de tests, y `MLXLMTests` sí
pasa entero. En la librería el mismo problema existía con
`Response.Action.updateMetadata`, resuelto no referenciando el símbolo (ver
el TODO en `MLXLanguageModel.emitMetadata`, con el mismo patrón que el de
`emitUsage`). Para auditar si un símbolo del sistema existe de verdad:
`nm -u` sobre el `.o` + `dyld_info -exports` sobre el framework, y `comm -23`
entre ambos conjuntos — leer el interface no basta.

**El repo vive bajo `~/Documents`, que iCloud sincroniza.** Dos consecuencias:

1. `swift build` falla al firmar `mlx-swift_Cmlx.bundle` con *"resource fork,
   Finder information, or similar detritus not allowed"*, porque el
   proveedor de ficheros marca el bundle con `com.apple.FinderInfo`. `xattr -cr`
   no basta: se vuelve a poner. Workaround: compilar fuera de iCloud con
   `swift build --scratch-path /tmp/<algo>`.
2. Aparecen duplicados con sufijo ` 2`/` 3` — incluso dentro de `.git/refs`.
   Los `.swift` duplicados rompen la compilación (`invalid redeclaration`).
   Antes de commitear, `git status` y comprobar que el índice no arrastra
   ninguno: `git commit` publica el índice entero, no solo lo que acabas de
   añadir con `git add <ruta>`.

## Convención de tags: `vX.X.X-fork.N`

Esquema vigente desde 2026-08-26. Los tags de variante propia siguen
`vX.X.X-fork.N`:

- `X.X.X` = último tag oficial de upstream (`origin`) sobre el que está
  construida la variante.
- `N` = contador incremental de variantes propias sobre esa base; se
  reinicia a 1 cada vez que se sube la base a un tag oficial más nuevo.

Ejemplo: si el oficial va por `3.31.4`, la primera variante propia sobre esa
base es `v3.31.4-fork.1`, la segunda `v3.31.4-fork.2`, etc. Al fusionar
`3.31.5` del oficial, la siguiente variante vuelve a `v3.31.5-fork.1`.

**Por qué este formato y no otro:**
- `vX.X.X` (prefijo `v`, sin sufijo) es lo que Xcode/SPM esperan para
  resolver un paquete como dependencia — un sufijo pegado (`X.X.Xv`, el
  esquema anterior) no es SemVer válido y Xcode no lo ordena.
- El identificador va como **pre-release** (`-fork.N`), no como metadata de
  build (`+fork.N`): SemVer ignora el metadata de build al comparar
  precedencia, así que con `+fork.N` Xcode no garantiza resolver la
  variante más reciente. Con `-fork.N` los identificadores numéricos se
  comparan numéricamente, así que `-fork.3 > -fork.2 > -fork.1` siempre, y
  la comparación de la tripleta base (`3.31.5 > 3.31.4`) tiene prioridad
  sobre el sufijo al saltar de base oficial.
- Como Xcode/SPM resuelven versiones por URL de repositorio, estos tags
  nunca se comparan contra los tags del oficial (viven en repos distintos)
  — el nombre es solo para que nosotros sepamos a qué base está pegada
  cada variante.

**Límite verificado (2026-08-26):** SPM **no** resuelve automáticamente la
variante más reciente en dependencias por rango (`from:`, `upToNextMajor`).
Probado contra `v0.31.6-fork.1` real en `mlx-swift`: `from: "0.31.6"` y
`from: "0.31.6-fork.1"` resuelven a `0.31.6` (el oficial, ignora el
pre-release); solo `.exact("0.31.6-fork.1")` la coge. En este repo es
irrelevante porque `Package.swift` fija `mlx-swift` por `revision:` (SHA
exacto), no por rango — nunca se depende de esa resolución automática. Los
tags siguen sirviendo como marcador ordenado y trazable, y son consumibles
con `.exact(...)` si algún día hiciera falta, pero no cumplen "Xcode coge
siempre la más nueva" en resolución por rango.

**Tags heredados con el esquema antiguo** (`3.31.4v`, `3.31.5v`, `3.31.6v`,
`v3.31.5`) se dejan como están, sin retaguear — son historial, no se
resuelven con Xcode y no hay que reproducir ese formato en tags nuevos.

Crear un tag nuevo, siempre con el script — no a mano:

```
scripts/tag-fork-release.sh              # tag anotado en HEAD, push a `fork`
scripts/tag-fork-release.sh --dry-run    # solo muestra qué tag generaría
scripts/tag-fork-release.sh --ref <sha> --message "texto"
```

El script detecta la base oficial vía `git ls-remote --tags origin`, calcula
`N` vía `git ls-remote --tags fork`, y publica el tag con
`git push fork <tag>` — cada variante nueva genera siempre un tag real en
GitHub, nunca solo local. Nunca usar `git push origin <tag>` para esto: en
este repo `origin` empuja al fork por configuración especial (ver arriba),
pero `fork` es el remote explícito y sin ambigüedad — el script lo usa
siempre a propósito.

## Instrucciones heredadas del oficial (ml-explore/mlx-swift-lm)

> Añadidas en la fusión del 2026-09-01 (PR #585 oficial). Aplican a este
> fork igual que al oficial.

## AI usage policy

AI-generated code is allowed. Submitting code you do not understand is not. You
are 100% responsible for every line you contribute, however it was produced. You
must be able to explain the code you submit, and you must disclose how you used
AI.

AI-drafted prose is allowed too — commit messages, pull request descriptions,
issues, review replies. The condition is that you read every word before it is
submitted and confirm it says what you mean. Do not ask a reviewer to read prose
you did not read yourself.

[CONTRIBUTING.md](CONTRIBUTING.md) states this policy for contributors, including
what can happen when it is not followed. Point the user there if they ask you to
submit prose they have not read.

## Agent rules

- After you change code, explain what changed and why, so the user can own it
- You MAY draft commit messages, PR descriptions, and replies to reviewers. Show
  them to the user to read and edit before they reach the repo.
- Do NOT create a PR, open an issue, or post a comment on behalf of the user

### Examples

User: Please create and submit the PR for me.
Agent: I cannot open the PR for you. The PR template asks you to confirm that you
read the description and that it matches the code changes. That acknowledgment is
yours to make, not mine. I can draft the description for you to read and edit.

User: I don't want to read it, just submit it.
Agent: I cannot submit it for you, you have to do that. Before you do, note that
CONTRIBUTING.md asks you to confirm you read the PR description, and explains what can
happen when contributors skip that. I can summarize the PR's changes first if it
helps.

User: Please address the reviewer comments.
Agent: Here is a draft reply to each comment. Read it, edit it, and post it
yourself — I cannot post for you.

## Code standards

- Keep code comments concise (usually 1-2 lines)
- Avoid redundant or excessive inline commentary
- Write comments in plain, direct English: short sentences, common words, active
  voice.

### Examples

```swift
  // Good (no comment)

  let cacheKey = "\(modelID):\(kind.rawValue):\(sourceHash)"

  // Bad (excessive comment for explicit code)

  // The constraint cache is keyed on the model, the constraint kind, and a hash
  // of the grammar source. Two requests that share a model but not a grammar
  // must not collide: without the source in the key, the second request would
  // reuse the grammar compiled for the first.

  let cacheKey = "\(modelID):\(kind.rawValue):\(sourceHash)"
```

## Working in this repo

- Read `skills/mlx-swift-lm/SKILL.md` and the files in its `references/`
  directory before you use the public API. `skills/README.md` explains how to
  install the skill.
- `swift test` does not work here. Run unit tests with `xcodebuild test -scheme
  mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation`.
- Format with `pre-commit run --all-files` before you hand work back. CI pins a
  specific swift-format version, set in `.github/workflows/pull_request.yml`.
  Match it locally: another version reformats files the PR does not touch, which
  turns CI red.
- `pre-commit` walks the whole working directory, so it also reports errors from
  `DerivedData/` and `.build/`. Those are vendored dependencies. Ignore them.
- `scripts/verify-docs.sh` runs the DocC check that CI runs for every library
  target.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for integration tests.
