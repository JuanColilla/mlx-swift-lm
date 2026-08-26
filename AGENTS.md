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
