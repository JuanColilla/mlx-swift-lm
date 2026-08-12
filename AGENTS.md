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

## Convención de tags: sufijo `v`

Los tags que terminan en **`v`** marcan una **variante custom del fork**,
para diferenciarlos de los tags del oficial:

- `3.31.4` → tag oficial de upstream (mismo commit o antecesor directo).
- `3.31.4v` → tag nuestro en el fork, construido sobre esa misma base,
  con nuestras variaciones (ej. `DOCS/` de investigación) incluidas.

Al crear un tag nuevo tras un merge estable:
1. Identificar el último tag oficial relevante (`git ls-remote --tags
   https://github.com/ml-explore/mlx-swift-lm.git`).
2. Usar ese mismo número + sufijo `v` (o incrementar el patch + `v` si el
   HEAD ya tiene commits oficiales por delante del último tag oficial).
3. Tag anotado (`git tag -a <version>v -m "..."`) apuntando al commit
   mergeado en `main`.
4. `git push origin <version>v` para subirlo al fork.

Nunca crear un tag sin sufijo `v` en este fork — reservar los tags "pelados"
para los que vengan replicados del oficial.
