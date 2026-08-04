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

## Flujo de trabajo habitual

1. `git fetch origin` para comprobar si hay novedades del oficial.
2. Si hay cambios nuevos en `origin/main` (oficial) y no hay commits propios
   divergentes, hacer fast-forward de `main` local.
3. Desarrollar variaciones propias (docs, research, parches) en ramas
   locales normales (ej. `docs/...`, `feature/...`).
4. Mergear a `main` cuando esté estable.
5. `git push origin main` sube el resultado al fork (nunca al oficial —
   no se abren PRs al oficial salvo que el usuario lo pida explícitamente).

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
