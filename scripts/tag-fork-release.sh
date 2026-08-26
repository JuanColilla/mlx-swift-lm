#!/usr/bin/env bash
# Crea un tag de "variante de fork" con el esquema vX.X.X-fork.N y lo publica
# en GitHub (remote `fork`).
#
#   X.X.X = último tag oficial detectado en el remote `origin`.
#   N     = contador incremental de variantes propias sobre esa base
#           (se reinicia a 1 cada vez que el oficial publica una base nueva).
#
# Uso:
#   scripts/tag-fork-release.sh [--ref <commit-ish>] [--message <texto>] [--dry-run]
#
# Por qué -fork.N y no +fork.N ni X.X.Xv:
#   SemVer ignora el metadata de build (+...) al comparar precedencia, así
#   que Xcode/SPM no garantizan resolver la variante más reciente con ese
#   esquema. Un sufijo "v" pegado al final (X.X.Xv) ni siquiera es SemVer
#   válido. Los identificadores de pre-release (-fork.N) sí se comparan
#   numéricamente, así que -fork.3 siempre tiene más precedencia que
#   -fork.2 y -fork.1, y Xcode resuelve siempre la variante más nueva.
set -euo pipefail

REMOTE_OFFICIAL="origin"
REMOTE_FORK="fork"
REF="HEAD"
DRY_RUN=0
MESSAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --message) MESSAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Argumento desconocido: $1" >&2; exit 1 ;;
  esac
done

git rev-parse --git-dir >/dev/null 2>&1 || { echo "No es un repositorio git." >&2; exit 1; }
git remote get-url "$REMOTE_OFFICIAL" >/dev/null 2>&1 || { echo "No existe el remote '$REMOTE_OFFICIAL'." >&2; exit 1; }
git remote get-url "$REMOTE_FORK" >/dev/null 2>&1 || { echo "No existe el remote '$REMOTE_FORK'." >&2; exit 1; }

TARGET_COMMIT=$(git rev-parse --verify "${REF}^{commit}") || { echo "Ref inválida: $REF" >&2; exit 1; }

echo "Consultando último tag oficial en '$REMOTE_OFFICIAL'..." >&2
OFFICIAL_BASE=$( (git ls-remote --tags "$REMOTE_OFFICIAL" \
  | awk -F'refs/tags/' '{print $2}' \
  | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' \
  | sort -V | tail -1) || true)

if [[ -z "$OFFICIAL_BASE" ]]; then
  echo "No se ha encontrado ningún tag oficial X.Y.Z en '$REMOTE_OFFICIAL'." >&2
  exit 1
fi
echo "Base oficial detectada: $OFFICIAL_BASE" >&2

echo "Consultando variantes de fork existentes en '$REMOTE_FORK'..." >&2
ESCAPED_BASE=$(printf '%s' "$OFFICIAL_BASE" | sed 's/\./\\./g')
LAST_N=$( (git ls-remote --tags "$REMOTE_FORK" \
  | awk -F'refs/tags/' '{print $2}' \
  | grep -E "^v${ESCAPED_BASE}-fork\.[0-9]+$" \
  | sed -E 's/.*-fork\.([0-9]+)$/\1/' \
  | sort -n | tail -1) || true)

NEXT_N=$(( ${LAST_N:-0} + 1 ))
NEW_TAG="v${OFFICIAL_BASE}-fork.${NEXT_N}"

if git ls-remote --tags "$REMOTE_FORK" | grep -q "refs/tags/${NEW_TAG}\$"; then
  echo "El tag $NEW_TAG ya existe en '$REMOTE_FORK' (inconsistencia). Aborto." >&2
  exit 1
fi

DEFAULT_MESSAGE="Variante de fork sobre oficial ${OFFICIAL_BASE} (#${NEXT_N})"
TAG_MESSAGE="${MESSAGE:-$DEFAULT_MESSAGE}"

echo ""
echo "Nuevo tag: $NEW_TAG"
echo "Commit:    $TARGET_COMMIT ($(git log -1 --format='%s' "$TARGET_COMMIT"))"
echo "Mensaje:   $TAG_MESSAGE"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] No se crea ni se publica ningún tag." >&2
  exit 0
fi

git tag -a "$NEW_TAG" "$TARGET_COMMIT" -m "$TAG_MESSAGE"
git push "$REMOTE_FORK" "$NEW_TAG"

echo "Tag '$NEW_TAG' creado y publicado en '$REMOTE_FORK' (GitHub)."
