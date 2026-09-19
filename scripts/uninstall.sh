#!/usr/bin/env bash
# Removes every install recorded in docs/SETUP.md. Does not delete the repo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Formulae VectraX added; written right after install, so later
# unrelated brew installs are not touched.
BREW_ADDED="$ROOT/tools/brew_added.txt"
CVAT_DIR="$ROOT/tools/cvat"

if [[ -d "$CVAT_DIR" ]]; then
  (cd "$CVAT_DIR" && docker compose down -v --rmi all)
fi

if [[ -s "$BREW_ADDED" ]]; then
  echo "Removing brew formulae: $(tr '\n' ' ' < "$BREW_ADDED")"
  xargs brew uninstall --ignore-dependencies < "$BREW_ADDED"
fi

rm -rf "$ROOT/.venv"
uv cache prune

echo "Done. Delete $ROOT to remove the project itself."
