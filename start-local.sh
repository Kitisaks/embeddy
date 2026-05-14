#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
ENV_FILE="$ROOT_DIR/.env"

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing virtual environment at .venv"
  echo "Create it with:"
  echo "  python3 -m venv .venv"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file."
  echo "Create it with:"
  echo "  echo \"EMBED_API_TOKEN=your-secret-token\" > .env"
  exit 1
fi

if ! grep -q '^EMBED_API_TOKEN=' "$ENV_FILE"; then
  echo ".env exists but EMBED_API_TOKEN is missing."
  echo "Add this line:"
  echo "  EMBED_API_TOKEN=your-secret-token"
  exit 1
fi

echo "Starting local API on http://127.0.0.1:8000"
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
