#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
ENV_FILE="$ROOT_DIR/.env"

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing virtual environment at .venv"
  echo "Create it with asdf Python 3.13 (matches the Docker image):"
  echo "  asdf install"
  echo "  python -m venv .venv"
  exit 1
fi

# Warn if the venv is not on the same major.minor as production (3.13)
_venv_py="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$_venv_py" != "3.13" ]]; then
  echo "Warning: .venv is Python ${_venv_py}; production image uses 3.13."
  echo "Recreate with:  rm -rf .venv && asdf install && python -m venv .venv"
fi
unset _venv_py

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file."
  echo "Create it with:"
  echo "  cp .env.example .env"
  echo "  # then set EMBED_API_TOKEN"
  exit 1
fi

if ! grep -q '^EMBED_API_TOKEN=' "$ENV_FILE"; then
  echo ".env exists but EMBED_API_TOKEN is missing."
  echo "Add this line:"
  echo "  EMBED_API_TOKEN=your-secret-token"
  exit 1
fi

# Load .env into the shell so TORCH_*/OMP_*/MKL_* are visible to the process
# (python-dotenv also loads them, but OpenMP/MKL read env at library init).
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Sensible local defaults if not set in .env (do not overwrite user values)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"
export TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Starting local API on http://127.0.0.1:8000"
echo "  TORCH_NUM_THREADS=$TORCH_NUM_THREADS  OMP_NUM_THREADS=$OMP_NUM_THREADS  MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-4}"
# --reload implies a single worker (required: each worker loads a ~2GB model copy)
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
