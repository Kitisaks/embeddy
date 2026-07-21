FROM python:3.13-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Tuned for the target Dokku host: 48 CPUs / 251 GiB RAM.
# These thread settings optimize inference within a 24-CPU allocation. They are
# not hard limits; enforce those with `dokku resource:limit` (see readme.md).
ENV OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24 \
    TORCH_NUM_THREADS=24 \
    TORCH_INTEROP_THREADS=1 \
    MAX_CONCURRENT_REQUESTS=1 \
    MAX_BATCH_SIZE=32 \
    ENABLE_QUANTIZATION=false \
    TOKENIZERS_PARALLELISM=false \
    MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install basic system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy app files
COPY app/requirements.txt .
# CPU-only torch via --extra-index-url in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights so container startup requires no network I/O
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

COPY app/ .

# Single worker: each worker loads a full ~2GB model copy into RAM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
