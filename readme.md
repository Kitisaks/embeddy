# Multilingual Embedding API

A FastAPI-powered service for generating multilingual sentence embeddings using the [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) model. Supports local CPU inference, token-based authentication, and Docker deployment with configurable resource limits.

---

## Features

- Supports 100+ languages
- Local embedding generation (no external API calls)
- Cosine similarity-ready embeddings
- API key authentication
- CPU-only PyTorch (smaller image, faster builds)
- Configurable CPU / concurrency limits
- Dockerized for easy deployment

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kitisaks/embeddy.git
cd embeddy
```

### 2. Configure environment

```bash
cp .env.example .env
# edit .env and set EMBED_API_TOKEN
```

### 3. Build & Run with Docker

```bash
docker build -t embeddy .
docker run --env-file .env -p 8000:8000 embeddy
```

### 4. Local (venv)

```bash
# Requires Python 3.13 (managed via asdf — see .tool-versions)
asdf install          # installs python 3.13.14 if missing
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
./start-local.sh
```

---

## Configuration

All knobs are environment variables (see `.env.example`).

- **`EMBED_API_TOKEN`** (required) — Bearer token for `/embed` endpoints
- **`MAX_BATCH_SIZE`** (default `32`) — max texts per `/embed/batch` request
- **`MAX_CONCURRENT_REQUESTS`** (default `1` in Docker, `4` in code) — semaphore limiting parallel inference
- **`TORCH_NUM_THREADS`** (default `24` in Docker) — PyTorch intra-op CPU threads
- **`TORCH_INTEROP_THREADS`** (default `1`) — PyTorch inter-op threads
- **`OMP_NUM_THREADS`** / **`MKL_NUM_THREADS`** (default `24` in Docker) — OpenMP / MKL thread caps (set before process start)
- **`ENABLE_QUANTIZATION`** (default `false`) — dynamic int8 on Linear layers (~1.5–2× CPU throughput, slight quality trade-off)
- **`TOKENIZERS_PARALLELISM`** (default `false`) — disables HF tokenizer thread pool oversubscription

Inspect the live values (auth required):

```bash
curl -H "Authorization: Bearer $EMBED_API_TOKEN" http://127.0.0.1:8000/debug/config
```

**Workers:** keep a single uvicorn worker. Each worker loads a full copy of the ~2GB model into RAM.

---

## Limiting CPU / resources

### Dokku target: 48 CPUs / 251 GiB RAM

The Dockerfile is tuned to use at most 24 compute threads per inference. Thread
settings are performance controls, not hard resource boundaries. Apply Dokku
cgroup limits to guarantee that the app cannot exceed 50% of host CPU and the
selected practical 16 GiB RAM ceiling:

```bash
APP=embeddy

# Runtime/deploy containers: hard limit of 24 CPUs and 16 GiB RAM+swap.
# Setting memory-swap equal to memory prevents additional swap consumption.
dokku resource:limit --cpu 24 --memory 16g --memory-swap 16g "$APP"

# Ensure Dokku passes the matching inference settings.
dokku config:set "$APP" \
  TORCH_NUM_THREADS=24 \
  TORCH_INTEROP_THREADS=1 \
  OMP_NUM_THREADS=24 \
  MKL_NUM_THREADS=24 \
  MAX_CONCURRENT_REQUESTS=1 \
  TOKENIZERS_PARALLELISM=false

# Limits take effect on the next deploy/rebuild.
dokku ps:rebuild "$APP"

# Verify the configured limits.
dokku resource:report "$APP"
```

The 16 GiB memory limit is intentionally much lower than half of the currently
available ~163 GiB because this model does not need an 80 GiB allowance. Do not
run multiple web processes: every process loads another full model copy.

To limit Dockerfile build memory as well:

```bash
dokku resource:limit --memory 16g --memory-swap 16g --process-type build "$APP"
```

Dokku's Dockerfile builder supports build-time memory limits but not build-time
CPU limits. Constrain build CPU at the host/cgroup level if that is also
required. Runtime CPU remains hard-limited by the 24-CPU app limit above.

### Plain Docker runtime

Cap container CPU and memory, and optionally override thread env vars:

```bash
docker run --env-file .env \
  --cpus="24" \
  --memory="16g" \
  --memory-swap="16g" \
  -e TORCH_NUM_THREADS=24 \
  -e OMP_NUM_THREADS=24 \
  -e MAX_CONCURRENT_REQUESTS=1 \
  -p 8000:8000 \
  embeddy
```

Keep `TORCH_NUM_THREADS` / `OMP_NUM_THREADS` equal to the CPU allocation, and
keep `MAX_CONCURRENT_REQUESTS=1` so simultaneous inferences do not compete for
the same cores.

### At build time

Model download and `pip install` are CPU-heavy. Constrain the builder, not just the running container:

- **Docker Desktop**: lower CPUs/memory under **Settings → Resources** before `docker build`.
- **Linux / CI**: cap the build job at the orchestrator (Kubernetes limits, `cpulimit`, cgroup quotas), e.g.:

```bash
# Example: run the build under a 2-CPU / 4G cgroup (systemd)
systemd-run --scope -p CPUQuota=200% -p MemoryMax=4G \
  docker build -t embeddy .
```

- **BuildKit / buildx** on a dedicated builder VM: size the builder host (or remote builder) to the CPU budget you want; `docker build` itself does not take `--cpus` the way `docker run` does.

### Local start

`./start-local.sh` exports thread defaults from `.env` (or `2` / `1` if unset) before launching uvicorn so OpenMP/MKL see them at library init.

---

## API

- `GET /health` — liveness
- `POST /embed` — single text (`{"text": "..."}`)
- `POST /embed/batch` — batch (`{"texts": ["...", "..."]}`)
- `GET /debug/config` — effective resource / model config (auth required)

All `/embed*` and `/debug/*` routes require `Authorization: Bearer <EMBED_API_TOKEN>`.

---

## References

- [SentenceTransformers](https://www.sbert.net/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Hugging Face Model](https://huggingface.co/intfloat/multilingual-e5-large)
