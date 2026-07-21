import os
import logging
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Optional

import torch
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL = "intfloat/multilingual-e5-large"
# Production is CPU-only; pin to CPU by default so we don't land on MPS/CUDA
# where our manual tokenize -> forward path would mismatch tensor devices.
DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
API_TOKEN = os.environ.get("EMBED_API_TOKEN")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
# Limit simultaneous inference calls so threads don't compete for CPU/GPU
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "4"))
ENABLE_QUANTIZATION = os.environ.get("ENABLE_QUANTIZATION", "false").lower() in (
    "1",
    "true",
    "yes",
)


def _configure_torch_threads() -> tuple[int, int]:
    """Cap PyTorch CPU threads before any model is created.

    Must run at import time (before get_model). Defaults leave torch's
    auto-detection alone when env vars are unset.
    """
    num_threads = os.environ.get("TORCH_NUM_THREADS")
    interop_threads = os.environ.get("TORCH_INTEROP_THREADS")

    if num_threads:
        torch.set_num_threads(int(num_threads))
    if interop_threads:
        torch.set_num_interop_threads(int(interop_threads))

    return torch.get_num_threads(), torch.get_num_interop_threads()


TORCH_NUM_THREADS, TORCH_INTEROP_THREADS = _configure_torch_threads()

_inference_semaphore: asyncio.Semaphore


def debug_api_key() -> bool:
    if API_TOKEN:
        masked = f"{API_TOKEN[:8]}...{API_TOKEN[-4:]}" if len(API_TOKEN) > 12 else "***"
        logger.info(f"API token loaded: {masked} ({len(API_TOKEN)} chars)")
    else:
        logger.error("EMBED_API_TOKEN not set — authentication will fail")
    return API_TOKEN is not None


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info(f"Loading model: {MODEL} on device={DEVICE}")
    model = SentenceTransformer(MODEL, device=DEVICE)
    model.eval()

    if ENABLE_QUANTIZATION:
        # Dynamic int8 on Linear layers — ~1.5-2x CPU throughput, slight quality trade-off
        logger.info("Applying dynamic int8 quantization (ENABLE_QUANTIZATION=true)")
        underlying = model[0].auto_model
        # torch.ao.quantization is the supported path on torch 2.x
        quantized = torch.ao.quantization.quantize_dynamic(
            underlying,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        model[0].auto_model = quantized

    return model


def _encode(model: SentenceTransformer, texts: list[str]) -> tuple[list[list[float]], list[int]]:
    """Single tokenization pass: count tokens from attention mask, then run forward."""
    features = model.preprocess(texts)
    # Attention mask sum = actual (non-padding) token count per item
    token_counts: list[int] = features["attention_mask"].sum(dim=1).tolist()
    # tokenize() returns CPU tensors; move them onto the model's device so the
    # forward pass doesn't fail on MPS/CUDA with an unallocated-storage error.
    features = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in features.items()
    }
    with torch.inference_mode():
        out = model.forward(features)
    embeddings: list[list[float]] = out["sentence_embedding"].cpu().numpy().tolist()
    return embeddings, token_counts


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _inference_semaphore
    logger.info("Starting embedding service...")
    logger.info(
        "CPU threads: torch=%s interop=%s | OMP=%s MKL=%s | quantization=%s",
        TORCH_NUM_THREADS,
        TORCH_INTEROP_THREADS,
        os.environ.get("OMP_NUM_THREADS", "unset"),
        os.environ.get("MKL_NUM_THREADS", "unset"),
        ENABLE_QUANTIZATION,
    )
    debug_api_key()
    _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    try:
        model = get_model()
        # Warm-up: first real request otherwise pays for oneDNN / tokenizer lazy init
        await run_in_threadpool(_encode, model, ["query: warm-up"])
        logger.info("Model loaded and warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield


app = FastAPI(
    title="Text Embedding API",
    description="API for generating text embeddings using multilingual E5 model",
    version="1.0.0",
    lifespan=lifespan,
)

# Compress responses >= 1 KB (embeddings are ~4 KB uncompressed)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ---------- Pydantic models ----------

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)


class BatchTextInput(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=8192)]] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE
    )


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    usage: dict


class BatchEmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    usage: dict


# ---------- Auth ----------

async def verify_token(authorization: Optional[str] = Header(None, alias="Authorization")):
    if not authorization:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:]
    if not API_TOKEN:
        logger.error("API token not configured")
        raise HTTPException(status_code=500, detail="Server configuration error")
    if token != API_TOKEN:
        logger.warning(f"Invalid token attempt: {token[:8]}...")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------- Endpoints ----------

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL, "api_key_configured": API_TOKEN is not None}


@app.post("/embed", response_model=EmbeddingResponse)
async def get_embedding(
    data: TextInput,
    _: None = Depends(verify_token),
) -> EmbeddingResponse:
    try:
        model = get_model()
        text = f"query: {data.text.strip()}"
        async with _inference_semaphore:
            embeddings, token_counts = await run_in_threadpool(_encode, model, [text])
        embedding, total_tokens = embeddings[0], token_counts[0]
        logger.info(f"Embedded len={len(data.text)}, tokens={total_tokens}")
        return EmbeddingResponse(
            embedding=embedding,
            usage={"model": MODEL, "total_tokens": total_tokens, "text_length": len(data.text)},
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")


@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def get_embeddings_batch(
    data: BatchTextInput,
    _: None = Depends(verify_token),
) -> BatchEmbeddingResponse:
    """Encode multiple texts in a single forward pass — far cheaper than N separate calls."""
    try:
        model = get_model()
        texts = [f"query: {t.strip()}" for t in data.texts]
        async with _inference_semaphore:
            embeddings, token_counts = await run_in_threadpool(_encode, model, texts)
        total_tokens = sum(token_counts)
        logger.info(f"Batch embedded count={len(texts)}, total_tokens={total_tokens}")
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            usage={"model": MODEL, "total_tokens": total_tokens, "count": len(texts)},
        )
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")


@app.get("/debug/config")
async def debug_config(_: None = Depends(verify_token)):
    return {
        "model": MODEL,
        "device": DEVICE,
        "api_key_configured": API_TOKEN is not None,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "torch_num_threads": TORCH_NUM_THREADS,
        "torch_interop_threads": TORCH_INTEROP_THREADS,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "enable_quantization": ENABLE_QUANTIZATION,
    }


if __name__ == "__main__":
    import uvicorn
    # Single worker recommended: each worker loads a full ~2GB model copy into RAM
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
