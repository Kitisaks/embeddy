import os
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MODEL = "intfloat/multilingual-e5-large"
API_TOKEN = os.environ.get("EMBED_API_TOKEN")

# Debug API key after initialization
def debug_api_key():
    """Debug API key configuration"""
    if API_TOKEN:
        # Show first 8 and last 4 characters for security
        masked_token = f"{API_TOKEN[:8]}...{API_TOKEN[-4:]}" if len(API_TOKEN) > 12 else "***"
        logger.info(f"API token loaded successfully: {masked_token}")
        logger.info(f"API token length: {len(API_TOKEN)} characters")
    else:
        logger.error("API token not found in environment variables!")
        logger.error("Please set EMBED_API_TOKEN in your .env file or environment")
    return API_TOKEN is not None

# Cached model initialization to avoid reloading
@lru_cache(maxsize=1)
def get_model():
    """Load and cache the sentence transformer model"""
    logger.info(f"Loading model: {MODEL}")
    return SentenceTransformer(MODEL)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize models and debug API key on startup"""
    logger.info("Starting up embedding service...")
    
    # Debug API key
    api_key_ok = debug_api_key()
    if not api_key_ok:
        logger.warning("Service starting without valid API token - authentication will fail!")
    
    # Warm up models
    try:
        get_model()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

# Initialize app
app = FastAPI(
    title="Text Embedding API",
    description="API for generating text embeddings using multilingual E5 model",
    version="1.0.0",
    lifespan=lifespan,
)

# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192, description="Text to embed")

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    usage: dict

# Optimized token verification
async def verify_token(authorization: Optional[str] = Header(None, alias="Authorization")):
    """Verify Bearer token from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, 
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, 
            detail="Invalid authorization header format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    if not API_TOKEN:
        logger.error("API token not configured")
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error"
        )
    
    if token != API_TOKEN:
        logger.warning(f"Invalid token attempt: {token[:8]}...")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, 
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL,
        "api_key_configured": API_TOKEN is not None
    }

# Main embedding endpoint
@app.post("/embed", response_model=EmbeddingResponse)
async def get_embedding(
    data: TextInput, 
    _: None = Depends(verify_token)
) -> EmbeddingResponse:
    """Generate embeddings for input text"""
    try:
        # Get cached model (tokenizer is available from model internals)
        model = get_model()
        tokenizer = model.tokenizer
        
        # Prepare text with query prefix for E5 model
        text = f"query: {data.text.strip()}"
        
        # Run CPU/GPU-heavy inference in threadpool to avoid blocking event loop
        embedding = await run_in_threadpool(
            lambda: model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=False,
            ).tolist()
        )
        
        # Calculate token count in one tokenizer pass
        tokenized = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=True,
        )
        total_tokens = len(tokenized["input_ids"])
        
        logger.info(f"Generated embedding for text length: {len(data.text)}, tokens: {total_tokens}")
        
        return EmbeddingResponse(
            embedding=embedding,
            usage={
                "model": MODEL,
                "total_tokens": total_tokens,
                "text_length": len(data.text)
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to generate embedding"
        )

# Debug endpoint (remove in production)
@app.get("/debug/config")
async def debug_config(_: None = Depends(verify_token)):
    """Debug configuration endpoint"""
    return {
        "model": MODEL,
        "api_key_configured": API_TOKEN is not None,
        "api_key_length": len(API_TOKEN) if API_TOKEN else 0,
        "model_max_length": get_model().tokenizer.model_max_length
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
