import os
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.status import HTTP_401_UNAUTHORIZED
from dotenv import load_dotenv
load_dotenv()

# Load token from environment variable
API_TOKEN = os.environ.get("EMBED_API_TOKEN")

app = FastAPI()
model = SentenceTransformer("intfloat/multilingual-e5-large")

class TextInput(BaseModel):
    text: str

# Dependency to check token
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid auth header format")
    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/embed")
def get_embedding(data: TextInput, _: None = Depends(verify_token)):
    text = f"query: {data.text.strip()}"
    embedding = model.encode(text, normalize_embeddings=True).tolist()
    return {"embedding": embedding}
