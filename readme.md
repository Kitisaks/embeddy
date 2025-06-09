# 🌍 Multilingual Embedding API

A FastAPI-powered service for generating multilingual sentence embeddings using the [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) model. Supports local inference, token-based authentication via API keys, and can be deployed easily using Docker.

---

## 🚀 Features

- Supports 100+ languages
- Local embedding generation (no external API calls)
- Cosine similarity-ready embeddings
- API key authentication
- Dockerized for easy deployment

---

## 📦 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kitisaks/embeddy.git
cd embeddy
```

### 2. Generate an API Key

add it to `.env`:

```bash
EMBED_API_TOKEN=yourtokensecret
```

### 3. Build & Run with Docker

```bash
docker build -t embeddy .
docker run -p 8000:8000 embeddy
```

# 📚 References

- [SentenceTransformers](https://www.sbert.net/)

- [FastAPI Docs](https://fastapi.tiangolo.com/)

- [Hugging Face Model](https://huggingface.co/intfloat/multilingual-e5-base)
