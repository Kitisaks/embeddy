FROM python:3.10-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system packages
RUN apt-get update && apt-get install -y \
  git && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy app files
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Expose FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "auto"]
