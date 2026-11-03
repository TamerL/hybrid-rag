FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download models at build time (optional; increases image size)
# RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
#     SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
#     CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default: run the API (indexes are built on first request / lifespan)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
