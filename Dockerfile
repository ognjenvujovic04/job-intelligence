# syntax=docker/dockerfile:1
#
# Self-contained CPU inference image for the Job Intelligence API.
#
# The build context must already contain the exported models/ folder and
# data/precomputed/ -- generate them on the host first:
#     python -m src.data.run_pipeline        # writes data/precomputed/
#     python -m scripts.export_models        # writes models/<track>/
#
# Build & run:
#     docker build -t job-intelligence .
#     docker run -p 8000:8000 job-intelligence      # docs at http://localhost:8000/docs
FROM python:3.13-slim

# libgomp1 is required at runtime by LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV MODELS_DIR=/app/models \
    HF_HOME=/app/hf_cache \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install CPU-only torch first so the bare `torch` pin in requirements-serve.txt
# is already satisfied -- this avoids pulling the multi-GB CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Pre-bake the HuggingFace models (domain BERT + T5) into the image's HF cache so
# the container runs fully offline with no first-request download stall.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('0xnbk/nbk-ats-domain-v1-en')" \
    && python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; T5Tokenizer.from_pretrained('t5-base', legacy=True); T5ForConditionalGeneration.from_pretrained('t5-base')"

# Application code + the local artifacts inference loads at run time. Copied last
# so dependency layers stay cached across code/model changes.
COPY src/ ./src/
COPY models/ ./models/
COPY data/precomputed/ ./data/precomputed/

EXPOSE 8000

# Single worker on purpose: models load once into this process (see src/api/__main__.py).
CMD ["python", "-m", "src.api"]
