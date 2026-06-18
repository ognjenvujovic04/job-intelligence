# Inference API

A FastAPI server that exposes the five modeling tracks over the unified
[`src.models.serve.inference`](../models/serve/inference.py) facade. Input is JSON in the raw
job-postings schema (the columns of `data/raw/postings.csv`); output is JSON
whose shape depends on the route.

A sixth route, `POST /rag`, is different in kind: it takes a free-text query
(not postings), retrieves similar jobs from a FAISS index, and generates an
answer with a local Ollama LLM — see [RAG](#rag-retrieval--ollama) below.

## Running

```powershell
venv\Scripts\activate
python -m src.api          # http://localhost:8000  (docs at /docs)
```

- Host/port are overridable via `API_HOST` / `API_PORT`.
- **Single worker only** — models are loaded into the process at startup, so
  extra workers would each reload them and multiply memory use.

## Running with Docker

The image is **self-contained** — it bakes in the served models, the precomputed
artifacts, and the HuggingFace models (domain BERT + T5), so it runs fully
offline with no MLflow DB. Build it from the repo root.

```powershell
# One-time host prep: generate the artifacts the image copies in.
python -m src.data.run_pipeline        # writes data/precomputed/
python -m scripts.export_models        # writes models/<track>/ from the pinned runs

docker build -t job-intelligence .
docker run -p 8000:8000 job-intelligence     # docs at http://localhost:8000/docs
```

- `scripts/export_models.py` promotes the four pinned MLflow runs
  (`tracking/config.py`) into a flat `models/` folder; the serving loaders read
  from there via `MODELS_DIR` (default `/app/models` in the image) instead of
  resolving `runs:/` through `mlflow.db`. Re-run it after retraining, then
  rebuild the image.
- Serving deps come from `requirements-serve.txt` (CPU-only torch, no
  tensorflow / training-only packages). See the repo-root `Dockerfile`.
- The `/rag` route needs an Ollama server on the **host**; the container reaches it
  via `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434`) and uses the
  model named by `OLLAMA_MODEL` (default `mistral` — pull it with `ollama pull mistral`,
  or point at a model you already have, e.g. `-e OLLAMA_MODEL=llama3.2:3b`). Add
  `--add-host=host.docker.internal:host-gateway` to `docker run` so that hostname
  resolves (required on Linux, and on any host where Docker doesn't provide it
  automatically). Ensure Ollama listens beyond loopback (`OLLAMA_HOST=0.0.0.0`) so the
  container can reach it. The nomic embedder is pre-baked into the image; no LLM weights
  ship in it.

### Warmup behavior

On startup (`lifespan`), the server calls `service.warmup()` to eagerly load the
four light-track models **and** the domain-classification BERT, so the first real
request is fast. **T5 summarization is not loaded at startup** — it loads lazily
on the first `/summarize` request and then stays resident (its model is cached
module-wide). Warmup is best-effort: a track that can't load (e.g. missing local
artifacts) logs a warning and is reported as not-warm by `/health` rather than
aborting startup.

## Routes

All prediction routes are `POST` and take the same batch body:

```json
{
  "postings": [
    { "job_id": 3906270, "title": "Associate Attorney", "description": "...", "max_salary": 120000, "pay_period": "YEARLY", "...": "..." }
  ]
}
```

Only `job_id` is required; every other raw column is optional (the pipeline
imputes/encodes the rest). Responses return one object per input posting under
`results`.

| Method & route | Result fields (besides `job_id`) |
|---|---|
| `POST /predict/experience-level` | `predicted_experience_level_ord`, `predicted_experience_level` |
| `POST /predict/salary` | `predicted_salary` |
| `POST /predict/clusters` | `cluster`, `cluster_label`, `cluster_description` |
| `POST /detect/anomalies` | `anomaly_isolation_forest`, `anomaly_copod`, `anomaly_autoencoder`, `anomaly_vae`, `anomaly_score` |
| `POST /summarize` | `summary` (T5; first call is slow) |
| `POST /rag` | takes `{ "query": ... }`; returns `answer` + `retrieved` jobs (see [RAG](#rag-retrieval--ollama)) |
| `GET /health` | `status`, `warm` (per-model cache flags) |

### Example

```powershell
$body = '{ "postings": [ { "job_id": 1, "title": "Data Scientist", "description": "We are hiring a data scientist to build ML models...", "pay_period": "YEARLY" } ] }'
Invoke-RestMethod -Uri http://localhost:8000/predict/experience-level -Method Post -ContentType application/json -Body $body
```

```json
{ "results": [ { "job_id": 1, "predicted_experience_level_ord": 3, "predicted_experience_level": "Mid-Senior level" } ] }
```

### RAG (retrieval + Ollama)

`POST /rag` takes a free-text query instead of postings and returns the LLM answer
plus the jobs retrieved as context:

```json
{ "query": "What remote ML jobs require Python and SQL?" }
```

```json
{
  "query": "What remote ML jobs require Python and SQL?",
  "answer": "Based on the postings, ...",
  "model": "mistral",
  "retrieved": [
    { "rank": 1, "job_id": 3871631334, "score": 0.657, "document": "search_document: Job Title: ..." }
  ]
}
```

Generation is delegated to an **Ollama server you run yourself** (`ollama serve`,
then pull a model). The API reaches it at `OLLAMA_BASE_URL` (default
`http://localhost:11434`; the Docker image defaults to `http://host.docker.internal:11434`
to reach the host) and uses the model named by `OLLAMA_MODEL` (default `mistral`). Both
are environment-overridable; the embedding model (`nomic-ai/nomic-embed-text-v1`) and
`top_k` (5) are fixed in [`src/models/serve/rag.py`](../models/serve/rag.py). Retrieval
uses the prebuilt FAISS index + metadata under `data/precomputed/rag/`. If Ollama can't
be reached, `/rag` returns **503** with a message naming the URL it tried (rather than a
generic 500).

## Prerequisites

Inference depends on local-only, gitignored artifacts existing on the host:
the exported `models/<track>/` folders (written by `python -m scripts.export_models`)
and `data/precomputed/` (fitted encoders, prototype embeddings, written by
`python -m src.data.run_pipeline`). The serving path loads models from
`models/` (override with `MODELS_DIR`) and does **not** read `mlflow.db` /
`mlruns/` — those are only needed by the export step. T5 also downloads ~500 MB
from HuggingFace on first use and is slow on CPU (pre-baked in the Docker image).

The `/rag` route additionally needs the prebuilt FAISS index + metadata under
`data/precomputed/rag/`, a reachable Ollama server, and the nomic embedder
(~550 MB on first use; pre-baked in the Docker image). See [RAG](#rag-retrieval--ollama).

## Layout

- `schemas.py` — Pydantic request/response models.
- `service.py` — adapters (postings → DataFrame → inference → JSON-safe dicts) and `warmup()`.
- `app.py` — `create_app()` factory, routes, lifespan warmup.
- `__main__.py` — `python -m src.api` entry point.

## Tests

`pytest tests/test_api.py` runs endpoint smoke tests via `TestClient` against the
synthetic postings (skipped if the local artifacts are absent). Use
`-m "not slow"` to skip the T5 `/summarize` test.
