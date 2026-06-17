# Inference API

A FastAPI server that exposes the five modeling tracks over the unified
[`src.models.inference`](../models/inference.py) facade. Input is JSON in the raw
job-postings schema (the columns of `data/raw/postings.csv`); output is JSON
whose shape depends on the route.

## Running

```powershell
venv\Scripts\activate
python -m src.api          # http://localhost:8000  (docs at /docs)
```

- Host/port are overridable via `API_HOST` / `API_PORT`.
- **Single worker only** — models are loaded into the process at startup, so
  extra workers would each reload them and multiply memory use. (This is also why
  the server is the next thing slated for dockerization, with a single worker.)

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
| `GET /health` | `status`, `warm` (per-model cache flags) |

### Example

```powershell
$body = '{ "postings": [ { "job_id": 1, "title": "Data Scientist", "description": "We are hiring a data scientist to build ML models...", "pay_period": "YEARLY" } ] }'
Invoke-RestMethod -Uri http://localhost:8000/predict/experience-level -Method Post -ContentType application/json -Body $body
```

```json
{ "results": [ { "job_id": 1, "predicted_experience_level_ord": 3, "predicted_experience_level": "Mid-Senior level" } ] }
```

## Prerequisites

Inference depends on local-only, gitignored artifacts existing on the host:
`mlflow.db`, the `mlruns/` logged models, and `data/precomputed/` (fitted
encoders, prototype embeddings). Regenerate them with `python -m src.data.run_pipeline`
and the training entry points if missing. T5 also downloads ~500 MB from
HuggingFace on first use and is slow on CPU.

## Layout

- `schemas.py` — Pydantic request/response models.
- `service.py` — adapters (postings → DataFrame → inference → JSON-safe dicts) and `warmup()`.
- `app.py` — `create_app()` factory, routes, lifespan warmup.
- `__main__.py` — `python -m src.api` entry point.

## Tests

`pytest tests/test_api.py` runs endpoint smoke tests via `TestClient` against the
synthetic postings (skipped if the local artifacts are absent). Use
`-m "not slow"` to skip the T5 `/summarize` test.
