# Job Intelligence

**An end-to-end ML system over a large job-postings dataset - five modeling
tracks plus retrieval-augmented Q&A, all served from one FastAPI server.**

One shared feature-engineering pipeline feeds every track, training is
MLflow-tracked, and the whole inference stack ships as a self-contained Docker
image. At a glance, this project demonstrates:

- **Breadth of modeling** - ordinal classification, regression, clustering,
  an unsupervised anomaly-detection ensemble, and abstractive summarization.
- **A modern RAG endpoint** - FAISS retrieval + local-LLM generation over the
  posting corpus.
- **Production shape** - versioned datasets, a tracked training package, and a
  single offline-capable inference API behind one facade.

| Track | Type | Target / output |
|---|---|---|
| Experience-level classification | Ordinal classification | `experience_level_ord` (Internship → Executive) |
| Salary regression | Regression | `normalized_salary` |
| Job clustering | Unsupervised (KMeans, k=26) | cluster id + human label |
| Anomaly detection | Unsupervised ensemble | per-detector flags + consensus score |
| Summarization | Abstractive (T5) | short posting summary |

### Retrieval-augmented Q&A (RAG)

Beyond the prediction tracks, the API answers free-text questions about the
postings. It embeds the query (`nomic-ai/nomic-embed-text-v1`), retrieves the
most similar jobs from a prebuilt **FAISS** index, and has a local
[Ollama](https://ollama.com) LLM generate an answer grounded in those jobs:

```jsonc
// POST /rag
{ "query": "What remote ML jobs require Python and SQL?" }
// ->
{
  "answer": "Based on the retrieved postings, several remote ML roles list Python and SQL...",
  "model": "mistral",
  "retrieved": [ { "rank": 1, "job_id": 3871631334, "score": 0.657, "document": "..." } ]
}
```

Unlike the other tracks it takes a query string rather than postings, and the
answer cites the jobs it retrieved as context.

## Data

Built on the [LinkedIn Job Postings dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
(~124k postings, `data/raw/postings.csv`). The `data/` tree is gitignored -
download the source and drop `postings.csv` there before running the pipeline.

## Key findings

Metrics below are from the production runs currently pinned for serving
(`src/models/tracking/config.py`); reproduce via the MLflow UI or the numbered
notebooks (stage 3 modeling, stage 4 evaluation).

- **Experience-level classification** - Optuna-tuned **LightGBM** over 43
  features (~79k train rows) across 6 ordinal levels. Accuracy **0.68**,
  macro-F1 **0.57**, ordinal MAE **0.53**. Performance is strongest on the
  well-populated middle of the distribution (entry-level F1 0.74, mid-senior
  0.69) and weakest at the sparse senior end (director/executive F1 ≈ 0.36-0.38)
  - a class-imbalance effect.
- **Salary regression** - **LightGBM** on `normalized_salary`. R² **0.49**,
  MAE **≈ $26.2k**, RMSE **≈ $43.3k**. Salary is moderately predictable from
  posting features but with wide residuals, reflecting how much pay variance
  isn't captured by the posting text/metadata.
- **Job clustering** - **KMeans, k=26** over a 40-feature matrix (~123k
  postings); the full imputer → scaler → KMeans pipeline is logged so a raw
  feature matrix maps straight to a cluster id. Silhouette **0.24**,
  Davies-Bouldin **1.20**. `k` was chosen from a sweep (`artifacts/kmeans_sweep.png`),
  with HDBSCAN evaluated as an alternative (`artifacts/hdbscan_sweep.png`). Clusters
  are interpreted via top-words / wordclouds (notebook 4.0,
  `scripts/analyze_cluster_top_words.py`).
- **Anomaly detection** - unsupervised **ensemble of four detectors** (Isolation
  Forest, COPOD, Autoencoder, VAE) at 0.5% contamination, combined into a 0-4
  consensus score. On the test split, 130 postings were flagged by ≥2 detectors
  and 9 by all four, surfacing the most clearly atypical postings.
- **Summarization** - abstractive **T5-base** with a domain-aware task prefix,
  loaded lazily in the serving layer to keep startup light.

## Quickstart

Serve the inference API locally. It warms the light-track models + domain BERT
on startup; T5 and the RAG embedder load lazily.

```powershell
venv\Scripts\activate
pip install -r requirements-serve.txt      # CPU-only serving stack
python -m src.api                          # http://localhost:8000  (docs at /docs)
```

Then hit a route - e.g. predict experience level:

```powershell
$body = '{ "postings": [ { "job_id": 1, "title": "Data Scientist", "description": "We are hiring...", "pay_period": "YEARLY" } ] }'
Invoke-RestMethod -Uri http://localhost:8000/predict/experience-level -Method Post -ContentType application/json -Body $body
```

Inference reads local-only artifacts (`models/<track>/`, `data/precomputed/`);
generate them with the pipeline below if they aren't present. See
**[src/api/README.md](src/api/README.md)** for the full route table and request
bodies.

<details>
<summary><b>Full pipeline, training & Docker</b></summary>

### Install

```powershell
venv\Scripts\activate                 # PowerShell / cmd
pip install -r requirements.txt       # full stack (training + heavy NLP)
```

Heavy NLP deps (`torch`, `transformers`, `sentence-transformers`) are only
needed for the domain-classification and summarization steps. For serving only,
`requirements-serve.txt` is enough (CPU-only torch, no TensorFlow) - that's what
the Docker image installs.

### Train & track (run all `-m` commands from the repo root)

```powershell
# 1. Data pipeline: clean -> domain classify -> feature-engineer (80/20 split).
#    Fits and persists encoders/caps/feature columns to data/precomputed/.
python -m src.data.run_pipeline

# 2. Train + log a model to MLflow.
python -m src.models.tracking --model-type lightgbm --run-name lgbm-v1
python -m src.models.tracking --all          # all model types in sequence

# 3. Build the v6 clustering dataset (salary imputed via a logged regression run).
python -m src.data.prepare_clustering_dataset

# 4. Inspect runs in the MLflow UI.
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

### Docker

The image is self-contained - it bakes in the served models, the precomputed
artifacts, and the HuggingFace weights (domain BERT, T5, the nomic RAG
embedder), so the container runs fully offline with no MLflow DB.

```powershell
# One-time host prep (writes data/precomputed/ and models/<track>/).
python -m src.data.run_pipeline
python -m scripts.export_models

docker build -t job-intelligence .
docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway job-intelligence
# docs at http://localhost:8000/docs
```

The `/rag` route needs an [Ollama](https://ollama.com) server running on the
**host** (it isn't bundled). Pull a model first (`ollama pull mistral`), serve it
beyond loopback (`OLLAMA_HOST=0.0.0.0 ollama serve`), and the container reaches it
via `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434`) - that's what
the `--add-host` flag resolves. Point at a different model with
`-e OLLAMA_MODEL=llama3.2:3b`. The other tracks work without Ollama.

### Tests

```powershell
pytest                                         # all tests
pytest -m "not slow"                           # skip the T5 summarization test
```

</details>

## Repository structure

```
src/
  api/                   # FastAPI inference server
  data/                  # data pipeline (clean, domain tag, features)
  models/                # modeling code, split by stage
    serve/                 # inference: load model -> predict
    tracking/              # MLflow tracking package
    train/                 # per-model trainers
    mlflow_tracking.py     # backward-compat shim re-exporting the tracking package
  utils/                 # shared utilities
notebooks/               # exploration, numbered by stage
scripts/                 # standalone tools
tests/                   # integration smoke tests
data/                    # raw / processed datasets - gitignored
models/                  # exported served models - gitignored
artifacts/               # analysis plots
```

Notebooks are named `<step>.<sub-step>-<author-initials>-<short-description>.ipynb`,
where the step prefix marks the stage: **1** = data, **2** = features,
**3** = modeling, **4** = evaluation.

## Future improvements

- **Senior-level classification.** Address director/executive under-performance
  with class weighting, resampling, or merging the sparsest levels.
- **Richer salary features.** R² ≈ 0.49 leaves headroom - explore location
  cost-of-living, company size, and stronger text features.
- **Anomaly evaluation.** The ensemble is unsupervised; a small labeled set
  would let us tune contamination and the consensus threshold against ground truth.
- **Serving scalability.** The API is single-worker by design (models are
  process-resident). A shared model server or multi-process layout would lift
  throughput.

## More documentation

- **[src/api/README.md](src/api/README.md)** - inference server: routes, Docker, warmup.
- **[src/models/tracking/README.md](src/models/tracking/README.md)** - MLflow tracking package API.
- **[data/processed/VERSIONS.md](data/processed/VERSIONS.md)** - dataset version table.

## Acknowledgments

- Domain classification is built on [**0xnbk/nbk-ats-domain-v1-en**](https://huggingface.co/0xnbk/nbk-ats-domain-v1-en)
  by [0xnbk](https://huggingface.co/0xnbk) - thanks for sharing the model that powers
  the zero-shot domain tagging in `fe_domain_classification.py`.
