# Job Intelligence

An end-to-end ML project over a large job-postings dataset. One shared
feature-engineering pipeline feeds several modeling tracks, all of which are
served from a single FastAPI inference server:

| Track | Type | Target / output |
|---|---|---|
| Experience-level classification | Ordinal classification | `experience_level_ord` (Internship → Executive) |
| Salary regression | Regression | `normalized_salary` |
| Job clustering | Unsupervised (KMeans, k=26) | cluster id + human label |
| Anomaly detection | Unsupervised ensemble | per-detector flags + consensus score |
| Summarization | Abstractive (T5) | short posting summary |

Work flows through a versioned data pipeline, an MLflow-tracked training package,
numbered exploration notebooks, and the inference API.

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

Notebooks are named `<step>.<sub-step>-<author-initials>-<short-description>.ipynb`. 

The step prefix marks the stage:
**1** = data, **2** = features, **3** = modeling, **4** = evaluation.

## Setup

A `venv/` is checked out at the repo root. Activate it, then install deps:

```powershell
venv\Scripts\activate                 # PowerShell / cmd
pip install -r requirements.txt       # full stack (training + heavy NLP)
```

- Heavy NLP deps (`torch`, `transformers`, `sentence-transformers`) are only
  needed for the domain-classification and summarization steps.
- For **serving only**, use `requirements-serve.txt` (CPU-only torch, no
  TensorFlow / training-only packages) - this is what the Docker image installs.

## Usage

Run all `-m` commands from the **repo root**.

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

**Serving:** see **[src/api/README.md](src/api/README.md)** for the inference
server (local + Docker), the full route table, and example requests. In short:

```powershell
python -m src.api          # http://localhost:8000  (interactive docs at /docs)
```

**Tests:**

```powershell
pytest                                         # all tests
pytest -m "not slow"                           # skip the T5 summarization test
```

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
