# Job Intelligence

A Python project for collecting, analyzing, and serving job market data. It includes data processing pipelines, machine learning models, a FastAPI backend, and Jupyter notebooks for exploration.

## Project Structure

```
src/
  api/       # FastAPI application and routes
  data/      # Data loading and preprocessing
  models/    # ML models
  utils/     # Shared utilities
notebooks/   # Jupyter notebooks for exploration and analysis
scripts/     # Standalone scripts (data ingestion, etc.)
tests/       # Unit and integration tests
data/        # Raw and processed datasets
```

## Getting Started

Name notebooks using the following convention:
```
<step>.<sub-step>-<author-initials>-<short-description>.ipynb
```
Example: `1.0-sk-initial-data-exploration.ipynb`

Steps generally follow: `1` = data, `2` = features, `3` = modeling, `4` = evaluation.