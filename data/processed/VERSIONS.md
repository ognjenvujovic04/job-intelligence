# Dataset Versioning

All versions live under `data/processed/vN/`, most as a
`feature_matrix_train.csv` / `feature_matrix_test.csv` pair (80/20 split).
v6 is the exception: a single `feature_matrix.csv` with no split (see Split
column below).

| Version | `normalized_salary` | `salary_is_missing` flag | `job_id` | Split |
|---------|---------------------|--------------------------|----------|-------|
| v1 | NaN (original) | - | - | train/test |
| v2 | classification regression fill (no experience col) | - | - | train/test |
| v3 | classification regression fill (no experience col) | ✓ | - | train/test |
| v4 | NaN (original) | ✓ | - | train/test |
| v5 | NaN (original) | - | ✓ | train/test |
| v6 | full regression fill (all features) | - | ✓ | single file (no split) |
| v7 | full regression fill (all features) | - | ✓ | train/test |

## Notes

- **Two salary-regression fills.** There are two different imputation models
  behind the `normalized_salary` column:
  - **classification regression fill (v2, v3)** — the salary regressor is trained
    **without the experience-level column**, so the imputed salary can be fed as a
    feature into experience-level classification without leaking the target.
  - **full regression fill (v6, v7)** — the salary regressor uses **all features**
    (including experience level); it has the better MAE and is the right choice
    when salary is not being used to predict experience level (clustering,
    anomaly detection).
- **v5** is a re-export of v1 with `job_id` retained so that cluster labels
  can be joined back to `cleaned_job_postings.csv` for interpretation.
- **v6** is the canonical clustering input: salary is fully imputed, the
  `salary_is_missing` flag records which values were originally absent, and
  `job_id` is preserved for post-hoc cluster analysis.
  Regression model: MLflow run `d3867edc88124180aac8ae54eb294dce`
  (experiment `salary-regression`).
- Generate v6 with:
  ```
  python -m src.data.prepare_clustering_dataset
  ```
- **v7** is the canonical anomaly-detection input: the v5 80/20 train/test split
  is retained (anomaly detection fits its scaler on train and derives
  reconstruction thresholds from train scores), with `normalized_salary` replaced
  by v6's regressed values (mapped on `job_id`) instead of a flat median fill.
- Generate v7 with:
  ```
  python -m src.data.prepare_anomaly_dataset
  ```
