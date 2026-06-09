# Dataset Versioning

All versions live under `data/processed/vN/` as a
`feature_matrix_train.csv` / `feature_matrix_test.csv` pair (80/20 split).

| Version | `normalized_salary` | `salary_is_missing` flag | `job_id` |
|---------|---------------------|--------------------------|----------|
| v1 | NaN (original) | - | - |
| v2 | LightGBM regression fill | - | - |
| v3 | LightGBM regression fill | ✓ | - |
| v4 | NaN (original) | ✓ | - |
| v5 | NaN (original) | - | ✓ |
| v6 | LightGBM regression fill | - | ✓ |

## Notes

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
