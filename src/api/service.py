"""
Service layer between the FastAPI routes and the model inference facade.

Each public function takes the validated request postings, builds a raw-postings
DataFrame, calls the matching ``src.models.inference`` function (which composes
``prepare_data`` + the cached model), then projects out only that track's
prediction columns as JSON-safe dicts -- numpy scalars become native Python and
non-finite floats (``NaN``/``inf``) become ``None``.

``warmup`` eagerly loads the four light tracks' models plus the domain BERT so
the first real request is fast. T5 is intentionally left out: it loads on the
first ``/summarize`` request and stays resident thereafter (see
``src.models.summarize._load_t5``).
"""

import logging
import math
import os

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SYNTHETIC_CSV = os.path.join(_REPO_ROOT, "data", "raw", "synthetic_postings.csv")

# Tracks the warmup outcome so GET /health can report what is resident. T5 is
# only flagged warm once the first summarize request has loaded it.
_WARM = {
    "domain": False,
    "experience_level": False,
    "salary": False,
    "clusters": False,
    "anomalies": False,
    "t5": False,
}

# Cache of RawPosting's numeric field names; populated by _numeric_columns().
_NUMERIC_COLS = None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def _numeric_columns():
    """Names of RawPosting fields typed as int/float (cached after first call).

    These must become NaN -- not Python None -- when a posting omits them, or the
    feature pipeline's numeric ops (e.g. np.log1p) choke on None.
    """
    global _NUMERIC_COLS
    if _NUMERIC_COLS is None:
        import typing

        from src.api.schemas import RawPosting

        cols = set()
        for name, field in RawPosting.model_fields.items():
            args = typing.get_args(field.annotation) or (field.annotation,)
            if any(a in (int, float) for a in args):
                cols.add(name)
        _NUMERIC_COLS = cols
    return _NUMERIC_COLS


def _to_dataframe(postings):
    """Turn a list of RawPosting models (or plain dicts) into a DataFrame.

    Numeric columns are coerced so omitted fields become NaN (the pipeline's
    expected "missing" value), mirroring what pandas.read_csv would produce.
    """
    import pandas as pd

    records = [p.model_dump() if hasattr(p, "model_dump") else dict(p) for p in postings]
    df = pd.DataFrame(records)
    for col in _numeric_columns():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clean_value(v):
    """Coerce a single cell to a JSON-safe native Python value."""
    import numpy as np

    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if v is None:
        return None
    # pandas may surface missing values as a float NaN already handled above,
    # but guard against pd.NA / NaT slipping through.
    try:
        import pandas as pd

        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _records(df, columns):
    """Project ``columns`` from ``df`` into a list of JSON-safe dicts."""
    present = [c for c in columns if c in df.columns]
    subset = df[present]
    return [
        {k: _clean_value(v) for k, v in row.items()}
        for row in subset.to_dict(orient="records")
    ]


# ---------------------------------------------------------------------------
# Per-task service functions
# ---------------------------------------------------------------------------
def predict_experience_level(postings):
    from src.models.inference import classify

    out = classify(_to_dataframe(postings), verbose=False)
    return _records(
        out, ["job_id", "predicted_experience_level_ord", "predicted_experience_level"]
    )


def predict_salary(postings):
    from src.models.inference import predict_salary as _predict_salary

    out = _predict_salary(_to_dataframe(postings), verbose=False)
    return _records(out, ["job_id", "predicted_salary"])


def predict_clusters(postings):
    from src.models.inference import predict_clusters as _predict_clusters

    out = _predict_clusters(_to_dataframe(postings), verbose=False)
    return _records(
        out, ["job_id", "cluster", "cluster_label", "cluster_description"]
    )


def detect_anomalies(postings):
    from src.models.inference import detect_anomalies as _detect_anomalies

    out = _detect_anomalies(_to_dataframe(postings), verbose=False)
    return _records(
        out,
        [
            "job_id",
            "anomaly_isolation_forest",
            "anomaly_copod",
            "anomaly_autoencoder",
            "anomaly_vae",
            "anomaly_score",
        ],
    )


def summarize(postings, **kwargs):
    from src.models.inference import summarize_postings

    out = summarize_postings(_to_dataframe(postings), **kwargs)
    _WARM["t5"] = True
    return _records(out, ["job_id", "summary"])


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
def _warmup_frame():
    """One raw posting to drive cache loading, from synthetic data if present."""
    import pandas as pd

    if os.path.exists(_SYNTHETIC_CSV):
        return pd.read_csv(_SYNTHETIC_CSV, nrows=1)
    logger.warning("Synthetic postings CSV missing at %s; using a minimal row", _SYNTHETIC_CSV)
    return pd.DataFrame(
        [{"job_id": 0, "title": "Warmup", "description": "Warmup posting " * 10}]
    )


def warmup():
    """Eagerly load the domain BERT and the four light-track models.

    Best-effort: a track that fails to warm (e.g. missing local artifacts) logs a
    warning and leaves its flag False rather than aborting startup. T5 is excluded
    by design. Returns the warm-status dict.
    """
    # Domain model first so it is resident even if a downstream track fails.
    try:
        from src.data.fe_domain_classification import warmup as _domain_warmup

        _domain_warmup()
        _WARM["domain"] = True
        logger.info("Warmed domain-classification model")
    except Exception as exc:  # noqa: BLE001 - startup must survive a cold track
        logger.warning("Domain model warmup failed: %s", exc)

    # The service functions expect a list of postings (RawPosting or dict), so
    # pass record dicts here -- not the DataFrame itself.
    postings = _warmup_frame().to_dict(orient="records")
    tracks = (
        ("experience_level", predict_experience_level),
        ("salary", predict_salary),
        ("clusters", predict_clusters),
        ("anomalies", detect_anomalies),
    )
    for name, fn in tracks:
        try:
            fn(postings)
            _WARM[name] = True
            logger.info("Warmed %s model", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s warmup failed: %s", name, exc)

    return get_warm_status()


def get_warm_status():
    """Return a copy of the current model-cache warm flags."""
    return dict(_WARM)
