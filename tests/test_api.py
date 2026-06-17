"""
Endpoint smoke tests for the FastAPI inference API.

Like ``test_clustering_synthetic.py``, this is an end-to-end test against the
local-only artifacts (mlflow.db, mlruns/, data/precomputed/) and the synthetic
postings CSV. When any of those are missing the whole module is skipped rather
than failing, so a fresh checkout without the regenerated artifacts stays green.

The ``client`` fixture enters the TestClient context so the app's lifespan runs
``service.warmup()`` once -- loading the four light-track models and the domain
BERT before the prediction tests. The ``/summarize`` test is marked slow because
it triggers the lazy T5 load (~500 MB download, slow on CPU).

Run:
    pytest tests/test_api.py
    pytest tests/test_api.py -m "not slow"     # skip the T5 summarize test
"""

import math
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SYNTHETIC_CSV = os.path.join(REPO_ROOT, "data", "raw", "synthetic_postings.csv")

# Local-only artifacts the inference path depends on (all gitignored).
_REQUIRED_ARTIFACTS = [
    SYNTHETIC_CSV,
    os.path.join(REPO_ROOT, "mlflow.db"),
    os.path.join(REPO_ROOT, "data", "precomputed", "prep_artifacts.joblib"),
    os.path.join(REPO_ROOT, "data", "precomputed", "feature_columns.json"),
]
_missing = [p for p in _REQUIRED_ARTIFACTS if not os.path.exists(p)]

pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"missing local artifacts: {_missing}",
)


def _payload(n=3):
    """First ``n`` synthetic postings as a JSON-safe PredictionRequest body."""
    import pandas as pd

    df = pd.read_csv(SYNTHETIC_CSV).head(n)
    records = [
        {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()
        }
        for row in df.to_dict(orient="records")
    ]
    return {"postings": records}, len(records)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    # The context manager runs startup (warmup) and shutdown around the tests.
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # The four light tracks + domain should be warm after lifespan startup.
    assert body["warm"]["domain"] is True
    assert body["warm"]["experience_level"] is True


@pytest.mark.parametrize(
    "route, expected_keys",
    [
        (
            "/predict/experience-level",
            {"job_id", "predicted_experience_level_ord", "predicted_experience_level"},
        ),
        ("/predict/salary", {"job_id", "predicted_salary"}),
        (
            "/predict/clusters",
            {"job_id", "cluster", "cluster_label", "cluster_description"},
        ),
        (
            "/detect/anomalies",
            {
                "job_id",
                "anomaly_isolation_forest",
                "anomaly_copod",
                "anomaly_autoencoder",
                "anomaly_vae",
                "anomaly_score",
            },
        ),
    ],
)
def test_prediction_routes(client, route, expected_keys):
    body, n = _payload()
    resp = client.post(route, json=body)
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == n
    for item in results:
        assert expected_keys.issubset(item.keys())


def test_validation_rejects_empty_batch(client):
    resp = client.post("/predict/salary", json={"postings": []})
    assert resp.status_code == 422


@pytest.mark.slow
def test_summarize(client):
    body, n = _payload(n=2)
    resp = client.post("/summarize", json=body)
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == n
    for item in results:
        assert {"job_id", "summary"}.issubset(item.keys())
