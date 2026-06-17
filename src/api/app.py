"""
FastAPI application for the Job Intelligence inference API.

One POST route per modeling track, each accepting a batch of raw postings
(``PredictionRequest``) and returning one result per posting. Route handlers are
sync ``def`` so FastAPI runs the CPU-bound inference in its threadpool instead of
blocking the event loop.

On startup the ``lifespan`` handler calls ``service.warmup()`` to load the four
light-track models and the domain BERT, so the first real request is fast. T5
loads lazily on the first ``/summarize`` call and then stays resident.

Run it with ``python -m src.api`` (see ``__main__``). Keep it to a single worker:
the models are held in-process, so extra workers would duplicate them in memory.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api import service
from src.api.schemas import (
    AnomalyPrediction,
    ClusterPrediction,
    ExperienceLevelPrediction,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    SalaryPrediction,
    SummaryResult,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming inference models...")
    warm = service.warmup()
    logger.info("Warmup complete: %s", warm)
    yield


def create_app() -> FastAPI:
    """Application factory. Used by ``python -m src.api`` and the test client."""
    app = FastAPI(
        title="Job Intelligence Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", warm=service.get_warm_status())

    @app.post(
        "/predict/experience-level",
        response_model=PredictionResponse[ExperienceLevelPrediction],
    )
    def predict_experience_level(req: PredictionRequest):
        return {"results": service.predict_experience_level(req.postings)}

    @app.post("/predict/salary", response_model=PredictionResponse[SalaryPrediction])
    def predict_salary(req: PredictionRequest):
        return {"results": service.predict_salary(req.postings)}

    @app.post(
        "/predict/clusters", response_model=PredictionResponse[ClusterPrediction]
    )
    def predict_clusters(req: PredictionRequest):
        return {"results": service.predict_clusters(req.postings)}

    @app.post("/detect/anomalies", response_model=PredictionResponse[AnomalyPrediction])
    def detect_anomalies(req: PredictionRequest):
        return {"results": service.detect_anomalies(req.postings)}

    @app.post("/summarize", response_model=PredictionResponse[SummaryResult])
    def summarize(req: PredictionRequest):
        return {"results": service.summarize(req.postings)}

    return app
