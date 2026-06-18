"""
Pydantic request/response schemas for the inference API.

The request body mirrors the raw job-postings schema (the 31 columns of
``data/raw/postings.csv``), so a client posts the same shape the modeling
pipeline was trained on. Each posting flows through
``src.data.run_pipeline.prepare_data`` (except summarization, which reads the raw
``description``) before scoring, so only ``job_id`` is strictly required here --
every other field is optional and defaults to ``None`` so partial postings still
validate. The pipeline fills/encodes missing values downstream.

Responses are one object per input posting, keyed by ``job_id``, with that
track's prediction columns. The prediction value fields are optional so a
sanitized ``NaN`` (serialized as ``null``) never fails validation.
"""

from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class RawPosting(BaseModel):
    """A single raw job posting, matching the raw dataset schema.

    Only ``job_id`` is required; all other columns are optional because
    ``prepare_data`` imputes/encodes missing values. Unknown extra keys are
    ignored rather than rejected.
    """

    model_config = ConfigDict(extra="ignore")

    job_id: int

    company_name: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    max_salary: Optional[float] = None
    pay_period: Optional[str] = None
    location: Optional[str] = None
    company_id: Optional[float] = None
    views: Optional[float] = None
    med_salary: Optional[float] = None
    min_salary: Optional[float] = None
    formatted_work_type: Optional[str] = None
    applies: Optional[float] = None
    original_listed_time: Optional[float] = None
    remote_allowed: Optional[float] = None
    job_posting_url: Optional[str] = None
    application_url: Optional[str] = None
    application_type: Optional[str] = None
    expiry: Optional[float] = None
    closed_time: Optional[float] = None
    formatted_experience_level: Optional[str] = None
    skills_desc: Optional[str] = None
    listed_time: Optional[float] = None
    posting_domain: Optional[str] = None
    sponsored: Optional[float] = None
    work_type: Optional[str] = None
    currency: Optional[str] = None
    compensation_type: Optional[str] = None
    normalized_salary: Optional[float] = None
    zip_code: Optional[float] = None
    fips: Optional[float] = None


class PredictionRequest(BaseModel):
    """A batch of raw postings to score. Shared by every prediction route."""

    postings: List[RawPosting] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Per-task results (one per input posting)
# ---------------------------------------------------------------------------
class ExperienceLevelPrediction(BaseModel):
    job_id: int
    predicted_experience_level_ord: Optional[int] = None
    predicted_experience_level: Optional[str] = None


class SalaryPrediction(BaseModel):
    job_id: int
    predicted_salary: Optional[float] = None


class ClusterPrediction(BaseModel):
    job_id: int
    cluster: Optional[int] = None
    cluster_label: Optional[str] = None
    cluster_description: Optional[str] = None


class AnomalyPrediction(BaseModel):
    job_id: int
    anomaly_isolation_forest: Optional[int] = None
    anomaly_copod: Optional[int] = None
    anomaly_autoencoder: Optional[int] = None
    anomaly_vae: Optional[int] = None
    anomaly_score: Optional[int] = None


class SummaryResult(BaseModel):
    job_id: int
    summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel, Generic[T]):
    """Generic envelope wrapping one result object per input posting."""

    results: List[T]


# ---------------------------------------------------------------------------
# RAG (query -> answer + retrieved jobs)
# ---------------------------------------------------------------------------
class RagRequest(BaseModel):
    """A free-text question to answer over the job postings."""

    query: str = Field(..., min_length=1)


class RetrievedJob(BaseModel):
    """One posting retrieved for a RAG query, with its similarity score."""

    rank: int
    job_id: int
    score: float
    document: str


class RagResponse(BaseModel):
    """The Ollama answer plus the metadata of the jobs retrieved for the query."""

    query: str
    answer: str
    model: str
    retrieved: List[RetrievedJob]


class HealthResponse(BaseModel):
    """Reported by GET /health: liveness plus which model caches are warm."""

    status: str
    warm: dict
