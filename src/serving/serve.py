from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal, NoReturn

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.rag.answer import answer_complaint_question
from src.rag.retrieve import load_rag_artifacts


logger = logging.getLogger(__name__)

# Holds the current request's ID so any log line or error can reference it.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, so log tools can parse and filter it."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }

        for key in ("method", "path", "status", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging() -> None:
    # Attach to our own "src" logger only, so we never clobber the handlers
    # of uvicorn or pytest.
    app_logger = logging.getLogger("src")

    if app_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    app_logger.addHandler(handler)
    app_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    app_logger.propagate = False


configure_logging()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"

APP_VERSION = "0.1.0"


NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

MODEL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
        validate_by_alias=True,
    )

    age: int = Field(..., ge=16, le=100)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int = Field(..., ge=1)
    pdays: int = Field(..., ge=0)
    previous: int = Field(..., ge=0)
    poutcome: str

    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")


class PredictRequest(BaseModel):
    customer: CustomerFeatures


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    decision: Literal["likely_to_convert", "unlikely_to_convert"]
    conversion_band: Literal["low", "medium", "high"]
    model_version: str
    model_type: str


class AskComplaintsRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    product: str | None = None
    company: str | None = None
    issue: str | None = None
    state: str | None = None


class AskComplaintsResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    evidence_ids: list[str]
    retrieved_evidence: list[dict[str, Any]]
    evidence_sufficiency_note: str
    prompt_version: str
    index_version: str | None
    retrieval: dict[str, Any]


class CustomerIntelRequest(BaseModel):
    customer: CustomerFeatures
    complaint_question: str = Field(
        default="What are the main complaint themes for this customer segment?",
        min_length=3,
    )
    top_k: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    product: str | None = None
    company: str | None = None
    issue: str | None = None
    state: str | None = None


class CustomerIntelResponse(BaseModel):
    conversion: PredictResponse
    complaint_intelligence: AskComplaintsResponse
    integration_note: str


class BatchScoreRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=100)


class BatchPredictionItem(BaseModel):
    row_id: int
    prediction: int
    probability: float
    threshold: float
    decision: Literal["likely_to_convert", "unlikely_to_convert"]
    conversion_band: Literal["low", "medium", "high"]


class BatchScoreResponse(BaseModel):
    model_version: str
    model_type: str
    scored_count: int
    band_counts: dict[str, int]
    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_version: str
    model_loaded: bool
    model_version: str | None
    model_type: str | None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    rag_index_loaded: bool
    model_version: str | None = None
    index_version: str | None = None
    problems: list[str] = Field(default_factory=list)


_model_artifact: dict[str, Any] | None = None


def select_model_path() -> Path:
    if CHAMPION_MODEL_PATH.exists():
        return CHAMPION_MODEL_PATH

    if BASELINE_MODEL_PATH.exists():
        return BASELINE_MODEL_PATH

    raise FileNotFoundError(
        "No model artifact found. Run: python -m src.training.train "
        "and then python -m src.training.evaluate"
    )


def load_model_artifact(force_reload: bool = False) -> dict[str, Any]:
    global _model_artifact

    if _model_artifact is not None and not force_reload:
        return _model_artifact

    model_path = select_model_path()
    artifact = joblib.load(model_path)

    required_keys = {"model", "model_version", "threshold"}
    missing = required_keys - set(artifact.keys())

    if missing:
        raise ValueError(f"Model artifact missing keys: {sorted(missing)}")

    _model_artifact = artifact

    logger.info(
        "Loaded model %s from %s",
        artifact.get("model_version"),
        model_path.name,
    )

    return artifact


def get_model_type(artifact: dict[str, Any]) -> str:
    if artifact.get("model_type"):
        return str(artifact["model_type"])

    model = artifact.get("model")

    if model is None:
        return "unknown"

    return model.__class__.__name__


def customer_features_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    raw = customer.model_dump(by_alias=True)

    row = {}

    for column in MODEL_FEATURE_COLUMNS:
        if column not in raw:
            raise ValueError(f"Missing model feature after schema conversion: {column}")

        row[column] = raw[column]

    return pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)


def probability_to_band(probability: float) -> Literal["low", "medium", "high"]:
    if probability < 0.33:
        return "low"

    if probability < 0.66:
        return "medium"

    return "high"


def raise_internal_error(context: str, exc: Exception) -> NoReturn:
    """Log the real error server-side; give the client only a reference ID."""
    logger.error(context, exc_info=exc)

    raise HTTPException(
        status_code=500,
        detail=f"Internal error. Reference: {request_id_var.get()}",
    ) from exc


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        load_model_artifact(force_reload=True)
    except Exception:
        logger.exception("Model failed to load during startup")

    # Warm the RAG cache so the first user request is not slow.
    try:
        load_rag_artifacts()
    except Exception:
        logger.exception("RAG index failed to load during startup")

    yield


app = FastAPI(
    title="Customer Intelligence Platform API",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")

    if _REQUEST_ID_PATTERN.fullmatch(supplied):
        request_id = supplied
    else:
        request_id = uuid.uuid4().hex[:12]

    token = request_id_var.set(request_id)
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error")
        response = JSONResponse(
            status_code=500,
            content={"detail": f"Internal error. Reference: {request_id}"},
        )
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)

    # Probe traffic is frequent and noisy, so it only logs at DEBUG.
    level = logging.DEBUG if request.url.path.startswith("/health") else logging.INFO

    logger.log(
        level,
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id

    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        artifact = load_model_artifact()

        return HealthResponse(
            status="ok",
            app_version=APP_VERSION,
            model_loaded=True,
            model_version=str(artifact.get("model_version")),
            model_type=get_model_type(artifact),
        )

    except Exception:
        return HealthResponse(
            status="degraded",
            app_version=APP_VERSION,
            model_loaded=False,
            model_version=None,
            model_type=None,
        )


@app.get("/health/live")
def live() -> dict[str, str]:
    # Liveness: "the process is running". Never checks models, so a
    # missing artifact can't cause the platform to restart-loop us.
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadinessResponse)
def ready(response: Response) -> ReadinessResponse:
    # Readiness: "safe to send traffic to this replica".
    problems: list[str] = []
    model_version: str | None = None
    index_version: str | None = None

    try:
        artifact = load_model_artifact()
        model_version = str(artifact.get("model_version"))
    except Exception:
        logger.exception("Readiness check: model unavailable")
        problems.append("model_unavailable")

    try:
        _, _, metadata = load_rag_artifacts()
        index_version = metadata.get("index_version")
    except Exception:
        logger.exception("Readiness check: RAG index unavailable")
        problems.append("rag_index_unavailable")

    if problems:
        response.status_code = 503

    return ReadinessResponse(
        status="not_ready" if problems else "ready",
        model_loaded="model_unavailable" not in problems,
        rag_index_loaded="rag_index_unavailable" not in problems,
        model_version=model_version,
        index_version=index_version,
        problems=problems,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        artifact = load_model_artifact()
        model = artifact["model"]
        threshold = float(artifact.get("threshold", 0.5))

        features = customer_features_to_dataframe(request.customer)

        probability = float(model.predict_proba(features)[0][1])
        prediction = int(probability >= threshold)

        return PredictResponse(
            prediction=prediction,
            probability=probability,
            threshold=threshold,
            decision="likely_to_convert"
            if prediction == 1
            else "unlikely_to_convert",
            conversion_band=probability_to_band(probability),
            model_version=str(artifact.get("model_version")),
            model_type=get_model_type(artifact),
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise_internal_error("Prediction failed", exc)


@app.post("/batch-score", response_model=BatchScoreResponse)
def batch_score(request: BatchScoreRequest) -> BatchScoreResponse:
    try:
        artifact = load_model_artifact()
        model = artifact["model"]
        threshold = float(artifact.get("threshold", 0.5))

        rows = [
            customer_features_to_dataframe(customer)
            for customer in request.customers
        ]

        features = pd.concat(rows, ignore_index=True)
        probabilities = model.predict_proba(features)[:, 1]

        predictions: list[BatchPredictionItem] = []
        band_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
        }

        for row_id, probability in enumerate(probabilities):
            probability_float = float(probability)
            prediction = int(probability_float >= threshold)
            band = probability_to_band(probability_float)
            band_counts[band] += 1

            predictions.append(
                BatchPredictionItem(
                    row_id=row_id,
                    prediction=prediction,
                    probability=probability_float,
                    threshold=threshold,
                    decision="likely_to_convert"
                    if prediction == 1
                    else "unlikely_to_convert",
                    conversion_band=band,
                )
            )

        return BatchScoreResponse(
            model_version=str(artifact.get("model_version")),
            model_type=get_model_type(artifact),
            scored_count=len(predictions),
            band_counts=band_counts,
            predictions=predictions,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise_internal_error("Batch scoring failed", exc)


@app.post("/ask-complaints", response_model=AskComplaintsResponse)
def ask_complaints(request: AskComplaintsRequest) -> AskComplaintsResponse:
    try:
        result = answer_complaint_question(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
            product=request.product,
            company=request.company,
            issue=request.issue,
            state=request.state,
        )

        return AskComplaintsResponse(**result)

    except FileNotFoundError as exc:
        logger.error("RAG index not ready", exc_info=exc)
        raise HTTPException(
            status_code=503,
            detail="RAG index is not ready.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise_internal_error("Complaint answer failed", exc)


@app.post("/customer-intel", response_model=CustomerIntelResponse)
def customer_intel(request: CustomerIntelRequest) -> CustomerIntelResponse:
    try:
        prediction_result = predict(PredictRequest(customer=request.customer))

        complaint_result = answer_complaint_question(
            question=request.complaint_question,
            top_k=request.top_k,
            min_score=request.min_score,
            product=request.product,
            company=request.company,
            issue=request.issue,
            state=request.state,
        )

        complaint_response = AskComplaintsResponse(**complaint_result)

        return CustomerIntelResponse(
            conversion=prediction_result,
            complaint_intelligence=complaint_response,
            integration_note=(
                "This response combines campaign conversion scoring with grounded "
                "complaint intelligence. Complaint evidence is limited to retrieved "
                "records and should not be treated as legal or financial advice."
            ),
        )

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        logger.error("Required artifact not ready", exc_info=exc)
        raise HTTPException(
            status_code=503,
            detail="A required artifact is not ready.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise_internal_error("Customer intelligence request failed", exc)