from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.rag.answer import answer_complaint_question


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
    model_path: str


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
    model_path: str
    scored_count: int
    band_counts: dict[str, int]
    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_version: str
    model_loaded: bool
    model_version: str | None
    model_type: str | None
    model_path: str | None


_model_artifact: dict[str, Any] | None = None
_model_path: Path | None = None


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
    global _model_artifact, _model_path

    if _model_artifact is not None and not force_reload:
        return _model_artifact

    model_path = select_model_path()
    artifact = joblib.load(model_path)

    required_keys = {"model", "model_version", "threshold"}
    missing = required_keys - set(artifact.keys())

    if missing:
        raise ValueError(f"Model artifact missing keys: {sorted(missing)}")

    _model_artifact = artifact
    _model_path = model_path

    return artifact


def get_model_type(artifact: dict[str, Any]) -> str:
    if artifact.get("model_type"):
        return str(artifact["model_type"])

    model = artifact.get("model")

    if model is None:
        return "unknown"

    return model.__class__.__name__


def get_current_model_path() -> str | None:
    if _model_path is None:
        return None

    return str(_model_path.relative_to(PROJECT_ROOT))


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


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        load_model_artifact(force_reload=True)
    except Exception as exc:
        print(f"Model failed to load during startup: {exc}")

    yield


app = FastAPI(
    title="Customer Intelligence Platform API",
    version=APP_VERSION,
    lifespan=lifespan,
)


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
            model_path=get_current_model_path(),
        )

    except Exception:
        return HealthResponse(
            status="degraded",
            app_version=APP_VERSION,
            model_loaded=False,
            model_version=None,
            model_type=None,
            model_path=None,
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
            model_path=get_current_model_path() or "unknown",
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


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
            model_path=get_current_model_path() or "unknown",
            scored_count=len(predictions),
            band_counts=band_counts,
            predictions=predictions,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch scoring failed: {exc}",
        ) from exc


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
        raise HTTPException(
            status_code=503,
            detail=f"RAG index is not ready: {exc}",
        ) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Complaint answer failed: {exc}",
        ) from exc


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
        raise HTTPException(
            status_code=503,
            detail=f"Required artifact is not ready: {exc}",
        ) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Customer intelligence request failed: {exc}",
        ) from exc