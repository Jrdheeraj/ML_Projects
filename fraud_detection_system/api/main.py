"""FastAPI application for fraud scoring."""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response

from src.config.configuration import AppConfig
from src.logger.logging import get_logger, setup_logger
from src.pipelines.prediction_pipeline import run_prediction_pipeline


class PredictRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    user_id: int = Field(..., ge=1)
    transaction_amount: float = Field(..., gt=0)
    transaction_time: str = Field(..., min_length=4)
    location: str = Field(..., min_length=1)
    device: str = Field(..., min_length=1)
    merchant_category: str = Field(..., min_length=1)
    payment_channel: str = Field(..., min_length=1)
    is_international: int = Field(..., ge=0, le=1)
    card_present: int = Field(..., ge=0, le=1)
    previous_transactions_24h: int = Field(..., ge=0)
    avg_spend_7d: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    fraud: bool
    probability: float
    threshold: float
    model_version: str


config = AppConfig.from_env()
setup_logger(config.log_dir)
logger = get_logger(__name__)
app = FastAPI(title="Fraud Detection API", version="1.0.0")


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((perf_counter() - start) * 1000.0, 2)
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        "request_completed path=%s method=%s status=%s duration_ms=%s client=%s",
        request.url.path,
        request.method,
        response.status_code,
        duration_ms,
        client_host,
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = run_prediction_pipeline(payload.model_dump(), config=config)
        return PredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Fraud prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
