"""FastAPI app for churn prediction serving."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel

from src.config.configuration import AppConfig
from src.logger.logging import get_logger, setup_logger
from src.pipelines.prediction_pipeline import run_prediction_pipeline


class PredictRequest(RootModel[dict[str, Any]]):
    """Raw customer feature payload for a single prediction."""


class PredictResponseItem(BaseModel):
    churn: str
    probability: float


class PredictResponse(BaseModel):
    predictions: list[PredictResponseItem]


config = AppConfig.from_env()
setup_logger(config.log_dir)
logger = get_logger(__name__)
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        prediction = run_prediction_pipeline(payload.root, config=config)
        return PredictResponse(predictions=[PredictResponseItem(**prediction)])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
