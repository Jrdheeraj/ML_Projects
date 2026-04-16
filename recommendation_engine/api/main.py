"""FastAPI app for recommendation serving."""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response

from src.config.configuration import AppConfig
from src.logger.logging import get_logger, setup_logger
from src.pipelines.recommendation_pipeline import run_recommendation_pipeline


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1, le=10_000_000_000)
    top_n: int = Field(default=10, ge=1, le=50)


class RecommendResponse(BaseModel):
    recommendations: list[str]


config = AppConfig.from_env()
setup_logger(config.log_dir)
logger = get_logger(__name__)
app = FastAPI(title="Recommendation Engine API", version="1.0.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((perf_counter() - start) * 1000.0, 2)
    logger.info(
        "request_completed path=%s method=%s status=%s duration_ms=%s",
        request.url.path,
        request.method,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        recommendations = run_recommendation_pipeline(payload.user_id, top_n=payload.top_n, config=config)
        return RecommendResponse(recommendations=recommendations)
    except ValueError as exc:
        logger.warning("Invalid recommendation request user_id=%s top_n=%s", payload.user_id, payload.top_n)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Recommendation request failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
