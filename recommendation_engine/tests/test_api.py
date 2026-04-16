from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api.main as recommendation_api
from src.config.configuration import AppConfig
from src.pipelines.training_pipeline import run_training_pipeline


def test_health_endpoint() -> None:
    client = TestClient(recommendation_api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_endpoint(tmp_path: Path) -> None:
    project_root = tmp_path
    repo_root = Path(__file__).resolve().parents[1]

    config = AppConfig.from_env()
    config.project_root = project_root
    config.data.interactions_path = repo_root / "data" / "raw" / "interactions.csv"
    config.data.items_path = repo_root / "data" / "raw" / "items.csv"
    config.data.processed_interactions_path = project_root / "data" / "processed" / "interactions_processed.csv"
    config.data.processed_items_path = project_root / "data" / "processed" / "items_processed.csv"
    config.model.model_path = project_root / "models" / "hybrid_recommender.joblib"
    config.model.metrics_path = project_root / "artifacts" / "metrics.json"
    config.log_dir = project_root / "artifacts" / "logs"

    run_training_pipeline(config)
    recommendation_api.config = config

    client = TestClient(recommendation_api.app)
    response = client.post("/recommend", json={"user_id": 101, "top_n": 3})
    assert response.status_code == 200
    payload = response.json()
    assert "recommendations" in payload
    assert len(payload["recommendations"]) == 3


def test_recommend_endpoint_rejects_invalid_user_id() -> None:
    client = TestClient(recommendation_api.app)
    response = client.post("/recommend", json={"user_id": 0, "top_n": 3})
    assert response.status_code == 422
