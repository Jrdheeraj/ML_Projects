from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api.main as fraud_api
from src.config.configuration import AppConfig
from src.pipelines.training_pipeline import run_training_pipeline


def test_health_endpoint() -> None:
    client = TestClient(fraud_api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(tmp_path: Path) -> None:
    project_root = tmp_path
    repo_root = Path(__file__).resolve().parents[1]

    config = AppConfig.from_env()
    config.project_root = project_root
    config.data.raw_data_path = repo_root / "data" / "raw" / "transactions.csv"
    config.data.train_data_path = project_root / "data" / "processed" / "train.csv"
    config.data.test_data_path = project_root / "data" / "processed" / "test.csv"
    config.model.model_path = project_root / "models" / "fraud_model.joblib"
    config.model.metrics_path = project_root / "artifacts" / "metrics.json"
    config.model.feature_importance_path = project_root / "artifacts" / "feature_importance.csv"
    config.model.metadata_path = project_root / "artifacts" / "model_metadata.json"
    config.log_dir = project_root / "artifacts" / "logs"
    config.model.n_iter = 2
    config.model.cv_folds = 3

    run_training_pipeline(config)
    fraud_api.config = config

    client = TestClient(fraud_api.app)
    response = client.post(
        "/predict",
        json={
            "transaction_id": "tx_api_001",
            "user_id": 802,
            "transaction_amount": 6700.0,
            "transaction_time": "2024-03-04 01:20:00",
            "location": "TX",
            "device": "desktop",
            "merchant_category": "electronics",
            "payment_channel": "wallet",
            "is_international": 1,
            "card_present": 0,
            "previous_transactions_24h": 11,
            "avg_spend_7d": 300.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "fraud" in payload
    assert "probability" in payload
    assert "model_version" in payload


def test_predict_endpoint_invalid_payload_returns_422() -> None:
    client = TestClient(fraud_api.app)
    response = client.post(
        "/predict",
        json={
            "transaction_id": "",
            "user_id": 0,
        },
    )
    assert response.status_code == 422
