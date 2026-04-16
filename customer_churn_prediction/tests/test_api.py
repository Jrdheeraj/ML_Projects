from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import api.main as churn_api
from src.config.configuration import AppConfig
from src.pipelines.training_pipeline import run_training_pipeline


def _synthetic_telco_df(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    tenure = rng.integers(1, 72, size=n_rows)
    monthly = rng.uniform(20, 120, size=n_rows).round(2)
    total = (tenure * monthly + rng.normal(0, 35, size=n_rows)).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.6, 0.25, 0.15])
    support = rng.choice(["Yes", "No"], size=n_rows, p=[0.35, 0.65])
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.4, 0.5, 0.1])
    raw_score = 0.035 * monthly - 0.03 * tenure + (contract == "Month-to-month") * 1.1 + (support == "No") * 0.6 + rng.normal(0, 0.6, size=n_rows)
    churn = np.where(raw_score > 1.3, "Yes", "No")

    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "TechSupport": support,
            "InternetService": internet,
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_rows),
            "Churn": churn,
        }
    )


def test_health_endpoint() -> None:
    client = TestClient(churn_api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(tmp_path: Path) -> None:
    project_root = tmp_path
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    data_path = raw_data_dir / "telco_churn.csv"
    _synthetic_telco_df().to_csv(data_path, index=False)

    config = AppConfig.from_env()
    config.project_root = project_root
    config.data.raw_data_path = data_path
    config.data.train_data_path = project_root / "artifacts" / "train.csv"
    config.data.test_data_path = project_root / "artifacts" / "test.csv"
    config.model.model_output_path = project_root / "models" / "best_model.joblib"
    config.model.metrics_output_path = project_root / "artifacts" / "metrics.json"
    config.model.feature_schema_output_path = project_root / "artifacts" / "feature_schema.json"
    config.log_dir = project_root / "artifacts" / "logs"
    config.model.n_iter = 2
    config.model.cv_folds = 3

    run_training_pipeline(config)
    churn_api.config = config

    client = TestClient(churn_api.app)
    response = client.post(
        "/predict",
        json={
            "tenure": 6,
            "MonthlyCharges": 95.0,
            "TotalCharges": 540.0,
            "Contract": "Month-to-month",
            "TechSupport": "No",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "predictions" in payload
    assert payload["predictions"][0]["churn"] in {"Yes", "No"}
    assert 0.0 <= payload["predictions"][0]["probability"] <= 1.0
