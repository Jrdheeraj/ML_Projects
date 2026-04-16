from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.configuration import AppConfig
from src.pipelines.prediction_pipeline import run_prediction_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.io_utils import save_json


def _synthetic_telco_df(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 72, size=n_rows)
    monthly = rng.uniform(20, 120, size=n_rows).round(2)
    total = (tenure * monthly + rng.normal(0, 50, size=n_rows)).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.6, 0.25, 0.15])
    tech = rng.choice(["Yes", "No"], size=n_rows, p=[0.4, 0.6])
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.4, 0.5, 0.1])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_rows)

    raw_score = (
        0.03 * monthly
        - 0.02 * tenure
        + (contract == "Month-to-month") * 1.2
        + (internet == "Fiber optic") * 0.6
        + (tech == "No") * 0.5
        + rng.normal(0, 0.7, size=n_rows)
    )
    churn = np.where(raw_score > 1.4, "Yes", "No")

    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "TechSupport": tech,
            "InternetService": internet,
            "PaymentMethod": payment,
            "Churn": churn,
        }
    )


def test_end_to_end_training_and_prediction(tmp_path: Path) -> None:
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
    config.log_dir = project_root / "artifacts" / "logs"
    config.model.n_iter = 3
    config.model.cv_folds = 3

    result = run_training_pipeline(config)

    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert "f1_score" in result["metrics"]
    assert "roc_auc" in result["metrics"]

    sample = {
        "tenure": 6,
        "MonthlyCharges": 95.0,
        "TotalCharges": 540.0,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    pred = run_prediction_pipeline(sample, config)
    assert pred["churn"] in {"Yes", "No"}
    assert 0.0 <= pred["probability"] <= 1.0


def test_prediction_uses_model_schema_when_feature_schema_is_stale(tmp_path: Path) -> None:
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
    save_json({"feature_columns": ["tenure", "MonthlyCharges"]}, config.model.feature_schema_output_path)

    sample = {
        "tenure": 6,
        "MonthlyCharges": 95.0,
        "TotalCharges": 540.0,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    pred = run_prediction_pipeline(sample, config)
    assert pred["churn"] in {"Yes", "No"}
    assert 0.0 <= pred["probability"] <= 1.0
