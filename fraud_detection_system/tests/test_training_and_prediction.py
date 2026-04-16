from __future__ import annotations

from pathlib import Path

import pytest

from src.config.configuration import AppConfig
from src.pipelines.prediction_pipeline import run_prediction_pipeline
from src.pipelines.training_pipeline import run_training_pipeline


def test_end_to_end_training_and_prediction(tmp_path: Path) -> None:
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
    config.model.n_iter = 3
    config.model.cv_folds = 3

    result = run_training_pipeline(config)
    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["metadata_path"]).exists()
    assert Path(result["strategy_comparison_path"]).exists()
    assert Path(result["roc_curve_path"]).exists()
    assert Path(result["precision_recall_curve_path"]).exists()
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]
    assert "false_negative_rate" in result["metrics"]
    assert "false_positive_rate" in result["metrics"]

    sample = {
        "transaction_id": "tx_test_001",
        "user_id": 901,
        "transaction_amount": 8200.0,
        "transaction_time": "2024-03-01 02:15:00",
        "location": "NY",
        "device": "mobile",
        "merchant_category": "electronics",
        "payment_channel": "wallet",
        "is_international": 1,
        "card_present": 0,
        "previous_transactions_24h": 12,
        "avg_spend_7d": 450.0,
    }
    pred = run_prediction_pipeline(sample, config)
    assert isinstance(pred["fraud"], bool)
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["model_version"] == "1.0"


def test_prediction_missing_required_feature_raises_value_error(tmp_path: Path) -> None:
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

    invalid_sample = {
        "transaction_id": "tx_missing_001",
        "user_id": 901,
        "transaction_amount": 8200.0,
        "transaction_time": "2024-03-01 02:15:00",
        "location": "NY",
        "device": "mobile",
        "merchant_category": "electronics",
        "payment_channel": "wallet",
        "is_international": 1,
        "card_present": 0,
        "avg_spend_7d": 450.0,
    }

    with pytest.raises(ValueError, match="Missing required features"):
        run_prediction_pipeline(invalid_sample, config)
