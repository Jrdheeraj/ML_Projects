"""End-to-end training pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.model_evaluation import evaluate_classification, model_summary
from src.components.model_trainer import ModelTrainer
from src.config.configuration import AppConfig
from src.exception.custom_exception import ChurnException
from src.logger.logging import get_logger, setup_logger
from src.utils.io_utils import load_dataframe, save_json, save_model
from src.utils.model_utils import churn_probability_from_model


logger = get_logger(__name__)


def _load_train_test_frames(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    ingestion = DataIngestion(config)
    train_path, test_path = ingestion.run()
    return load_dataframe(Path(train_path)), load_dataframe(Path(test_path))


def _normalize_target(y: pd.Series) -> pd.Series:
    if y.dtype.kind in {"O", "U", "S"}:
        y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    return y.astype(int)


def run_training_pipeline(config: AppConfig | None = None) -> dict:
    config = config or AppConfig.from_env()
    setup_logger(config.log_dir)

    try:
        train_df, test_df = _load_train_test_frames(config)

        target_col = config.data.target_column
        if target_col not in train_df.columns or target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' missing from train or test data")

        x_train = train_df.drop(columns=[target_col])
        y_train = _normalize_target(train_df[target_col])
        x_test = test_df.drop(columns=[target_col])
        y_test = _normalize_target(test_df[target_col])

        trainer = ModelTrainer(config)
        result = trainer.train(x_train, y_train)

        y_pred = result.best_model.predict(x_test)
        y_prob = churn_probability_from_model(result.best_model, x_test)

        metrics = evaluate_classification(y_test, y_pred, y_prob)
        summary = model_summary(result.best_model_name, metrics)

        save_model(result.best_model, config.model.model_output_path)
        save_json(summary, config.model.metrics_output_path)
        save_json({"feature_columns": x_train.columns.tolist()}, config.model.feature_schema_output_path)

        logger.info("Training pipeline completed with best model '%s'", result.best_model_name)
        return {
            "best_model": result.best_model_name,
            "cv_best_score": result.best_score,
            "metrics": metrics,
            "model_path": str(config.model.model_output_path),
            "metrics_path": str(config.model.metrics_output_path),
            "feature_schema_path": str(config.model.feature_schema_output_path),
        }
    except Exception as exc:
        raise ChurnException("Training pipeline failed", exc) from exc


if __name__ == "__main__":
    run_training_pipeline()
