"""Prediction pipeline for real-time churn inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.configuration import AppConfig
from src.exception.custom_exception import ChurnException
from src.logger.logging import get_logger, setup_logger
from src.utils.io_utils import load_model
from src.utils.model_utils import align_input_schema, churn_probability_from_model, load_feature_columns


logger = get_logger(__name__)


@dataclass
class PredictionPipeline:
    config: AppConfig
    model: Any = None
    feature_columns: list[str] | None = None

    @staticmethod
    def _default_optional_fields(record: dict[str, Any]) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "PaperlessBilling": "No",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
        }
        normalized = dict(record)
        for key, value in defaults.items():
            normalized.setdefault(key, value)
        return normalized

    def _load_model(self):
        if self.model is None:
            if not self.config.model.model_output_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.config.model.model_output_path}. Run training first."
                )
            self.model = load_model(self.config.model.model_output_path)
        return self.model

    def _load_feature_columns(self) -> list[str]:
        if self.feature_columns is None:
            model_columns = list(getattr(self._load_model(), "feature_names_in_", []))
            configured_columns = load_feature_columns(
                self.config.model.feature_schema_output_path,
                fallback_columns=model_columns,
            )
            if model_columns and set(configured_columns) != set(model_columns):
                logger.warning(
                    "Feature schema mismatch detected; using model feature_names_in_ for inference alignment"
                )
                self.feature_columns = model_columns
            else:
                self.feature_columns = configured_columns
        return self.feature_columns

    def predict(self, records: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            model = self._load_model()
            feature_columns = self._load_feature_columns()

            if isinstance(records, dict):
                normalized_records: dict[str, Any] | list[dict[str, Any]] = self._default_optional_fields(records)
            else:
                normalized_records = [self._default_optional_fields(record) for record in records]

            x = align_input_schema(normalized_records, feature_columns)
            try:
                probs = churn_probability_from_model(model, x)
            except ValueError as value_error:
                if "columns are missing" not in str(value_error).lower():
                    raise
                logger.warning(
                    "Aligned schema caused missing columns at inference; retrying with raw payload columns"
                )
                x = pd.DataFrame([normalized_records]) if isinstance(normalized_records, dict) else pd.DataFrame(normalized_records)
                probs = churn_probability_from_model(model, x)
            labels = (probs >= 0.5).astype(int)

            outputs = []
            for label, prob in zip(labels, probs):
                outputs.append(
                    {
                        "churn": "Yes" if int(label) == 1 else "No",
                        "probability": float(prob),
                    }
                )

            if isinstance(normalized_records, dict):
                return outputs[0]

            return outputs
        except Exception as exc:
            raise ChurnException("Prediction pipeline failed", exc) from exc


def run_prediction_pipeline(records: dict[str, Any] | list[dict[str, Any]], config: AppConfig | None = None):
    config = config or AppConfig.from_env()
    setup_logger(config.log_dir)
    return PredictionPipeline(config).predict(records)
