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
            fallback_columns = list(getattr(self._load_model(), "feature_names_in_", []))
            self.feature_columns = load_feature_columns(
                self.config.model.feature_schema_output_path,
                fallback_columns=fallback_columns,
            )
        return self.feature_columns

    def predict(self, records: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            model = self._load_model()
            feature_columns = self._load_feature_columns()
            x = align_input_schema(records, feature_columns)
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

            if isinstance(records, dict):
                return outputs[0]

            return outputs
        except Exception as exc:
            raise ChurnException("Prediction pipeline failed", exc) from exc


def run_prediction_pipeline(records: dict[str, Any] | list[dict[str, Any]], config: AppConfig | None = None):
    config = config or AppConfig.from_env()
    setup_logger(config.log_dir)
    return PredictionPipeline(config).predict(records)
