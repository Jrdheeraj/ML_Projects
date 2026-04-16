"""Prediction pipeline for real-time fraud scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.configuration import AppConfig
from src.exception.custom_exception import FraudDetectionException
from src.logger.logging import get_logger, setup_logger
from src.utils.io_utils import load_json, load_model


logger = get_logger(__name__)


@dataclass
class FraudPredictionPipeline:
    """Load trained artifacts and return fraud prediction payload."""

    config: AppConfig
    model: Any = None
    threshold: float | None = None
    expected_features: list[str] | None = None

    def _load_model(self):
        if self.model is None:
            if not self.config.model.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.config.model.model_path}. Run training first."
                )
            self.model = load_model(self.config.model.model_path)
        return self.model

    def _load_threshold(self) -> float:
        if self.threshold is None:
            if self.config.model.metadata_path.exists():
                metadata = load_json(self.config.model.metadata_path)
                self.threshold = float(metadata.get("threshold", 0.5))
            else:
                self.threshold = 0.5
        return self.threshold

    def _load_expected_features(self) -> list[str] | None:
        if self.expected_features is None and self.config.model.metadata_path.exists():
            metadata = load_json(self.config.model.metadata_path)
            expected = metadata.get("expected_features")
            if isinstance(expected, list) and expected:
                self.expected_features = [str(feature) for feature in expected]
        return self.expected_features

    def _validate_input_schema(self, record: dict[str, Any]) -> None:
        expected_features = self._load_expected_features()
        if not expected_features:
            return

        incoming_features = set(record.keys())
        expected_set = set(expected_features)
        missing = sorted(expected_set - incoming_features)
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")

        extra = sorted(incoming_features - expected_set)
        if extra:
            logger.warning("Dropping extra inference features: %s", extra)

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        try:
            model = self._load_model()
            threshold = self._load_threshold()
            self._validate_input_schema(record)

            expected_features = self._load_expected_features()
            if expected_features:
                filtered = {feature: record.get(feature) for feature in expected_features}
            else:
                filtered = record

            x = pd.DataFrame([filtered])
            probability = float(model.predict_proba(x)[:, 1][0])
            fraud = bool(probability >= threshold)

            result = {
                "fraud": fraud,
                "probability": probability,
                "threshold": threshold,
                "model_version": "1.0",
            }
            logger.info("Prediction generated fraud=%s probability=%.4f", fraud, probability)
            return result
        except ValueError:
            raise
        except Exception as exc:
            raise FraudDetectionException("Prediction pipeline failed", exc) from exc


def run_prediction_pipeline(record: dict[str, Any], config: AppConfig | None = None) -> dict[str, Any]:
    config = config or AppConfig.from_env()
    setup_logger(config.log_dir)
    return FraudPredictionPipeline(config).predict(record)
