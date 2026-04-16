"""Evaluation and threshold tuning for fraud model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.configuration import AppConfig
from src.exception.custom_exception import FraudDetectionException
from src.logger.logging import get_logger


logger = get_logger(__name__)


def tune_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    min_precision: float,
) -> float:
    """Tune threshold to maximize recall under precision constraint when possible."""

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    thresholds = thresholds

    if len(thresholds) == 0:
        return 0.5

    f1_values = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    valid = precision >= min_precision

    if np.any(valid):
        candidate_indices = np.where(valid)[0]
        best_idx = candidate_indices[np.argmax(recall[candidate_indices] + 0.001 * f1_values[candidate_indices])]
        return float(thresholds[best_idx])

    return float(thresholds[np.argmax(f1_values)])


def extract_feature_importance(model_pipeline, max_features: int) -> pd.DataFrame:
    """Extract model feature importance from fitted pipeline."""

    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({"feature": feature_names, "importance": scores})
    importance_df = importance_df.sort_values("importance", ascending=False).head(max_features)
    return importance_df


def build_curve_data(y_true: pd.Series, probabilities: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ROC and precision-recall curve point tables for artifact export."""

    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)

    roc_df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": np.append(roc_thresholds, np.nan)[: len(fpr)],
        }
    )
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    return roc_df, pr_df


@dataclass
class ModelEvaluator:
    """Evaluate fraud model and produce threshold-aware metrics."""

    config: AppConfig

    def evaluate(self, model_pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        try:
            probabilities = model_pipeline.predict_proba(x_test)[:, 1]
            threshold = tune_threshold(
                y_true=y_test,
                probabilities=probabilities,
                min_precision=self.config.model.min_precision_for_threshold,
            )
            predictions = (probabilities >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            metrics = {
                "threshold": float(threshold),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "false_negative_rate": float(fn / max(fn + tp, 1)),
                "false_positive_rate": float(fp / max(fp + tn, 1)),
            }
            logger.info("Evaluation metrics=%s", metrics)
            return metrics
        except Exception as exc:
            raise FraudDetectionException("Model evaluation failed", exc) from exc
