"""Recommendation evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger


logger = get_logger(__name__)


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k if recommended[:k] else 0.0


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def tune_hybrid_weights(
    model,
    validation_df: pd.DataFrame,
    user_column: str,
    item_column: str,
    k: int,
) -> dict[str, float]:
    weight_grid = [
        (0.6, 0.3, 0.1),
        (0.5, 0.35, 0.15),
        (0.55, 0.25, 0.2),
        (0.45, 0.4, 0.15),
    ]
    best = {"cf": 0.55, "cb": 0.30, "pop": 0.15}
    best_precision = -1.0

    for cf_w, cb_w, pop_w in weight_grid:
        model.set_weights(cf_w, cb_w, pop_w)
        scores = []
        for user_id, user_test in validation_df.groupby(user_column):
            relevant = set(user_test[item_column].astype(str).tolist())
            recs = model.recommend(int(user_id), top_n=k)
            scores.append(precision_at_k(recs, relevant, k))
        mean_precision = float(np.mean(scores) if scores else 0.0)
        if mean_precision > best_precision:
            best_precision = mean_precision
            best = {"cf": cf_w, "cb": cb_w, "pop": pop_w}

    return best


@dataclass
class RecommendationEvaluator:
    config: AppConfig

    def evaluate(
        self,
        model,
        test_interactions: pd.DataFrame,
    ) -> dict[str, float]:
        try:
            precision_scores = []
            recall_scores = []
            y_true = []
            y_pred = []

            for user_id, user_test in test_interactions.groupby(self.config.data.user_column):
                relevant_items = set(user_test[self.config.data.item_column].astype(str).tolist())
                recommendations = model.recommend(int(user_id), top_n=self.config.model.top_k_eval)

                precision_scores.append(precision_at_k(recommendations, relevant_items, self.config.model.top_k_eval))
                recall_scores.append(recall_at_k(recommendations, relevant_items, self.config.model.top_k_eval))

                for _, row in user_test.iterrows():
                    y_true.append(float(row[self.config.data.target_column]))
                    y_pred.append(float(model.predict_score(int(user_id), str(row[self.config.data.item_column]))))

            metrics = {
                "precision_at_k": float(np.mean(precision_scores) if precision_scores else 0.0),
                "recall_at_k": float(np.mean(recall_scores) if recall_scores else 0.0),
                "rmse": rmse(np.array(y_true, dtype=float), np.array(y_pred, dtype=float)) if y_true else 0.0,
            }

            logger.info("Evaluation metrics: %s", metrics)
            return metrics
        except Exception as exc:
            raise RecommendationException("Evaluation failed", exc) from exc
