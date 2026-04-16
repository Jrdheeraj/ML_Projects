"""Collaborative filtering recommender using user-item similarity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger
from src.components.preprocessing import RecommendationPreprocessor


logger = get_logger(__name__)


@dataclass
class CollaborativeFilteringModel:
    config: AppConfig
    preprocessor: RecommendationPreprocessor
    user_item_matrix: pd.DataFrame | None = None
    user_similarity: pd.DataFrame | None = None
    item_similarity: pd.DataFrame | None = None
    popularity: pd.DataFrame | None = None
    train_interactions: pd.DataFrame | None = None
    seen_items_by_user: dict[int, set[str]] = field(default_factory=dict)

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFilteringModel":
        try:
            self.train_interactions = interactions.copy()
            self.user_item_matrix = self.preprocessor.build_user_item_matrix(interactions)

            user_sim = cosine_similarity(self.user_item_matrix.values)
            item_sim = cosine_similarity(self.user_item_matrix.T.values)

            self.user_similarity = pd.DataFrame(
                user_sim,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index,
            )
            self.item_similarity = pd.DataFrame(
                item_sim,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns,
            )
            self.popularity = self.preprocessor.popularity(interactions)
            self.seen_items_by_user = (
                interactions.groupby(self.config.data.user_column)[self.config.data.item_column]
                .apply(lambda series: set(series.astype(str)))
                .to_dict()
            )
            logger.info("Trained collaborative filtering model")
            return self
        except Exception as exc:
            raise RecommendationException("Collaborative filtering training failed", exc) from exc

    def _seen_items(self, user_id: int) -> set[str]:
        return self.seen_items_by_user.get(user_id, set())

    def score_items(self, user_id: int) -> pd.Series:
        if self.user_item_matrix is None or self.user_similarity is None or self.item_similarity is None:
            raise ValueError("Collaborative model is not fitted")

        if user_id not in self.user_item_matrix.index:
            return pd.Series(dtype=float)

        user_vector = self.user_item_matrix.loc[user_id]

        item_based_scores = pd.Series(dtype=float)
        consumed_items = user_vector[user_vector > 0]
        if not consumed_items.empty:
            sim_subset = self.item_similarity.loc[:, consumed_items.index]
            numerator = sim_subset.dot(consumed_items.values)
            denominator = sim_subset.abs().sum(axis=1).replace(0, np.nan)
            item_based_scores = (numerator / denominator).fillna(0.0)

        user_based_scores = pd.Series(dtype=float)
        if user_id in self.user_similarity.index:
            neighbors = self.user_similarity.loc[user_id].drop(user_id, errors="ignore")
            neighbors = neighbors[neighbors > 0].sort_values(ascending=False).head(self.config.model.top_k_neighbors)
            if not neighbors.empty:
                neighbor_matrix = self.user_item_matrix.loc[neighbors.index]
                numerator = neighbor_matrix.T.dot(neighbors.values)
                denominator = np.sum(np.abs(neighbors.values))
                user_based_scores = (numerator / denominator) if denominator else pd.Series(dtype=float)

        if item_based_scores.empty and user_based_scores.empty:
            return pd.Series(dtype=float)

        if item_based_scores.empty:
            return user_based_scores
        if user_based_scores.empty:
            return item_based_scores

        return (0.65 * item_based_scores.reindex(self.item_similarity.index).fillna(0.0)) + (
            0.35 * user_based_scores.reindex(self.item_similarity.index).fillna(0.0)
        )

    def predict_score(self, user_id: int, item_id: str) -> float:
        scores = self.score_items(user_id)
        if item_id in scores.index and scores[item_id] > 0:
            return float(np.clip(scores[item_id], 1.0, 5.0))

        if self.popularity is not None and item_id in self.popularity.index:
            return float(np.clip(self.popularity.loc[item_id, "avg_signal"], 1.0, 5.0))

        if self.popularity is not None and not self.popularity.empty:
            return float(np.clip(self.popularity["avg_signal"].mean(), 1.0, 5.0))

        return 3.0

    def recommend(self, user_id: int, top_n: int = 10) -> list[tuple[str, float]]:
        if self.item_similarity is None:
            raise ValueError("Collaborative model is not fitted")

        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return self._popular_recommendations(top_n)

        scores = self.score_items(user_id)
        if scores.empty:
            return self._popular_recommendations_for_user(user_id, top_n)

        seen = self._seen_items(user_id)
        ranking = scores.drop(labels=list(seen), errors="ignore").sort_values(ascending=False)
        if ranking.empty:
            return self._popular_recommendations_for_user(user_id, top_n)

        return [(str(item_id), float(np.clip(score, 1.0, 5.0))) for item_id, score in ranking.head(top_n).items()]

    def _popular_recommendations(self, top_n: int) -> list[tuple[str, float]]:
        if self.popularity is None or self.popularity.empty:
            return []
        top_items = self.popularity.head(top_n)
        return [(str(item_id), float(score["avg_signal"])) for item_id, score in top_items.iterrows()]

    def _popular_recommendations_for_user(self, user_id: int, top_n: int) -> list[tuple[str, float]]:
        if self.popularity is None or self.popularity.empty:
            return []
        seen = self._seen_items(user_id)
        ranked = self.popularity.drop(index=list(seen), errors="ignore").head(top_n)
        return [(str(item_id), float(score["avg_signal"])) for item_id, score in ranked.iterrows()]
