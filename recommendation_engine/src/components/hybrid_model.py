"""Hybrid recommendation model combining collaborative, content, and popularity signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger
from src.components.collaborative_filtering import CollaborativeFilteringModel
from src.components.content_based import ContentBasedRecommender
from src.components.preprocessing import RecommendationPreprocessor


logger = get_logger(__name__)


@dataclass
class HybridRecommendationModel:
    """Blend collaborative, content, and popularity signals for ranking."""

    config: AppConfig
    preprocessor: RecommendationPreprocessor
    collaborative_model: CollaborativeFilteringModel | None = None
    content_model: ContentBasedRecommender | None = None
    item_catalog: pd.DataFrame | None = None
    popularity: pd.DataFrame | None = None
    train_interactions: pd.DataFrame | None = None
    recommendation_weights: dict[str, float] = field(default_factory=dict)
    _score_cache: dict[tuple[int, float, float, float], pd.Series] = field(default_factory=dict)

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "HybridRecommendationModel":
        try:
            self.train_interactions = interactions.copy()
            self.item_catalog = items.copy()
            self.collaborative_model = CollaborativeFilteringModel(self.config, self.preprocessor).fit(interactions)
            self.content_model = ContentBasedRecommender(self.config, self.preprocessor).fit(items, interactions)
            self.popularity = self.preprocessor.popularity(interactions)
            self.recommendation_weights = {
                "cf": self.config.model.cf_weight,
                "cb": self.config.model.cb_weight,
                "pop": self.config.model.popularity_weight,
            }
            self._score_cache = {}
            logger.info("Trained hybrid recommendation model")
            return self
        except Exception as exc:
            raise RecommendationException("Hybrid model training failed", exc) from exc

    def _known_user_history(self, user_id: int) -> pd.DataFrame:
        if self.train_interactions is None:
            return pd.DataFrame()
        history = self.train_interactions[self.train_interactions[self.config.data.user_column] == user_id]
        return history.copy()

    def _popular_recommendations(self, top_n: int) -> list[str]:
        if self.popularity is None or self.popularity.empty:
            return []
        return self.popularity.head(top_n).index.astype(str).tolist()

    def _popular_recommendations_for_user(self, user_id: int, top_n: int) -> list[str]:
        if self.popularity is None or self.popularity.empty:
            return []
        seen = set(self._known_user_history(user_id)[self.config.data.item_column].astype(str).tolist())
        ranked = self.popularity.copy()
        ranked = ranked.drop(index=list(seen), errors="ignore")
        return ranked.head(top_n).index.astype(str).tolist()

    def _all_candidates(self) -> pd.Index:
        if self.item_catalog is not None:
            return pd.Index(self.item_catalog[self.config.data.item_column].astype(str).unique())
        if self.popularity is not None:
            return pd.Index(self.popularity.index.astype(str))
        return pd.Index([])

    def set_weights(self, cf_weight: float, cb_weight: float, popularity_weight: float) -> None:
        total = cf_weight + cb_weight + popularity_weight
        if total <= 0:
            raise ValueError("Recommendation weights must have positive total")
        self.recommendation_weights = {
            "cf": cf_weight / total,
            "cb": cb_weight / total,
            "pop": popularity_weight / total,
        }
        self._score_cache.clear()

    def score_items(self, user_id: int) -> pd.Series:
        cache_key = (
            user_id,
            self.recommendation_weights.get("cf", 0.5),
            self.recommendation_weights.get("cb", 0.3),
            self.recommendation_weights.get("pop", 0.2),
        )
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        candidates = self._all_candidates()
        if candidates.empty:
            return pd.Series(dtype=float)

        if self.collaborative_model is None or self.content_model is None:
            return pd.Series(dtype=float)

        cf_scores = self.collaborative_model.score_items(user_id).reindex(candidates).fillna(0.0)
        cb_scores = self.content_model.score_items(user_id).reindex(candidates).fillna(0.0)

        if self.popularity is not None and not self.popularity.empty:
            popularity_scores = self.popularity["avg_signal"].reindex(candidates).fillna(self.popularity["avg_signal"].mean())
        else:
            popularity_scores = pd.Series(3.0, index=candidates)

        combined = (
            self.recommendation_weights.get("cf", 0.5) * cf_scores
            + self.recommendation_weights.get("cb", 0.3) * cb_scores
            + self.recommendation_weights.get("pop", 0.2) * popularity_scores
        )
        ranked = combined.sort_values(ascending=False)
        self._score_cache[cache_key] = ranked
        return ranked

    def predict_score(self, user_id: int, item_id: str) -> float:
        scores = self.score_items(user_id)
        if item_id in scores.index and scores[item_id] > 0:
            return float(np.clip(scores[item_id], 1.0, 5.0))

        if self.popularity is not None and item_id in self.popularity.index:
            return float(np.clip(self.popularity.loc[item_id, "avg_signal"], 1.0, 5.0))

        if self.popularity is not None and not self.popularity.empty:
            return float(np.clip(self.popularity["avg_signal"].mean(), 1.0, 5.0))

        return 3.0

    def recommend(self, user_id: int, top_n: int = 10) -> list[str]:
        history = self._known_user_history(user_id)
        if history.empty:
            return self._popular_recommendations(top_n)

        seen = set(history[self.config.data.item_column].astype(str).tolist())
        ranked = self.score_items(user_id).drop(labels=list(seen), errors="ignore")
        if ranked.empty:
            return self._popular_recommendations_for_user(user_id, top_n)
        return ranked.head(top_n).index.astype(str).tolist()
