"""Content-based recommender using item metadata and user profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger
from src.components.preprocessing import RecommendationPreprocessor


logger = get_logger(__name__)


@dataclass
class ContentBasedRecommender:
    config: AppConfig
    preprocessor: RecommendationPreprocessor
    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    )
    item_features: pd.DataFrame | None = None
    item_tfidf_matrix: any = None
    item_similarity: pd.DataFrame | None = None
    user_profiles: dict[int, np.ndarray] = field(default_factory=dict)
    popularity: pd.DataFrame | None = None
    train_interactions: pd.DataFrame | None = None

    def fit(self, items: pd.DataFrame, interactions: pd.DataFrame) -> "ContentBasedRecommender":
        try:
            self.item_features = items.copy()
            self.train_interactions = interactions.copy()
            corpus = self.item_features["content_text"].fillna("").astype(str).tolist()
            self.item_tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarity = linear_kernel(self.item_tfidf_matrix, self.item_tfidf_matrix)
            self.item_similarity = pd.DataFrame(
                similarity,
                index=self.item_features[self.config.data.item_column].astype(str),
                columns=self.item_features[self.config.data.item_column].astype(str),
            )
            self.popularity = self.preprocessor.popularity(interactions)
            self._build_user_profiles()
            logger.info("Trained content-based recommender")
            return self
        except Exception as exc:
            raise RecommendationException("Content-based training failed", exc) from exc

    def _build_user_profiles(self) -> None:
        self.user_profiles = {}
        if self.train_interactions is None or self.item_features is None:
            return

        item_index = pd.Series(
            data=np.arange(len(self.item_features)),
            index=self.item_features[self.config.data.item_column].astype(str),
        )

        for user_id, group in self.train_interactions.groupby(self.config.data.user_column):
            item_ids = group[self.config.data.item_column].astype(str).tolist()
            signals = group[self.config.data.target_column].astype(float).tolist()

            indices = [item_index[item_id] for item_id in item_ids if item_id in item_index.index]
            if not indices:
                continue

            vectors = self.item_tfidf_matrix[indices]
            weights = np.array([signal for item_id, signal in zip(item_ids, signals) if item_id in item_index.index], dtype=float)
            if np.sum(weights) <= 0:
                weights = np.ones(len(indices), dtype=float)

            profile = np.asarray(vectors.multiply(weights[:, None]).sum(axis=0)).ravel() / np.sum(weights)
            self.user_profiles[int(user_id)] = profile

    def score_items(self, user_id: int) -> pd.Series:
        if self.item_features is None or self.item_tfidf_matrix is None:
            raise ValueError("Content-based model is not fitted")

        if user_id not in self.user_profiles:
            return pd.Series(dtype=float)

        profile = self.user_profiles[user_id].reshape(1, -1)
        scores = linear_kernel(profile, self.item_tfidf_matrix).ravel()
        item_ids = self.item_features[self.config.data.item_column].astype(str).tolist()
        return pd.Series(1.0 + 4.0 * scores, index=item_ids)

    def predict_score(self, user_id: int, item_id: str) -> float:
        scores = self.score_items(user_id)
        if item_id in scores.index and scores[item_id] > 0:
            return float(np.clip(scores[item_id], 1.0, 5.0))

        if self.popularity is not None and item_id in self.popularity.index:
            return float(np.clip(self.popularity.loc[item_id, "avg_signal"], 1.0, 5.0))

        if self.popularity is not None and not self.popularity.empty:
            return float(np.clip(self.popularity["avg_signal"].mean(), 1.0, 5.0))

        return 3.0
