"""Inference pipeline for serving recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger, setup_logger
from src.utils.io_utils import load_model


logger = get_logger(__name__)


@dataclass
class RecommendationPipeline:
    """Load a trained recommender and serve top-N results with caching."""

    config: AppConfig
    model: object | None = None
    cache: dict[tuple[int, int], tuple[str, ...]] | None = None

    def _load_model(self):
        if self.model is None:
            if not self.config.model.model_path.exists():
                raise FileNotFoundError(
                    f"Recommendation model not found at {self.config.model.model_path}. Run training first."
                )
            self.model = load_model(self.config.model.model_path)
            self.cache = {}
        return self.model

    def _recommend_cached(self, user_id: int, top_n: int) -> tuple[str, ...]:
        if self.cache is None:
            self.cache = {}

        key = (user_id, top_n)
        if key in self.cache:
            return self.cache[key]

        model = self._load_model()
        recommendations = tuple(model.recommend(user_id, top_n=top_n))

        if len(self.cache) >= self.config.model.recommendation_cache_size:
            # Drop one entry in FIFO-like manner to bound memory.
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = recommendations
        return recommendations

    def recommend(self, user_id: int, top_n: int = 10) -> list[str]:
        try:
            if user_id <= 0:
                raise ValueError("user_id must be a positive integer")

            top_n = max(1, min(top_n, self.config.model.max_top_n))
            recommendations = list(self._recommend_cached(user_id, top_n))
            logger.info("Generated %d recommendations for user=%s", len(recommendations), user_id)
            return recommendations
        except Exception as exc:
            raise RecommendationException("Recommendation pipeline failed", exc) from exc


def run_recommendation_pipeline(user_id: int, top_n: int = 10, config: AppConfig | None = None) -> list[str]:
    config = config or AppConfig.from_env()
    setup_logger(config.log_dir)
    return RecommendationPipeline(config).recommend(user_id, top_n=top_n)
