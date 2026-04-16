"""Central configuration for the recommendation engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    interactions_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "raw" / "interactions.csv"
    )
    items_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "raw" / "items.csv"
    )
    processed_interactions_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "processed" / "interactions_processed.csv"
    )
    processed_items_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "processed" / "items_processed.csv"
    )
    target_column: str = "signal"
    user_column: str = "user_id"
    item_column: str = "item_id"
    timestamp_column: str = "timestamp"
    rating_column: str = "rating"
    interaction_column: str = "interaction_type"


@dataclass
class ModelConfig:
    model_path: Path = field(
        default_factory=lambda: _project_root() / "models" / "hybrid_recommender.joblib"
    )
    metrics_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "metrics.json"
    )
    top_k_neighbors: int = 20
    top_k_eval: int = 10
    default_top_n: int = 10
    max_top_n: int = 50
    recommendation_cache_size: int = 1024
    min_interactions_per_user_for_eval: int = 2
    cf_weight: float = 0.55
    cb_weight: float = 0.30
    popularity_weight: float = 0.15


@dataclass
class AppConfig:
    project_root: Path = field(default_factory=_project_root)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log_dir: Path = field(default_factory=lambda: _project_root() / "artifacts" / "logs")
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @classmethod
    def from_env(cls) -> "AppConfig":
        config = cls()

        config.data.interactions_path = Path(
            os.getenv("INTERACTIONS_PATH", str(config.data.interactions_path))
        )
        config.data.items_path = Path(os.getenv("ITEMS_PATH", str(config.data.items_path)))
        config.data.processed_interactions_path = Path(
            os.getenv("PROCESSED_INTERACTIONS_PATH", str(config.data.processed_interactions_path))
        )
        config.data.processed_items_path = Path(
            os.getenv("PROCESSED_ITEMS_PATH", str(config.data.processed_items_path))
        )

        config.model.model_path = Path(os.getenv("MODEL_PATH", str(config.model.model_path)))
        config.model.metrics_path = Path(os.getenv("METRICS_PATH", str(config.model.metrics_path)))
        config.model.top_k_neighbors = int(os.getenv("TOP_K_NEIGHBORS", str(config.model.top_k_neighbors)))
        config.model.top_k_eval = int(os.getenv("TOP_K_EVAL", str(config.model.top_k_eval)))
        config.model.default_top_n = int(os.getenv("DEFAULT_TOP_N", str(config.model.default_top_n)))
        config.model.max_top_n = int(os.getenv("MAX_TOP_N", str(config.model.max_top_n)))
        config.model.recommendation_cache_size = int(
            os.getenv("RECOMMENDATION_CACHE_SIZE", str(config.model.recommendation_cache_size))
        )
        config.model.min_interactions_per_user_for_eval = int(
            os.getenv(
                "MIN_INTERACTIONS_PER_USER_FOR_EVAL",
                str(config.model.min_interactions_per_user_for_eval),
            )
        )
        config.model.cf_weight = float(os.getenv("CF_WEIGHT", str(config.model.cf_weight)))
        config.model.cb_weight = float(os.getenv("CB_WEIGHT", str(config.model.cb_weight)))
        config.model.popularity_weight = float(
            os.getenv("POPULARITY_WEIGHT", str(config.model.popularity_weight))
        )

        config.api_host = os.getenv("API_HOST", config.api_host)
        config.api_port = int(os.getenv("API_PORT", str(config.api_port)))

        return config
