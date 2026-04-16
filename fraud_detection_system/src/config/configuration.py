"""Configuration objects for fraud training/inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    raw_data_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "raw" / "transactions.csv"
    )
    train_data_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "processed" / "train.csv"
    )
    test_data_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "processed" / "test.csv"
    )
    target_column: str = "label"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    model_path: Path = field(
        default_factory=lambda: _project_root() / "models" / "fraud_model.joblib"
    )
    metrics_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "metrics.json"
    )
    feature_importance_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "feature_importance.csv"
    )
    metadata_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "model_metadata.json"
    )
    cv_folds: int = 4
    n_iter: int = 10
    min_precision_for_threshold: float = 0.70
    max_top_features: int = 30


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
        cfg = cls()

        cfg.data.raw_data_path = Path(os.getenv("RAW_DATA_PATH", str(cfg.data.raw_data_path)))
        cfg.data.target_column = os.getenv("TARGET_COLUMN", cfg.data.target_column)
        cfg.data.test_size = float(os.getenv("TEST_SIZE", str(cfg.data.test_size)))
        cfg.data.random_state = int(os.getenv("RANDOM_STATE", str(cfg.data.random_state)))

        cfg.model.model_path = Path(os.getenv("MODEL_PATH", str(cfg.model.model_path)))
        cfg.model.metrics_path = Path(os.getenv("METRICS_PATH", str(cfg.model.metrics_path)))
        cfg.model.feature_importance_path = Path(
            os.getenv("FEATURE_IMPORTANCE_PATH", str(cfg.model.feature_importance_path))
        )
        cfg.model.metadata_path = Path(os.getenv("METADATA_PATH", str(cfg.model.metadata_path)))
        cfg.model.cv_folds = int(os.getenv("CV_FOLDS", str(cfg.model.cv_folds)))
        cfg.model.n_iter = int(os.getenv("N_ITER", str(cfg.model.n_iter)))
        cfg.model.min_precision_for_threshold = float(
            os.getenv("MIN_PRECISION_FOR_THRESHOLD", str(cfg.model.min_precision_for_threshold))
        )

        cfg.api_host = os.getenv("API_HOST", cfg.api_host)
        cfg.api_port = int(os.getenv("API_PORT", str(cfg.api_port)))
        return cfg
