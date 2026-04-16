"""Configuration objects for churn training and serving."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    raw_data_path: Path = field(
        default_factory=lambda: _project_root() / "data" / "raw" / "telco_churn.csv"
    )
    train_data_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "train.csv"
    )
    test_data_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "test.csv"
    )
    target_column: str = "Churn"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    model_output_path: Path = field(
        default_factory=lambda: _project_root() / "models" / "best_model.joblib"
    )
    metrics_output_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "metrics.json"
    )
    feature_schema_output_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "feature_schema.json"
    )
    cv_folds: int = 5
    scoring: str = "f1"
    n_iter: int = 20


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

        cfg.data.raw_data_path = Path(
            os.getenv("RAW_DATA_PATH", str(cfg.data.raw_data_path))
        )
        cfg.data.target_column = os.getenv("TARGET_COLUMN", cfg.data.target_column)
        cfg.data.test_size = float(os.getenv("TEST_SIZE", str(cfg.data.test_size)))
        cfg.data.random_state = int(
            os.getenv("RANDOM_STATE", str(cfg.data.random_state))
        )

        cfg.model.model_output_path = Path(
            os.getenv("MODEL_OUTPUT_PATH", str(cfg.model.model_output_path))
        )
        cfg.model.metrics_output_path = Path(
            os.getenv("METRICS_OUTPUT_PATH", str(cfg.model.metrics_output_path))
        )
        cfg.model.feature_schema_output_path = Path(
            os.getenv(
                "FEATURE_SCHEMA_OUTPUT_PATH",
                str(cfg.model.feature_schema_output_path),
            )
        )
        cfg.model.cv_folds = int(os.getenv("CV_FOLDS", str(cfg.model.cv_folds)))
        cfg.model.n_iter = int(os.getenv("N_ITER", str(cfg.model.n_iter)))

        cfg.api_host = os.getenv("API_HOST", cfg.api_host)
        cfg.api_port = int(os.getenv("API_PORT", str(cfg.api_port)))

        return cfg
