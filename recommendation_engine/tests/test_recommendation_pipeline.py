from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.configuration import AppConfig
from src.pipelines.recommendation_pipeline import run_recommendation_pipeline
from src.pipelines.training_pipeline import run_training_pipeline


def test_training_and_recommendation_pipeline(tmp_path: Path) -> None:
    project_root = tmp_path
    repo_root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_env()
    config.project_root = project_root
    config.data.interactions_path = repo_root / "data" / "raw" / "interactions.csv"
    config.data.items_path = repo_root / "data" / "raw" / "items.csv"
    config.data.processed_interactions_path = project_root / "data" / "processed" / "interactions_processed.csv"
    config.data.processed_items_path = project_root / "data" / "processed" / "items_processed.csv"
    config.model.model_path = project_root / "models" / "hybrid_recommender.joblib"
    config.model.metrics_path = project_root / "artifacts" / "metrics.json"
    config.log_dir = project_root / "artifacts" / "logs"
    config.model.top_k_eval = 5

    result = run_training_pipeline(config)

    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert "precision_at_k" in result["metrics"]
    assert "recall_at_k" in result["metrics"]

    recommendations = run_recommendation_pipeline(101, top_n=3, config=config)
    assert len(recommendations) == 3


def test_cold_start_user_recommendations(tmp_path: Path) -> None:
    project_root = tmp_path
    repo_root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_env()
    config.project_root = project_root
    config.data.interactions_path = repo_root / "data" / "raw" / "interactions.csv"
    config.data.items_path = repo_root / "data" / "raw" / "items.csv"
    config.data.processed_interactions_path = project_root / "data" / "processed" / "interactions_processed.csv"
    config.data.processed_items_path = project_root / "data" / "processed" / "items_processed.csv"
    config.model.model_path = project_root / "models" / "hybrid_recommender.joblib"
    config.model.metrics_path = project_root / "artifacts" / "metrics.json"
    config.log_dir = project_root / "artifacts" / "logs"

    run_training_pipeline(config)
    recommendations = run_recommendation_pipeline(999999, top_n=5, config=config)
    assert len(recommendations) == 5


def test_recommendations_exclude_seen_items(tmp_path: Path) -> None:
    project_root = tmp_path
    repo_root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_env()
    config.project_root = project_root
    config.data.interactions_path = repo_root / "data" / "raw" / "interactions.csv"
    config.data.items_path = repo_root / "data" / "raw" / "items.csv"
    config.data.processed_interactions_path = project_root / "data" / "processed" / "interactions_processed.csv"
    config.data.processed_items_path = project_root / "data" / "processed" / "items_processed.csv"
    config.model.model_path = project_root / "models" / "hybrid_recommender.joblib"
    config.model.metrics_path = project_root / "artifacts" / "metrics.json"
    config.log_dir = project_root / "artifacts" / "logs"

    run_training_pipeline(config)
    recs = run_recommendation_pipeline(101, top_n=5, config=config)

    train_df = pd.read_csv(config.data.processed_interactions_path)
    seen_items = set(
        train_df.loc[train_df[config.data.user_column] == 101, config.data.item_column]
        .astype(str)
        .tolist()
    )
    assert not (set(recs) & seen_items)
