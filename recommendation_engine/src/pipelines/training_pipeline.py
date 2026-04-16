"""Training pipeline for the recommendation engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.components.data_loader import RecommendationDataLoader
from src.components.evaluation import RecommendationEvaluator, tune_hybrid_weights
from src.components.hybrid_model import HybridRecommendationModel
from src.components.preprocessing import RecommendationPreprocessor
from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger, setup_logger
from src.utils.io_utils import save_csv, save_json, save_model


logger = get_logger(__name__)


def run_training_pipeline(config: AppConfig | None = None) -> dict:
    config = config or AppConfig.from_env()
    setup_logger(config.log_dir)

    try:
        loader = RecommendationDataLoader(config)
        interactions, items = loader.load()

        preprocessor = RecommendationPreprocessor(config)
        interactions = preprocessor.prepare_interactions(interactions)
        items = preprocessor.prepare_items(items)

        train_interactions, test_interactions = preprocessor.split_train_test_by_user(interactions)

        save_csv(train_interactions, config.data.processed_interactions_path)
        save_csv(items, config.data.processed_items_path)

        model = HybridRecommendationModel(config, preprocessor).fit(train_interactions, items)
        best_weights = tune_hybrid_weights(
            model,
            test_interactions,
            user_column=config.data.user_column,
            item_column=config.data.item_column,
            k=config.model.top_k_eval,
        )
        model.set_weights(best_weights["cf"], best_weights["cb"], best_weights["pop"])

        evaluator = RecommendationEvaluator(config)
        metrics = evaluator.evaluate(model, test_interactions)

        save_model(model, config.model.model_path)
        save_json(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "weights": model.recommendation_weights,
                "train_users": int(train_interactions[config.data.user_column].nunique()),
                "train_items": int(train_interactions[config.data.item_column].nunique()),
                "sparsity": preprocessor.sparsity(preprocessor.build_user_item_matrix(train_interactions)),
            },
            config.model.metrics_path,
        )

        logger.info("Training pipeline finished successfully")
        return {
            "model_path": str(config.model.model_path),
            "metrics_path": str(config.model.metrics_path),
            "metrics": metrics,
            "weights": model.recommendation_weights,
        }
    except Exception as exc:
        raise RecommendationException("Training pipeline failed", exc) from exc


if __name__ == "__main__":
    run_training_pipeline()
