"""Training pipeline for fraud detection system."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.model_evaluation import (
    ModelEvaluator,
    build_curve_data,
    extract_feature_importance,
)
from src.components.model_trainer import ModelTrainer
from src.config.configuration import AppConfig
from src.exception.custom_exception import FraudDetectionException
from src.logger.logging import get_logger, setup_logger
from src.utils.io_utils import save_dataframe, save_json, save_model


logger = get_logger(__name__)


def run_training_pipeline(config: AppConfig | None = None) -> dict:
    """Run full fraud model training/evaluation and persist artifacts."""

    config = config or AppConfig.from_env()
    setup_logger(config.log_dir)

    try:
        ingestion = DataIngestion(config)
        train_df, test_df = ingestion.run()

        target_col = config.data.target_column
        x_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col].astype(int)
        x_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].astype(int)

        trainer = ModelTrainer(config)
        trained = trainer.train(x_train, y_train)
        comparison_df = pd.DataFrame(trained.comparison_table).sort_values(
            "cv_average_precision", ascending=False
        )

        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate(trained.model, x_test, y_test)
        probabilities = trained.model.predict_proba(x_test)[:, 1]
        roc_df, pr_df = build_curve_data(y_test, probabilities)

        importance_df = extract_feature_importance(
            model_pipeline=trained.model,
            max_features=config.model.max_top_features,
        )

        artifact_dir = Path(config.model.metrics_path).parent
        strategy_path = artifact_dir / "strategy_comparison.csv"
        confusion_path = artifact_dir / "confusion_analysis.json"
        roc_curve_path = artifact_dir / "roc_curve.csv"
        pr_curve_path = artifact_dir / "precision_recall_curve.csv"
        roc_plot_path = artifact_dir / "roc_curve.png"
        pr_plot_path = artifact_dir / "precision_recall_curve.png"

        save_model(trained.model, config.model.model_path)
        save_dataframe(importance_df, config.model.feature_importance_path)
        save_dataframe(comparison_df, strategy_path)
        save_dataframe(roc_df, roc_curve_path)
        save_dataframe(pr_df, pr_curve_path)

        save_json(
            {
                "true_negative": metrics["true_negative"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_positive": metrics["true_positive"],
                "false_negative_rate": metrics["false_negative_rate"],
                "false_positive_rate": metrics["false_positive_rate"],
            },
            confusion_path,
        )

        plt.figure(figsize=(7, 5))
        plt.plot(roc_df["fpr"], roc_df["tpr"], label=f"ROC AUC = {metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(roc_plot_path)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(pr_df["recall"], pr_df["precision"], color="darkorange")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        pr_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pr_plot_path)
        plt.close()

        save_json(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "best_model": trained.model_name,
                "sampler_strategy": trained.sampler_name,
                "best_cv_average_precision": trained.best_cv_score,
                "threshold": metrics["threshold"],
                "expected_features": x_train.columns.tolist(),
            },
            config.model.metadata_path,
        )
        save_json(metrics, config.model.metrics_path)

        logger.info("Training pipeline completed with model=%s sampler=%s", trained.model_name, trained.sampler_name)
        return {
            "model_path": str(config.model.model_path),
            "metrics_path": str(config.model.metrics_path),
            "feature_importance_path": str(config.model.feature_importance_path),
            "metadata_path": str(config.model.metadata_path),
            "strategy_comparison_path": str(strategy_path),
            "roc_curve_path": str(roc_curve_path),
            "precision_recall_curve_path": str(pr_curve_path),
            "metrics": metrics,
        }
    except Exception as exc:
        raise FraudDetectionException("Training pipeline failed", exc) from exc


if __name__ == "__main__":
    run_training_pipeline()
