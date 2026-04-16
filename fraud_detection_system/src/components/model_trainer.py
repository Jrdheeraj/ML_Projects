"""Model training with CV and hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.components.feature_engineering import FraudFeatureEngineer
from src.components.imbalance_handler import ImbalanceStrategy, available_strategies
from src.components.preprocessing import build_preprocessor
from src.config.configuration import AppConfig
from src.exception.custom_exception import FraudDetectionException
from src.logger.logging import get_logger


logger = get_logger(__name__)


def _xgboost_model(random_state: int):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
            n_estimators=200,
        )
    except Exception:
        return None


@dataclass
class TrainingOutput:
    model: object
    model_name: str
    sampler_name: str
    best_cv_score: float
    comparison_table: list[dict[str, float | str]]


@dataclass
class ModelTrainer:
    """Train, compare, and tune models across imbalance strategies."""

    config: AppConfig

    def _candidate_models(self) -> dict[str, tuple[object, dict]]:
        candidates: dict[str, tuple[object, dict]] = {
            "logistic_regression": (
                LogisticRegression(max_iter=2500, random_state=self.config.data.random_state),
                {
                    "model__C": np.logspace(-3, 2, 25),
                    "model__solver": ["liblinear", "lbfgs"],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "random_forest": (
                RandomForestClassifier(random_state=self.config.data.random_state),
                {
                    "model__n_estimators": [200, 400, 600],
                    "model__max_depth": [None, 5, 10, 15],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__class_weight": [None, "balanced", "balanced_subsample"],
                },
            ),
        }

        xgb = _xgboost_model(self.config.data.random_state)
        if xgb is not None:
            candidates["xgboost"] = (
                xgb,
                {
                    "model__n_estimators": [200, 400, 600],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__subsample": [0.7, 0.9, 1.0],
                    "model__colsample_bytree": [0.7, 0.9, 1.0],
                    "model__scale_pos_weight": [1, 2, 5, 10],
                },
            )
        else:
            logger.warning("xgboost unavailable, skipping xgboost candidate")

        return candidates

    def _build_preprocessor_once(self, x_train: pd.DataFrame) -> ColumnTransformer:
        feature_engineer = FraudFeatureEngineer()
        transformed_preview = feature_engineer.fit_transform(x_train)
        return build_preprocessor(transformed_preview)

    def _build_pipeline(
        self,
        preprocessor: ColumnTransformer,
        model: object,
        strategy: ImbalanceStrategy,
    ) -> ImbPipeline:
        feature_engineer = FraudFeatureEngineer()

        steps = [
            ("feature_engineering", feature_engineer),
            ("preprocessor", preprocessor),
        ]
        if strategy.sampler is not None:
            steps.append(("sampler", strategy.sampler))
        steps.append(("model", model))

        return ImbPipeline(steps=steps)

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> TrainingOutput:
        try:
            cv = StratifiedKFold(
                n_splits=self.config.model.cv_folds,
                shuffle=True,
                random_state=self.config.data.random_state,
            )
            candidates = self._candidate_models()
            strategies = available_strategies(self.config.data.random_state)
            preprocessor = self._build_preprocessor_once(x_train)

            best_model = None
            best_name = ""
            best_sampler = ""
            best_score = float("-inf")
            comparison_rows: list[dict[str, float | str]] = []

            for strategy in strategies:
                for model_name, (model, params) in candidates.items():
                    pipeline = self._build_pipeline(preprocessor, model, strategy)
                    search = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=params,
                        n_iter=self.config.model.n_iter,
                        scoring="average_precision",
                        cv=cv,
                        random_state=self.config.data.random_state,
                        n_jobs=-1,
                        verbose=0,
                    )
                    search.fit(x_train, y_train)

                    logger.info(
                        "CV done model=%s sampler=%s score=%.4f",
                        model_name,
                        strategy.name,
                        search.best_score_,
                    )
                    comparison_rows.append(
                        {
                            "model": model_name,
                            "imbalance_strategy": strategy.name,
                            "cv_average_precision": float(search.best_score_),
                        }
                    )

                    if search.best_score_ > best_score:
                        best_model = search.best_estimator_
                        best_name = model_name
                        best_sampler = strategy.name
                        best_score = float(search.best_score_)

            if best_model is None:
                raise RuntimeError("No candidate model trained successfully")

            return TrainingOutput(
                model=best_model,
                model_name=best_name,
                sampler_name=best_sampler,
                best_cv_score=best_score,
                comparison_table=comparison_rows,
            )
        except Exception as exc:
            raise FraudDetectionException("Model training failed", exc) from exc
