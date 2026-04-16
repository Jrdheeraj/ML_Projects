"""Model training and hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.components.feature_engineering import ChurnFeatureEngineer
from src.components.preprocessing import build_preprocessor
from src.config.configuration import AppConfig
from src.exception.custom_exception import ChurnException
from src.logger.logging import get_logger


logger = get_logger(__name__)


def _get_xgboost_model(random_state: int):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
        )
    except Exception:
        return None


@dataclass
class TrainingResult:
    best_model: Any
    best_model_name: str
    best_score: float


@dataclass
class ModelTrainer:
    config: AppConfig

    def _base_pipeline(self, x_train: pd.DataFrame) -> Pipeline:
        feature_engineer = ChurnFeatureEngineer()
        engineered_x = feature_engineer.fit_transform(x_train)
        preprocessor = build_preprocessor(engineered_x)
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=self.config.data.random_state),
            threshold="median",
        )

        return Pipeline(
            steps=[
                ("feature_engineering", feature_engineer),
                ("preprocessor", preprocessor),
                ("feature_selector", selector),
                ("model", LogisticRegression(max_iter=1000, random_state=self.config.data.random_state)),
            ]
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> TrainingResult:
        try:
            base_pipeline = self._base_pipeline(x_train)

            xgb_model = _get_xgboost_model(self.config.data.random_state)
            model_candidates = {
                "logistic_regression": {
                    "model": LogisticRegression(max_iter=2000, random_state=self.config.data.random_state),
                    "params": {
                        "model__C": np.logspace(-3, 2, 20),
                        "model__class_weight": [None, "balanced"],
                        "model__solver": ["liblinear", "lbfgs"],
                    },
                },
                "random_forest": {
                    "model": RandomForestClassifier(random_state=self.config.data.random_state),
                    "params": {
                        "model__n_estimators": [100, 200, 400],
                        "model__max_depth": [None, 5, 10, 20],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 4],
                        "model__class_weight": [None, "balanced"],
                    },
                },
            }

            if xgb_model is not None:
                model_candidates["xgboost"] = {
                    "model": xgb_model,
                    "params": {
                        "model__n_estimators": [100, 200, 400],
                        "model__max_depth": [3, 4, 6],
                        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "model__subsample": [0.7, 0.9, 1.0],
                        "model__colsample_bytree": [0.7, 0.9, 1.0],
                    },
                }
            else:
                logger.warning("xgboost is unavailable, continuing with sklearn models")

            best_model = None
            best_score = float("-inf")
            best_model_name = ""

            for name, candidate in model_candidates.items():
                logger.info("Tuning candidate model: %s", name)
                pipeline = base_pipeline.set_params(model=candidate["model"])
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=candidate["params"],
                    n_iter=self.config.model.n_iter,
                    cv=self.config.model.cv_folds,
                    scoring=self.config.model.scoring,
                    n_jobs=-1,
                    random_state=self.config.data.random_state,
                    verbose=0,
                )
                search.fit(x_train, y_train)
                logger.info("Best CV %s for %s: %.4f", self.config.model.scoring, name, search.best_score_)

                if search.best_score_ > best_score:
                    best_score = float(search.best_score_)
                    best_model = search.best_estimator_
                    best_model_name = name

            if best_model is None:
                raise RuntimeError("No model was trained successfully")

            return TrainingResult(
                best_model=best_model,
                best_model_name=best_model_name,
                best_score=best_score,
            )
        except Exception as exc:
            raise ChurnException("Model training failed", exc) from exc
