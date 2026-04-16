"""Preprocessing builder for mixed-type fraud features."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing transformer for numeric/categorical columns."""

    numeric_cols = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable columns found for preprocessing")

    return ColumnTransformer(transformers=transformers, remainder="drop")
