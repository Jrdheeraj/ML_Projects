"""Preprocessing utilities for churn model."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    x = x.copy()

    numeric_cols = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    transformers = []

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No features available after initial preparation.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
