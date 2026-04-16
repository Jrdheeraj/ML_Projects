"""Utilities for model-level operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import load_json


def churn_probability_from_model(model, x):
    """Return positive class probability when available, else scaled score."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score < 1e-12:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    predictions = model.predict(x)
    return np.array(predictions, dtype=float)


def load_feature_columns(schema_path: Path, fallback_columns: list[str] | None = None) -> list[str]:
    if schema_path.exists():
        payload = load_json(schema_path)
        feature_columns = payload.get("feature_columns", [])
        if isinstance(feature_columns, list) and feature_columns:
            return [str(column) for column in feature_columns]

    return fallback_columns or []


def align_input_schema(
    records: dict[str, Any] | list[dict[str, Any]],
    feature_columns: list[str],
) -> pd.DataFrame:
    if isinstance(records, dict):
        dataframe = pd.DataFrame([records])
        single_record = True
    else:
        if not records:
            raise ValueError("At least one record is required for prediction.")
        dataframe = pd.DataFrame(records)
        single_record = False

    if not feature_columns:
        return dataframe.iloc[[0]] if single_record else dataframe

    aligned = dataframe.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = None

    aligned = aligned[feature_columns]
    return aligned.iloc[[0]] if single_record else aligned
