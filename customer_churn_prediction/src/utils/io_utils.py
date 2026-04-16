"""I/O helpers for saving and loading artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_model(model: Any, path: Path) -> None:
    ensure_parent(path)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)
