"""Preprocessing and feature preparation for recommendation models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger


logger = get_logger(__name__)


INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 2.5,
    "rating": 3.5,
    "purchase": 5.0,
    "like": 4.0,
}


@dataclass
class RecommendationPreprocessor:
    """Preprocess interactions and item metadata for training and inference."""

    config: AppConfig

    def prepare_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        try:
            df = interactions.copy()
            df[self.config.data.user_column] = df[self.config.data.user_column].astype(int)
            df[self.config.data.item_column] = df[self.config.data.item_column].astype(str)

            if self.config.data.timestamp_column in df.columns:
                df[self.config.data.timestamp_column] = pd.to_datetime(df[self.config.data.timestamp_column], errors="coerce")
                df = df.sort_values([self.config.data.user_column, self.config.data.timestamp_column])

            if self.config.data.rating_column in df.columns:
                rating = pd.to_numeric(df[self.config.data.rating_column], errors="coerce").clip(1.0, 5.0)
            else:
                rating = pd.Series([np.nan] * len(df), index=df.index, dtype=float)

            interaction_type = df.get(
                self.config.data.interaction_column,
                pd.Series(["view"] * len(df), index=df.index),
            ).astype(str).str.lower()
            interaction_weight = interaction_type.map(INTERACTION_WEIGHTS).fillna(1.0)

            if rating.notna().any():
                df[self.config.data.target_column] = (
                    0.7 * rating.fillna(3.0) + 0.3 * interaction_weight
                ).clip(1.0, 5.0)
            else:
                df[self.config.data.target_column] = interaction_weight.clip(1.0, 5.0)

            df[self.config.data.interaction_column] = interaction_type

            df = df.drop_duplicates(
                subset=[self.config.data.user_column, self.config.data.item_column, self.config.data.timestamp_column]
                if self.config.data.timestamp_column in df.columns
                else [self.config.data.user_column, self.config.data.item_column]
            )

            logger.info("Prepared interactions shape=%s", df.shape)
            return df
        except Exception as exc:
            raise RecommendationException("Failed to prepare interactions", exc) from exc

    def prepare_items(self, items: pd.DataFrame) -> pd.DataFrame:
        try:
            df = items.copy()
            df[self.config.data.item_column] = df[self.config.data.item_column].astype(str)
            df = df.drop_duplicates(subset=[self.config.data.item_column], keep="first")
            text_columns = [column for column in ["title", "genres", "description", "tags", "category"] if column in df.columns]
            if not text_columns:
                raise ValueError("At least one descriptive item column is required for content-based ranking")

            df["content_text"] = (
                df[text_columns]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            logger.info("Prepared items shape=%s", df.shape)
            return df
        except Exception as exc:
            raise RecommendationException("Failed to prepare items", exc) from exc

    def build_user_item_matrix(self, interactions: pd.DataFrame, value_column: str | None = None) -> pd.DataFrame:
        value_column = value_column or self.config.data.target_column
        matrix = interactions.pivot_table(
            index=self.config.data.user_column,
            columns=self.config.data.item_column,
            values=value_column,
            aggfunc="mean",
            fill_value=0.0,
        )
        return matrix

    @staticmethod
    def sparsity(matrix: pd.DataFrame) -> float:
        total = matrix.shape[0] * matrix.shape[1]
        observed = np.count_nonzero(matrix.values)
        if total == 0:
            return 1.0
        return 1.0 - (observed / total)

    def popularity(self, interactions: pd.DataFrame) -> pd.DataFrame:
        popular = (
            interactions.groupby(self.config.data.item_column)[self.config.data.target_column]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_signal", "count": "interaction_count"})
            .sort_values(["avg_signal", "interaction_count"], ascending=False)
        )
        return popular

    def split_train_test_by_user(self, interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.config.data.timestamp_column in interactions.columns:
            ordered = interactions.sort_values([self.config.data.user_column, self.config.data.timestamp_column])
        else:
            ordered = interactions.sort_values([self.config.data.user_column, self.config.data.item_column])

        user_counts = ordered.groupby(self.config.data.user_column).size()
        eligible_users = user_counts[user_counts >= self.config.model.min_interactions_per_user_for_eval].index
        eligible_rows = ordered[ordered[self.config.data.user_column].isin(eligible_users)]

        test_idx = eligible_rows.groupby(self.config.data.user_column).tail(1).index
        test_df = ordered.loc[test_idx].copy()
        train_df = ordered.drop(index=test_idx).copy()

        if train_df.empty:
            raise ValueError("Train split is empty; each user needs at least two interactions")

        return train_df, test_df
