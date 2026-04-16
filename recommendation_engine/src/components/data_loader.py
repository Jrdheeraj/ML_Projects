"""Data loading and validation for recommendation datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.configuration import AppConfig
from src.exception.custom_exception import RecommendationException
from src.logger.logging import get_logger
from src.utils.io_utils import load_csv


logger = get_logger(__name__)


@dataclass
class RecommendationDataLoader:
    """Load and validate interaction and item datasets."""

    config: AppConfig

    def _validate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {
            self.config.data.user_column,
            self.config.data.item_column,
            self.config.data.timestamp_column,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required interaction columns: {sorted(missing)}")

        signal_sources = {
            self.config.data.rating_column,
            self.config.data.interaction_column,
        }
        if signal_sources.isdisjoint(df.columns):
            raise ValueError(
                "Interactions must include at least one signal source: "
                f"{self.config.data.rating_column} or {self.config.data.interaction_column}"
            )

        if df.empty:
            raise ValueError("Interactions dataset is empty")

        if df[self.config.data.user_column].isna().any() or df[self.config.data.item_column].isna().any():
            raise ValueError("Interactions contain null user_id or item_id")

        return df.copy()

    def _validate_items(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {self.config.data.item_column, "title"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required item columns: {sorted(missing)}")

        if df.empty:
            raise ValueError("Items dataset is empty")

        if df[self.config.data.item_column].isna().any():
            raise ValueError("Items contain null item_id")

        return df.copy()

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            if not self.config.data.interactions_path.exists():
                raise FileNotFoundError(f"Interactions file not found: {self.config.data.interactions_path}")
            if not self.config.data.items_path.exists():
                raise FileNotFoundError(f"Items file not found: {self.config.data.items_path}")

            interactions = self._validate_interactions(load_csv(self.config.data.interactions_path))
            items = self._validate_items(load_csv(self.config.data.items_path))

            unknown_items = set(interactions[self.config.data.item_column].astype(str)) - set(
                items[self.config.data.item_column].astype(str)
            )
            if unknown_items:
                logger.warning("Found %d unknown item_ids in interactions; they will be ignored downstream", len(unknown_items))

            logger.info("Loaded interactions shape=%s items shape=%s", interactions.shape, items.shape)
            return interactions, items
        except Exception as exc:
            raise RecommendationException("Failed to load recommendation data", exc) from exc
