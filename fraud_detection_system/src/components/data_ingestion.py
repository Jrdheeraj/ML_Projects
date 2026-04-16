"""Data ingestion component for fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import AppConfig
from src.exception.custom_exception import FraudDetectionException
from src.logger.logging import get_logger
from src.utils.io_utils import load_dataframe, save_dataframe


logger = get_logger(__name__)


@dataclass
class DataIngestion:
    """Load and split fraud transaction data."""

    config: AppConfig

    def _validate_schema(self, df: pd.DataFrame) -> None:
        required_columns = {
            "transaction_id",
            "user_id",
            "transaction_amount",
            "transaction_time",
            "location",
            "device",
            "merchant_category",
            "payment_channel",
            "is_international",
            "card_present",
            "previous_transactions_24h",
            "avg_spend_7d",
            self.config.data.target_column,
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in fraud dataset: {sorted(missing)}")

        if df.empty:
            raise ValueError("Input transaction dataset is empty")

        if df["transaction_id"].isna().any():
            raise ValueError("transaction_id contains null values")
        if df["user_id"].isna().any():
            raise ValueError("user_id contains null values")

        label_values = set(pd.to_numeric(df[self.config.data.target_column], errors="coerce").dropna().astype(int).unique())
        if not label_values.issubset({0, 1}):
            raise ValueError(
                f"Target column '{self.config.data.target_column}' must be binary 0/1, found {sorted(label_values)}"
            )

        duplicate_ids = int(df["transaction_id"].duplicated().sum())
        if duplicate_ids > 0:
            logger.warning("Found %d duplicate transaction_id values; duplicates will be dropped", duplicate_ids)

        parsed_time = pd.to_datetime(df["transaction_time"], errors="coerce")
        invalid_time_count = int(parsed_time.isna().sum())
        if invalid_time_count > 0:
            raise ValueError(f"Found {invalid_time_count} invalid transaction_time values")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            if not self.config.data.raw_data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {self.config.data.raw_data_path}."
                )

            df = load_dataframe(self.config.data.raw_data_path)
            self._validate_schema(df)
            df = df.drop_duplicates(subset=["transaction_id"], keep="first").copy()

            target = df[self.config.data.target_column].astype(int)
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.data.test_size,
                random_state=self.config.data.random_state,
                stratify=target,
            )

            save_dataframe(train_df, self.config.data.train_data_path)
            save_dataframe(test_df, self.config.data.test_data_path)
            logger.info("Data ingestion complete train=%s test=%s", train_df.shape, test_df.shape)

            return train_df, test_df
        except Exception as exc:
            raise FraudDetectionException("Data ingestion failed", exc) from exc
