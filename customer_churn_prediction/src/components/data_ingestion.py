"""Data ingestion for customer churn dataset."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.config.configuration import AppConfig
from src.exception.custom_exception import ChurnException
from src.logger.logging import get_logger
from src.utils.io_utils import load_dataframe, save_dataframe


logger = get_logger(__name__)


@dataclass
class DataIngestion:
    config: AppConfig

    def run(self) -> tuple[str, str]:
        try:
            raw_path = self.config.data.raw_data_path
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {raw_path}. Place Telco churn CSV there or set RAW_DATA_PATH."
                )

            df = load_dataframe(raw_path)
            logger.info("Loaded raw dataset with shape %s", df.shape)

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.data.test_size,
                random_state=self.config.data.random_state,
                stratify=df[self.config.data.target_column] if self.config.data.target_column in df.columns else None,
            )

            save_dataframe(train_df, self.config.data.train_data_path)
            save_dataframe(test_df, self.config.data.test_data_path)
            logger.info(
                "Saved split datasets to %s and %s",
                self.config.data.train_data_path,
                self.config.data.test_data_path,
            )
            return str(self.config.data.train_data_path), str(self.config.data.test_data_path)
        except Exception as exc:
            raise ChurnException("Data ingestion failed", exc) from exc
