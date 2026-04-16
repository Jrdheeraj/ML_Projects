"""Feature engineering for fraud transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-driven fraud features robustly."""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()

        df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
        df["avg_spend_7d"] = pd.to_numeric(df["avg_spend_7d"], errors="coerce")
        df["previous_transactions_24h"] = pd.to_numeric(
            df["previous_transactions_24h"], errors="coerce"
        )
        df["is_international"] = pd.to_numeric(df["is_international"], errors="coerce")
        df["card_present"] = pd.to_numeric(df["card_present"], errors="coerce")

        tx_time = pd.to_datetime(df["transaction_time"], errors="coerce")
        df["transaction_hour"] = tx_time.dt.hour
        df["transaction_dayofweek"] = tx_time.dt.dayofweek

        df["amount_to_avg_ratio"] = df["transaction_amount"] / (df["avg_spend_7d"] + 1.0)
        df["high_velocity_flag"] = (df["previous_transactions_24h"] >= 8).astype(int)
        df["night_transaction"] = df["transaction_hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
        df["intl_card_absent"] = (
            (df["is_international"] == 1) & (df["card_present"] == 0)
        ).astype(int)

        drop_cols = [column for column in ["transaction_id", "transaction_time"] if column in df.columns]
        return df.drop(columns=drop_cols)
