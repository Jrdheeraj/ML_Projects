"""Feature engineering transformer for churn data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-inspired churn features while staying robust to schema variation."""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()

        # Normalize common telco numeric fields that may arrive as strings.
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        if "tenure" in df.columns:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

        if {"TotalCharges", "tenure"}.issubset(df.columns):
            df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1.0)

        if "tenure" in df.columns:
            bins = [-1, 6, 12, 24, 48, np.inf]
            labels = ["0_6", "7_12", "13_24", "25_48", "49_plus"]
            df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

        service_cols = [
            col
            for col in [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
            if col in df.columns
        ]
        if service_cols:
            df["active_services_count"] = (
                df[service_cols].replace({"Yes": 1, "No": 0, "No internet service": 0}).fillna(0).sum(axis=1)
            )

        return df
