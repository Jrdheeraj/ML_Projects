"""Generate portfolio-ready visual assets and real prediction samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.configuration import AppConfig
from src.pipelines.prediction_pipeline import run_prediction_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.io_utils import load_dataframe, load_json, save_json


def _ensure_training_artifacts(config: AppConfig) -> None:
    if not config.model.model_path.exists() or not config.model.metrics_path.exists():
        run_training_pipeline(config)


def _save_eda_distribution(df: pd.DataFrame, assets_dir: Path) -> Path:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_counts = df["label"].value_counts().sort_index()
    class_labels = class_counts.index.astype(str)
    sns.barplot(
        x=class_labels,
        y=class_counts.values,
        hue=class_labels,
        legend=False,
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title("Class Distribution (0: Normal, 1: Fraud)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    sns.histplot(
        data=df,
        x="transaction_amount",
        hue="label",
        bins=35,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].set_title("Transaction Amount Density by Class")
    axes[1].set_xlabel("Transaction Amount")

    fig.suptitle("Fraud Data Distribution Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = assets_dir / "eda_distribution.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_missing_values(df: pd.DataFrame, assets_dir: Path) -> Path:
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100.0).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        x=summary.index,
        y=summary["missing_pct"],
        ax=ax,
        color="#4C78A8",
    )
    ax.set_title("Missing Values by Feature (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.set_ylim(0, max(1.0, float(summary["missing_pct"].max()) + 1.0))
    fig.tight_layout()

    path = assets_dir / "missing_values.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_correlation_heatmap(df: pd.DataFrame, assets_dir: Path) -> Path:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    corr = df[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = assets_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_model_comparison(artifacts_dir: Path, assets_dir: Path) -> Path:
    comparison_df = pd.read_csv(artifacts_dir / "strategy_comparison.csv")
    comparison_df = comparison_df.sort_values("cv_average_precision", ascending=False)
    comparison_df["label"] = comparison_df["model"] + " | " + comparison_df["imbalance_strategy"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=comparison_df,
        y="label",
        x="cv_average_precision",
        hue="label",
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Model + Imbalance Strategy Comparison (CV AP)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Precision (CV)")
    ax.set_ylabel("Candidate")
    fig.tight_layout()

    path = assets_dir / "model_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_confusion_matrix(artifacts_dir: Path, assets_dir: Path) -> Path:
    cm = load_json(artifacts_dir / "confusion_analysis.json")
    matrix = [
        [cm["true_negative"], cm["false_positive"]],
        [cm["false_negative"], cm["true_positive"]],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: Normal", "Pred: Fraud"],
        yticklabels=["Actual: Normal", "Actual: Fraud"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = assets_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_roc_curve(artifacts_dir: Path, assets_dir: Path, roc_auc: float) -> Path:
    roc_df = pd.read_csv(artifacts_dir / "roc_curve.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(roc_df["fpr"], roc_df["tpr"], color="#1f77b4", linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = assets_dir / "roc_curve.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_feature_importance(artifacts_dir: Path, assets_dir: Path) -> Path:
    importance_df = pd.read_csv(artifacts_dir / "feature_importance.csv").head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="magma",
        ax=ax,
    )
    ax.set_title("Top Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    path = assets_dir / "feature_importance.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _generate_prediction_samples(df: pd.DataFrame, config: AppConfig, output_path: Path) -> list[dict[str, Any]]:
    samples_df = df.sample(n=3, random_state=42).copy()
    records: list[dict[str, Any]] = []

    for _, row in samples_df.iterrows():
        model_input = {
            "transaction_id": str(row["transaction_id"]),
            "user_id": int(row["user_id"]),
            "transaction_amount": float(row["transaction_amount"]),
            "transaction_time": str(row["transaction_time"]),
            "location": str(row["location"]),
            "device": str(row["device"]),
            "merchant_category": str(row["merchant_category"]),
            "payment_channel": str(row["payment_channel"]),
            "is_international": int(row["is_international"]),
            "card_present": int(row["card_present"]),
            "previous_transactions_24h": int(row["previous_transactions_24h"]),
            "avg_spend_7d": float(row["avg_spend_7d"]),
        }
        prediction = run_prediction_pipeline(model_input, config=config)
        records.append(
            {
                "input": model_input,
                "actual_label": int(row["label"]),
                "output": {
                    "prediction": "Fraud" if prediction["fraud"] else "Normal",
                    "probability": round(float(prediction["probability"]), 4),
                    "threshold": round(float(prediction["threshold"]), 4),
                    "model_version": prediction["model_version"],
                },
            }
        )

    save_json({"samples": records}, output_path)
    return records


def _save_api_response_preview(sample_payload: dict[str, Any], assets_dir: Path) -> Path:
    first = sample_payload["samples"][0]
    request_json = json.dumps(first["input"], indent=2)
    response_json = json.dumps(first["output"], indent=2)

    fig = plt.figure(figsize=(13, 6), facecolor="#0f172a")
    left = fig.add_axes([0.05, 0.12, 0.42, 0.76])
    right = fig.add_axes([0.53, 0.12, 0.42, 0.76])

    for axis, title, text, color in [
        (left, "Request JSON", request_json, "#111827"),
        (right, "Response JSON", response_json, "#0b3b2e"),
    ]:
        axis.set_facecolor(color)
        axis.text(0.03, 0.98, title, fontsize=12, color="white", va="top", ha="left", weight="bold")
        axis.text(
            0.03,
            0.92,
            text,
            family="monospace",
            fontsize=9,
            color="#e5e7eb",
            va="top",
            ha="left",
            wrap=True,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#334155")

    fig.suptitle("Fraud Detection API Demo Output", fontsize=16, color="white", weight="bold")
    path = assets_dir / "api_response.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def run_portfolio_showcase_pipeline(config: AppConfig | None = None) -> dict[str, str]:
    config = config or AppConfig.from_env()
    project_root = config.project_root
    assets_dir = project_root / "assets"
    artifacts_dir = project_root / "artifacts"
    outputs_dir = project_root / "artifacts"
    assets_dir.mkdir(parents=True, exist_ok=True)

    _ensure_training_artifacts(config)

    raw_df = load_dataframe(config.data.raw_data_path)
    metrics = load_json(config.model.metrics_path)

    eda_distribution = _save_eda_distribution(raw_df, assets_dir)
    missing_values = _save_missing_values(raw_df, assets_dir)
    correlation = _save_correlation_heatmap(raw_df, assets_dir)
    model_comparison = _save_model_comparison(artifacts_dir, assets_dir)
    confusion = _save_confusion_matrix(artifacts_dir, assets_dir)
    roc_curve = _save_roc_curve(artifacts_dir, assets_dir, roc_auc=float(metrics.get("roc_auc", 0.0)))
    feature_importance = _save_feature_importance(artifacts_dir, assets_dir)

    output_samples_path = outputs_dir / "output_samples.json"
    samples = _generate_prediction_samples(raw_df, config, output_samples_path)
    sample_payload = {"samples": samples}
    api_response = _save_api_response_preview(sample_payload, assets_dir)

    summary = {
        "eda_distribution": str(eda_distribution),
        "missing_values": str(missing_values),
        "correlation_heatmap": str(correlation),
        "model_comparison": str(model_comparison),
        "confusion_matrix": str(confusion),
        "roc_curve": str(roc_curve),
        "feature_importance": str(feature_importance),
        "api_response": str(api_response),
        "output_samples": str(output_samples_path),
    }
    save_json(summary, outputs_dir / "showcase_assets_manifest.json")
    return summary


if __name__ == "__main__":
    run_portfolio_showcase_pipeline()