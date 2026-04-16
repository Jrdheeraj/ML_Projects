"""Generate portfolio visuals and real prediction outputs for churn project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.components.feature_engineering import ChurnFeatureEngineer
from src.components.preprocessing import build_preprocessor
from src.config.configuration import AppConfig
from src.pipelines.prediction_pipeline import run_prediction_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.io_utils import load_dataframe, load_json, load_model, save_json
from src.utils.model_utils import churn_probability_from_model


def _normalize_target(y: pd.Series) -> pd.Series:
    if y.dtype.kind in {"O", "U", "S"}:
        y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    return y.astype(int)


def _ensure_training_artifacts(config: AppConfig) -> None:
    required_paths = [
        config.model.model_output_path,
        config.model.metrics_output_path,
        config.data.train_data_path,
        config.data.test_data_path,
        config.model.feature_schema_output_path,
    ]
    if not all(path.exists() for path in required_paths):
        run_training_pipeline(config)


def _load_train_test(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = load_dataframe(config.data.train_data_path)
    test_df = load_dataframe(config.data.test_data_path)
    target_col = config.data.target_column
    x_train = train_df.drop(columns=[target_col])
    y_train = _normalize_target(train_df[target_col])
    x_test = test_df.drop(columns=[target_col])
    y_test = _normalize_target(test_df[target_col])
    return train_df, test_df, x_train, y_train, x_test, y_test


def _save_eda_distribution(raw_df: pd.DataFrame, assets_dir: Path, target_col: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_counts = raw_df[target_col].astype(str).value_counts()
    labels = class_counts.index.astype(str)
    sns.barplot(x=labels, y=class_counts.values, hue=labels, legend=False, palette="Set2", ax=axes[0])
    axes[0].set_title("Churn Class Distribution")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Count")

    if "MonthlyCharges" in raw_df.columns:
        sns.histplot(
            data=raw_df,
            x="MonthlyCharges",
            hue=target_col,
            bins=30,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=axes[1],
        )
        axes[1].set_title("Monthly Charges by Churn Class")
    else:
        axes[1].text(0.5, 0.5, "MonthlyCharges not available", ha="center", va="center")
        axes[1].set_axis_off()

    fig.suptitle("Customer Churn EDA Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output = assets_dir / "eda_distribution.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_missing_values(raw_df: pd.DataFrame, assets_dir: Path) -> Path:
    missing_pct = (raw_df.isna().mean() * 100.0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=missing_pct.index.astype(str), y=missing_pct.values, color="#4C78A8", ax=ax)
    ax.set_title("Missing Values by Feature (%)")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    output = assets_dir / "missing_values.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_correlation_heatmap(raw_df: pd.DataFrame, assets_dir: Path, target_col: str) -> Path:
    df = raw_df.copy()
    if target_col in df.columns:
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
        )

    numeric_df = df.select_dtypes(include=["number", "bool"]).copy()
    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=0.4, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()

    output = assets_dir / "correlation_heatmap.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _build_candidate_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", None),
            ("model", model),
        ]
    )


def _evaluate_model_candidates(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    candidates: list[tuple[str, Any]] = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]

    try:
        from xgboost import XGBClassifier

        candidates.append(
            (
                "XGBoost",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    tree_method="hist",
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.08,
                ),
            )
        )
    except Exception:
        pass

    rows: list[dict[str, float | str]] = []
    feature_engineer = ChurnFeatureEngineer()
    preprocessor = build_preprocessor(feature_engineer.fit_transform(x_train))

    for name, model in candidates:
        pipeline = Pipeline(
            steps=[
                ("feature_engineering", feature_engineer),
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        prob = churn_probability_from_model(pipeline, x_test)
        rows.append(
            {
                "model": name,
                "f1_score": float(f1_score(y_test, pred, zero_division=0)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, prob)),
            }
        )

    return pd.DataFrame(rows).sort_values("f1_score", ascending=False)


def _save_model_comparison(model_comparison_df: pd.DataFrame, assets_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=model_comparison_df,
        y="model",
        x="f1_score",
        hue="model",
        legend=False,
        palette="mako",
        ax=ax,
    )
    ax.set_title("Model Comparison by F1 Score")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Model")
    fig.tight_layout()

    output = assets_dir / "model_comparison.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, assets_dir: Path) -> tuple[Path, dict[str, int]]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["Actual: No", "Actual: Yes"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    output = assets_dir / "confusion_matrix.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output, {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def _save_roc_curve(y_true: pd.Series, y_prob: np.ndarray, assets_dir: Path) -> tuple[Path, float]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = float(roc_auc_score(y_true, y_prob))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, color="#1f77b4", label=f"ROC AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output = assets_dir / "roc_curve.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output, auc_value


def _save_feature_importance(model, x_test: pd.DataFrame, y_test: pd.Series, assets_dir: Path) -> Path:
    result = permutation_importance(
        estimator=model,
        X=x_test,
        y=y_test,
        scoring="f1",
        n_repeats=8,
        random_state=42,
    )
    importances = pd.DataFrame(
        {
            "feature": x_test.columns.astype(str),
            "importance": result.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    top = importances.head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, x="importance", y="feature", hue="feature", legend=False, palette="rocket", ax=ax)
    ax.set_title("Top Feature Importance (Permutation)")
    ax.set_xlabel("Mean F1 Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    output = assets_dir / "feature_importance.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _generate_output_samples(raw_df: pd.DataFrame, config: AppConfig, output_path: Path, target_col: str) -> dict[str, Any]:
    sample_pool = raw_df.drop(columns=[target_col], errors="ignore")
    sample_rows = sample_pool.sample(n=min(3, len(sample_pool)), random_state=42)

    payload_rows: list[dict[str, Any]] = []
    for _, row in sample_rows.iterrows():
        record = row.to_dict()
        prediction = run_prediction_pipeline(record, config=config)
        payload_rows.append(
            {
                "input": record,
                "output": {
                    "prediction": prediction["churn"],
                    "probability": round(float(prediction["probability"]), 4),
                },
            }
        )

    payload = {"samples": payload_rows}
    save_json(payload, output_path)
    return payload


def _save_api_response_image(output_samples: dict[str, Any], assets_dir: Path) -> Path:
    first = output_samples["samples"][0]
    req_text = json.dumps(first["input"], indent=2)
    resp_text = json.dumps(first["output"], indent=2)

    fig = plt.figure(figsize=(13, 6), facecolor="#0f172a")
    left = fig.add_axes([0.05, 0.12, 0.42, 0.76])
    right = fig.add_axes([0.53, 0.12, 0.42, 0.76])

    for axis, title, text, color in [
        (left, "Request JSON", req_text, "#111827"),
        (right, "Response JSON", resp_text, "#0b3b2e"),
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

    fig.suptitle("Customer Churn API Demo Output", fontsize=16, color="white", weight="bold")
    output = assets_dir / "api_response.png"
    fig.savefig(output, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def run_portfolio_showcase_pipeline(config: AppConfig | None = None) -> dict[str, Any]:
    config = config or AppConfig.from_env()
    assets_dir = config.project_root / "assets"
    artifacts_dir = config.project_root / "artifacts"
    assets_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    _ensure_training_artifacts(config)

    raw_df = load_dataframe(config.data.raw_data_path)
    metrics_payload = load_json(config.model.metrics_output_path)
    _, _, x_train, y_train, x_test, y_test = _load_train_test(config)
    model = load_model(config.model.model_output_path)

    eda_distribution = _save_eda_distribution(raw_df, assets_dir, config.data.target_column)
    missing_values = _save_missing_values(raw_df, assets_dir)
    correlation_heatmap = _save_correlation_heatmap(raw_df, assets_dir, config.data.target_column)

    model_comparison_df = _evaluate_model_candidates(x_train, y_train, x_test, y_test)
    model_comparison_path = assets_dir / "model_comparison.png"
    _save_model_comparison(model_comparison_df, assets_dir)
    model_comparison_df.to_csv(artifacts_dir / "model_comparison.csv", index=False)

    y_prob = churn_probability_from_model(model, x_test)
    y_pred = (y_prob >= 0.5).astype(int)
    confusion_path, confusion_summary = _save_confusion_matrix(y_test, y_pred, assets_dir)
    roc_curve_path, roc_auc = _save_roc_curve(y_test, y_prob, assets_dir)
    feature_importance = _save_feature_importance(model, x_test, y_test, assets_dir)

    output_samples_path = artifacts_dir / "output_samples.json"
    output_samples = _generate_output_samples(raw_df, config, output_samples_path, config.data.target_column)
    api_response = _save_api_response_image(output_samples, assets_dir)

    binary_metrics = {
        "roc_auc": roc_auc,
        "confusion_summary": confusion_summary,
    }
    save_json(binary_metrics, artifacts_dir / "binary_relevance_metrics.json")

    summary = {
        "assets": {
            "eda_distribution": str(eda_distribution),
            "missing_values": str(missing_values),
            "correlation_heatmap": str(correlation_heatmap),
            "model_comparison": str(model_comparison_path),
            "confusion_matrix": str(confusion_path),
            "roc_curve": str(roc_curve_path),
            "feature_importance": str(feature_importance),
            "api_response": str(api_response),
        },
        "artifacts": {
            "output_samples": str(output_samples_path),
            "model_comparison_csv": str(artifacts_dir / "model_comparison.csv"),
            "binary_relevance_metrics": str(artifacts_dir / "binary_relevance_metrics.json"),
            "metrics": str(config.model.metrics_output_path),
        },
        "training_summary": metrics_payload,
    }
    save_json(summary, artifacts_dir / "showcase_assets_manifest.json")
    return summary


if __name__ == "__main__":
    run_portfolio_showcase_pipeline()