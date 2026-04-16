"""Generate recruiter-ready portfolio visuals and real recommendation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.components.evaluation import precision_at_k
from src.components.preprocessing import RecommendationPreprocessor
from src.config.configuration import AppConfig
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.io_utils import load_csv, load_json, load_model, save_json


def _ensure_trained(config: AppConfig) -> None:
    if not config.model.model_path.exists() or not config.model.metrics_path.exists():
        run_training_pipeline(config)


def _prepare_data(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = load_csv(config.data.interactions_path)
    items = load_csv(config.data.items_path)
    preprocessor = RecommendationPreprocessor(config)
    interactions_prepared = preprocessor.prepare_interactions(interactions)
    items_prepared = preprocessor.prepare_items(items)
    train_df, test_df = preprocessor.split_train_test_by_user(interactions_prepared)
    return interactions_prepared, items_prepared, train_df, test_df


def _save_eda_distribution(interactions: pd.DataFrame, assets_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    per_user = interactions.groupby("user_id").size()
    sns.histplot(per_user, bins=25, kde=True, ax=axes[0], color="#2D728F")
    axes[0].set_title("Interactions Per User")
    axes[0].set_xlabel("Interaction Count")

    top_items = interactions.groupby("item_id").size().sort_values(ascending=False).head(12)
    item_labels = top_items.index.astype(str)
    sns.barplot(
        x=item_labels,
        y=top_items.values,
        hue=item_labels,
        legend=False,
        ax=axes[1],
        palette="viridis",
    )
    axes[1].set_title("Top Items by Interaction Volume")
    axes[1].set_xlabel("Item")
    axes[1].set_ylabel("Interactions")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Recommendation EDA Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output = assets_dir / "eda_distribution.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_missing_values(interactions: pd.DataFrame, items: pd.DataFrame, assets_dir: Path) -> Path:
    inter_missing = interactions.isna().mean() * 100.0
    item_missing = items.isna().mean() * 100.0
    missing = pd.concat([inter_missing, item_missing], axis=0).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=missing.index.astype(str), y=missing.values, ax=ax, color="#4C78A8")
    ax.set_title("Missing Values by Feature (%)")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=50)
    fig.tight_layout()

    output = assets_dir / "missing_values.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_correlation_heatmap(interactions: pd.DataFrame, assets_dir: Path) -> Path:
    matrix_df = interactions.copy()
    matrix_df["interaction_type_code"] = matrix_df["interaction_type"].astype("category").cat.codes
    numeric_cols = [
        col
        for col in ["user_id", "rating", "signal", "interaction_type_code"]
        if col in matrix_df.columns
    ]
    corr = matrix_df[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Interaction Feature Correlation")
    fig.tight_layout()

    output = assets_dir / "correlation_heatmap.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _evaluate_weight_presets(model, test_df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    presets: list[tuple[str, float, float, float]] = [
        ("Collaborative Only", 1.0, 0.0, 0.0),
        ("Content Only", 0.0, 1.0, 0.0),
        ("Popularity Only", 0.0, 0.0, 1.0),
        ("Balanced Hybrid", 0.55, 0.30, 0.15),
        ("Content Leaning", 0.40, 0.45, 0.15),
    ]

    rows: list[dict[str, float | str]] = []
    for name, cf_w, cb_w, pop_w in presets:
        model.set_weights(cf_w, cb_w, pop_w)

        precisions: list[float] = []
        recalls: list[float] = []
        y_true: list[float] = []
        y_pred: list[float] = []
        for user_id, user_test in test_df.groupby(config.data.user_column):
            relevant = set(user_test[config.data.item_column].astype(str).tolist())
            recs = model.recommend(int(user_id), top_n=config.model.top_k_eval)
            precisions.append(precision_at_k(recs, relevant, config.model.top_k_eval))
            recalls.append(len(set(recs[: config.model.top_k_eval]) & relevant) / max(len(relevant), 1))
            for _, row in user_test.iterrows():
                y_true.append(float(row[config.data.target_column]))
                y_pred.append(float(model.predict_score(int(user_id), str(row[config.data.item_column]))))

        rmse_value = float(np.sqrt(np.mean(np.square(np.array(y_true) - np.array(y_pred))))) if y_true else 0.0
        rows.append(
            {
                "strategy": name,
                "precision_at_k": float(np.mean(precisions) if precisions else 0.0),
                "recall_at_k": float(np.mean(recalls) if recalls else 0.0),
                "rmse": rmse_value,
                "cf_weight": cf_w,
                "cb_weight": cb_w,
                "popularity_weight": pop_w,
            }
        )

    result = pd.DataFrame(rows).sort_values("precision_at_k", ascending=False)
    winner = result.iloc[0]
    model.set_weights(
        float(winner["cf_weight"]),
        float(winner["cb_weight"]),
        float(winner["popularity_weight"]),
    )
    return result


def _save_model_comparison(model_comparison_df: pd.DataFrame, assets_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=model_comparison_df,
        y="strategy",
        x="precision_at_k",
        hue="strategy",
        legend=False,
        palette="mako",
        ax=ax,
    )
    ax.set_title("Model Strategy Comparison (Precision@K)")
    ax.set_xlabel("Precision@K")
    ax.set_ylabel("Strategy")
    fig.tight_layout()

    output = assets_dir / "model_comparison.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _binary_eval_curves(model, test_df: pd.DataFrame, config: AppConfig) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    y_true_bin: list[int] = []
    y_score: list[float] = []

    for _, row in test_df.iterrows():
        user_id = int(row[config.data.user_column])
        item_id = str(row[config.data.item_column])
        actual_signal = float(row[config.data.target_column])
        pred_signal = float(model.predict_score(user_id, item_id))

        y_true_bin.append(1 if actual_signal >= 4.0 else 0)
        y_score.append(np.clip(pred_signal / 5.0, 0.0, 1.0))

    y_true_arr = np.array(y_true_bin, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    roc_auc = float(auc(fpr, tpr))
    return y_true_arr, y_score_arr, roc_auc, thresholds


def _save_confusion_matrix(y_true_arr: np.ndarray, y_score_arr: np.ndarray, assets_dir: Path) -> tuple[Path, dict[str, int | float]]:
    thresholds = np.linspace(0.3, 0.8, num=11)
    best_threshold = 0.5
    best_score = -1.0
    for thr in thresholds:
        pred = (y_score_arr >= thr).astype(int)
        tp = int(((pred == 1) & (y_true_arr == 1)).sum())
        fp = int(((pred == 1) & (y_true_arr == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(int((y_true_arr == 1).sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best_score:
            best_score = f1
            best_threshold = float(thr)

    y_pred_bin = (y_score_arr >= best_threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred_bin)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Low", "Pred High"],
        yticklabels=["Actual Low", "Actual High"],
        ax=ax,
    )
    ax.set_title("Binary Relevance Confusion Matrix")
    fig.tight_layout()

    output = assets_dir / "confusion_matrix.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "threshold": best_threshold,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    return output, summary


def _save_roc_curve(y_true_arr: np.ndarray, y_score_arr: np.ndarray, assets_dir: Path) -> tuple[Path, float]:
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Binary Relevance ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output = assets_dir / "roc_curve.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output, roc_auc


def _save_feature_importance(model, assets_dir: Path) -> Path:
    weights = model.recommendation_weights
    importance = pd.DataFrame(
        {
            "feature": ["Collaborative Signal", "Content Signal", "Popularity Signal"],
            "importance": [
                float(weights.get("cf", 0.0)),
                float(weights.get("cb", 0.0)),
                float(weights.get("pop", 0.0)),
            ],
        }
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=importance,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="rocket",
        ax=ax,
    )
    ax.set_title("Hybrid Signal Importance")
    ax.set_xlabel("Normalized Weight")
    ax.set_ylabel("Signal")
    fig.tight_layout()

    output = assets_dir / "feature_importance.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_output_samples(model, train_df: pd.DataFrame, config: AppConfig, output_path: Path) -> dict[str, Any]:
    users = train_df[config.data.user_column].drop_duplicates().sample(n=min(3, train_df[config.data.user_column].nunique()), random_state=42)
    rows: list[dict[str, Any]] = []
    for user_id in users.tolist():
        top_n = 5
        recs = model.recommend(int(user_id), top_n=top_n)
        scored = [
            {
                "item_id": item_id,
                "predicted_signal": round(float(model.predict_score(int(user_id), str(item_id))), 4),
                "probability_like": round(float(np.clip(model.predict_score(int(user_id), str(item_id)) / 5.0, 0.0, 1.0)), 4),
            }
            for item_id in recs
        ]
        rows.append(
            {
                "input": {"user_id": int(user_id), "top_n": top_n},
                "output": {
                    "recommendations": recs,
                    "recommendations_with_scores": scored,
                },
            }
        )

    payload = {"samples": rows}
    save_json(payload, output_path)
    return payload


def _save_api_response_image(payload: dict[str, Any], assets_dir: Path) -> Path:
    first = payload["samples"][0]
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
        axis.text(0.03, 0.92, text, family="monospace", fontsize=9, color="#e5e7eb", va="top", ha="left", wrap=True)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#334155")

    fig.suptitle("Recommendation API Demo Output", fontsize=16, color="white", weight="bold")
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
    _ensure_trained(config)

    interactions, items, train_df, test_df = _prepare_data(config)
    model = load_model(config.model.model_path)
    metrics_payload = load_json(config.model.metrics_path)

    eda_distribution = _save_eda_distribution(interactions, assets_dir)
    missing_values = _save_missing_values(interactions, items, assets_dir)
    corr = _save_correlation_heatmap(interactions, assets_dir)

    model_comparison_df = _evaluate_weight_presets(model, test_df, config)
    model_comparison = _save_model_comparison(model_comparison_df, assets_dir)

    y_true_arr, y_score_arr, _, _ = _binary_eval_curves(model, test_df, config)
    confusion_matrix_path, confusion_summary = _save_confusion_matrix(y_true_arr, y_score_arr, assets_dir)
    roc_curve_path, roc_auc = _save_roc_curve(y_true_arr, y_score_arr, assets_dir)
    feature_importance = _save_feature_importance(model, assets_dir)

    output_samples_path = artifacts_dir / "output_samples.json"
    output_payload = _save_output_samples(model, train_df, config, output_samples_path)
    api_response = _save_api_response_image(output_payload, assets_dir)

    binary_eval_path = artifacts_dir / "binary_relevance_metrics.json"
    save_json(
        {
            "roc_auc": roc_auc,
            "confusion_summary": confusion_summary,
        },
        binary_eval_path,
    )
    model_comparison_csv = artifacts_dir / "model_comparison.csv"
    model_comparison_df.to_csv(model_comparison_csv, index=False)

    summary = {
        "assets": {
            "eda_distribution": str(eda_distribution),
            "missing_values": str(missing_values),
            "correlation_heatmap": str(corr),
            "model_comparison": str(model_comparison),
            "confusion_matrix": str(confusion_matrix_path),
            "roc_curve": str(roc_curve_path),
            "feature_importance": str(feature_importance),
            "api_response": str(api_response),
        },
        "artifacts": {
            "output_samples": str(output_samples_path),
            "binary_relevance_metrics": str(binary_eval_path),
            "model_comparison_csv": str(model_comparison_csv),
            "existing_metrics": str(config.model.metrics_path),
        },
        "training_metrics": metrics_payload.get("metrics", {}),
        "weights": metrics_payload.get("weights", {}),
    }
    save_json(summary, artifacts_dir / "showcase_assets_manifest.json")
    return summary


if __name__ == "__main__":
    run_portfolio_showcase_pipeline()