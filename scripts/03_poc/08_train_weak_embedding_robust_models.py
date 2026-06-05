#!/usr/bin/env python3
"""Train robust weak-label tRNA classifiers from frozen protein embeddings.

This script is intended to be the scientifically stricter follow-up to the
LazyPredict screen:

1. Keep the MMseqs cluster train/validation/test split fixed.
2. Use train only for hyperparameter screening.
3. Select regularization and threshold on validation only.
4. Refit the selected model on train+validation.
5. Evaluate once on the held-out test split.

The mechanism task defaults to mechanism-labeled tRNA modifier rows only, so
unknown/unrelated proteins are not treated as negatives for mechanism labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_LABEL_MATRIX = "data/processed/poc_weak/weak_trna_mod_label_matrix.tsv"
DEFAULT_SPLIT_ASSIGNMENTS = "data/processed/poc_weak/splits/mmseqs50/split_assignments.tsv"
DEFAULT_EMBEDDING_DIR = "data/processed/poc_weak/embeddings/esmc_6b"
DEFAULT_OUTPUT_DIR = "data/processed/poc_weak/ml_runs"
DEFAULT_RUN_NAME = "esmc6b_robust_logreg"
DEFAULT_C_GRID = "0.003,0.01,0.03,0.1,0.3,1,3,10"
DEFAULT_MIN_TRAIN_POSITIVES = 10
DEFAULT_MIN_EVAL_POSITIVES = 1
DEFAULT_MECHANISM_SCOPE = "mechanism_labeled"
DEFAULT_RANDOM_SEED = 42


def find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from script path.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
os.chdir(REPO_ROOT)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-matrix", default=DEFAULT_LABEL_MATRIX)
    parser.add_argument("--split-assignments", default=DEFAULT_SPLIT_ASSIGNMENTS)
    parser.add_argument("--embedding-dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--embedding-glob", default="*")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--task", choices=("binary", "mechanism", "all"), default="all")
    parser.add_argument(
        "--mechanism-scope",
        choices=("all_trainable", "trna_modifiers", "mechanism_labeled"),
        default=DEFAULT_MECHANISM_SCOPE,
    )
    parser.add_argument("--min-train-positives", type=int, default=DEFAULT_MIN_TRAIN_POSITIVES)
    parser.add_argument("--min-val-positives", type=int, default=DEFAULT_MIN_EVAL_POSITIVES)
    parser.add_argument("--min-test-positives", type=int, default=DEFAULT_MIN_EVAL_POSITIVES)
    parser.add_argument("--c-grid", default=DEFAULT_C_GRID)
    parser.add_argument("--rank-metric", choices=("aupr", "auroc", "f1"), default="aupr")
    parser.add_argument("--threshold-metric", choices=("f1", "balanced_accuracy"), default="f1")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-embedding-hash-audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_c_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid:
        raise SystemExit("--c-grid must contain at least one value.")
    return grid


def slugify(value: str) -> str:
    value = value.replace("target__", "").replace("mechanism__", "")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "label"


def strip_known_suffix(path: Path) -> str:
    return re.sub(r"_hidden_layer_steps\d+$", "", path.stem)


def binary_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)


def is_mechanism_column(column: str) -> bool:
    return column.startswith("mechanism__")


def mechanism_scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all_trainable":
        return pd.Series(True, index=frame.index)
    if scope == "trna_modifiers":
        return binary_series(frame, "target__trna_modifier").eq(1)
    if scope == "mechanism_labeled":
        return binary_series(frame, "has_mechanism_label").eq(1)
    raise ValueError(f"Unknown mechanism scope: {scope}")


def build_embedding_index(embedding_dir: Path, pattern: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(embedding_dir.glob(pattern)):
        if path.suffix.lower() not in {".npy", ".pt"}:
            continue
        index[path.stem] = path
        index[strip_known_suffix(path)] = path
    return index


def load_embedding(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".pt":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Loading .pt embeddings requires torch.") from exc
        tensor = torch.load(path, map_location="cpu")
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
        arr = np.asarray(tensor, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding extension: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1).mean(axis=0)
    raise ValueError(f"Unsupported embedding shape for {path}: {arr.shape}")


def load_rows_with_embeddings(frame: pd.DataFrame, embedding_dir: Path, pattern: str) -> tuple[np.ndarray, pd.DataFrame]:
    index = build_embedding_index(embedding_dir, pattern)
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        accession = str(row["accession"])
        path = index.get(accession)
        if path is None:
            continue
        vectors.append(load_embedding(path))
        rows.append(row)
    if not vectors:
        raise RuntimeError(f"No embeddings matched accessions in {embedding_dir}")
    return np.stack(vectors, axis=0), pd.DataFrame(rows)


def selected_label_columns(frame: pd.DataFrame, task: str) -> list[str]:
    columns: list[str] = []
    if task in {"binary", "all"}:
        columns.append("target__trna_modifier")
    if task in {"mechanism", "all"}:
        columns.extend([column for column in frame.columns if is_mechanism_column(column)])
    return columns


def label_scope(column: str, mechanism_scope: str) -> tuple[str, str]:
    if is_mechanism_column(column):
        return "mechanism", mechanism_scope
    return "binary", "all_trainable"


def scope_mask_for_column(rows: pd.DataFrame, column: str, mechanism_scope: str) -> np.ndarray:
    if is_mechanism_column(column):
        return mechanism_scope_mask(rows, mechanism_scope).to_numpy(dtype=bool)
    return np.ones(len(rows), dtype=bool)


def safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def threshold_scores(y_true: np.ndarray, y_score: np.ndarray, metric: str) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    candidates = np.unique(y_score)
    if len(candidates) > 500:
        candidates = np.quantile(y_score, np.linspace(0.0, 1.0, 500))
        candidates = np.unique(candidates)
    best_threshold = 0.5
    best_value = -np.inf
    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(np.uint8)
        if metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, y_pred)
        else:
            value = f1_score(y_true, y_pred, zero_division=0)
        if value > best_value:
            best_value = float(value)
            best_threshold = float(threshold)
    return best_threshold, best_value


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    y_pred = (y_score >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positives = int(y_true.sum())
    negatives = int((y_true == 0).sum())
    prevalence = positives / len(y_true) if len(y_true) else 0.0
    all_negative = np.zeros_like(y_true)
    out: dict[str, float | int | None] = {
        "n": int(len(y_true)),
        "positives": positives,
        "negatives": negatives,
        "prevalence": float(prevalence),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "aupr": safe_aupr(y_true, y_score),
        "auroc": safe_auroc(y_true, y_score),
        "all_negative_accuracy": float(accuracy_score(y_true, all_negative)),
        "all_negative_balanced_accuracy": (
            float(balanced_accuracy_score(y_true, all_negative)) if len(np.unique(y_true)) == 2 else None
        ),
        "all_negative_f1": float(f1_score(y_true, all_negative, zero_division=0)),
    }
    out["aupr_lift_over_prevalence"] = (
        float(out["aupr"] / prevalence) if out["aupr"] is not None and prevalence > 0 else None
    )
    return out


def make_logreg(C: float, random_seed: int, max_iter: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(C),
            solver="liblinear",
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_seed),
        ),
    )


def score_model(model: Any, X_eval: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_eval)[:, 1]
    if hasattr(model, "decision_function"):
        values = model.decision_function(X_eval)
        return 1.0 / (1.0 + np.exp(-values))
    return model.predict(X_eval).astype(np.float32)


def choose_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    rank_metric: str,
    random_seed: int,
    max_iter: int,
) -> tuple[float, pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    best_value = -np.inf
    best_c = c_grid[0]
    best_scores = np.zeros(len(y_val), dtype=np.float32)
    for C in c_grid:
        model = make_logreg(C, random_seed=random_seed, max_iter=max_iter)
        model.fit(X_train, y_train)
        scores = score_model(model, X_val)
        threshold, _ = threshold_scores(y_val, scores, "f1")
        metrics = metrics_at_threshold(y_val, scores, threshold)
        rank_value = metrics.get(rank_metric)
        if rank_value is None:
            rank_value = -np.inf
        rows.append({"C": float(C), "rank_metric": rank_metric, "rank_value": rank_value, **metrics})
        if float(rank_value) > best_value:
            best_value = float(rank_value)
            best_c = float(C)
            best_scores = scores.astype(np.float32)
    return best_c, pd.DataFrame(rows), best_scores


def split_audit(frame: pd.DataFrame, rows: pd.DataFrame, embedding_dir: Path, skip_hash: bool) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "n_label_rows_valid_split": int(len(frame)),
        "n_embedded_rows": int(len(rows)),
        "n_unique_accessions": int(rows["accession"].nunique()),
        "duplicate_accession_rows": int(rows["accession"].duplicated().sum()),
        "split_counts": rows["split"].value_counts().to_dict(),
        "missing_split_rows": int(rows["split"].isna().sum()),
    }
    cluster_splits = rows.groupby("cluster_id")["split"].nunique()
    leaky_clusters = cluster_splits[cluster_splits > 1]
    audit["clusters_spanning_splits"] = int(len(leaky_clusters))
    audit["cluster_split_leakage"] = bool(len(leaky_clusters) > 0)
    audit["example_leaky_clusters"] = leaky_clusters.head(20).index.astype(str).tolist()

    if not skip_hash:
        split_by_accession = dict(zip(rows["accession"].astype(str), rows["split"].astype(str)))
        by_hash: dict[str, list[tuple[str, str]]] = {}
        for path in embedding_dir.glob("*.npy"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            accession = path.stem
            by_hash.setdefault(digest, []).append((accession, split_by_accession.get(accession, "missing")))
        duplicate_groups = [items for items in by_hash.values() if len(items) > 1]
        spanning_groups = [items for items in duplicate_groups if len({split for _, split in items}) > 1]
        audit["duplicate_embedding_groups"] = int(len(duplicate_groups))
        audit["duplicate_embedding_groups_spanning_splits"] = int(len(spanning_groups))
        audit["embedding_duplicate_split_leakage"] = bool(len(spanning_groups) > 0)
        audit["example_duplicate_embedding_split_groups"] = spanning_groups[:20]
    return audit


def maybe_import_matplotlib(no_plots: bool):
    if no_plots:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric_bars(plt, metrics: pd.DataFrame, split: str, out_path: Path) -> None:
    data = metrics[metrics["eval_split"].eq(split)].copy()
    if data.empty:
        return
    metric_names = ["aupr", "auroc", "f1", "precision", "recall", "balanced_accuracy"]
    fig, axes = plt.subplots(2, 3, figsize=(16, max(6, 0.5 * len(data) + 3)))
    for ax, metric in zip(axes.ravel(), metric_names):
        if metric not in data.columns:
            ax.axis("off")
            continue
        subset = data.sort_values(metric, na_position="first")
        ax.barh(subset["label"], subset[metric].astype(float))
        ax.set_title(metric)
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{split} metrics by label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prevalence(plt, label_audit: pd.DataFrame, out_path: Path) -> None:
    included = label_audit[label_audit["included"].astype(bool)].copy()
    if included.empty:
        return
    included["test_prevalence"] = included["test_positives"] / included["n_test"]
    included = included.sort_values("test_prevalence")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(included) + 1.5)))
    ax.barh(included["label"], included["test_prevalence"].astype(float))
    for idx, (_, row) in enumerate(included.iterrows()):
        ax.text(
            float(row["test_prevalence"]),
            idx,
            f"  +{int(row['test_positives'])}/n={int(row['n_test'])}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("test prevalence")
    ax.set_xlim(0, min(1.0, max(0.05, float(included["test_prevalence"].max()) * 1.25)))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_curves(plt, curve_rows: list[dict[str, Any]], split: str, curve_type: str, out_path: Path) -> None:
    data = [row for row in curve_rows if row["eval_split"] == split]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for row in data:
        y_true = row["y_true"]
        y_score = row["y_score"]
        label = row["label"]
        if curve_type == "pr":
            if int(y_true.sum()) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            score = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, label=f"{label} AP={score:.3f}")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_title(f"{split} precision-recall curves")
        else:
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            score = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, label=f"{label} AUROC={score:.3f}")
            ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.25)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_title(f"{split} ROC curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def aggregate_metrics(curve_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in curve_rows}):
        for split in ["val", "test"]:
            payload = [row for row in curve_rows if row["family"] == family and row["eval_split"] == split]
            if not payload:
                continue
            y_true = np.concatenate([row["y_true"] for row in payload])
            y_score = np.concatenate([row["y_score"] for row in payload])
            per_label = [row for row in metric_rows if row["family"] == family and row["eval_split"] == split]
            rows.append(
                {
                    "family": family,
                    "eval_split": split,
                    "n_labels": len(payload),
                    "micro_aupr": safe_aupr(y_true, y_score),
                    "micro_auroc": safe_auroc(y_true, y_score),
                    "macro_aupr": float(np.nanmean([row["aupr"] for row in per_label if row["aupr"] is not None])),
                    "macro_auroc": float(np.nanmean([row["auroc"] for row in per_label if row["auroc"] is not None])),
                    "macro_f1": float(np.nanmean([row["f1"] for row in per_label])),
                    "macro_precision": float(np.nanmean([row["precision"] for row in per_label])),
                    "macro_recall": float(np.nanmean([row["recall"] for row in per_label])),
                }
            )
    return pd.DataFrame(rows)


def fmt_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(fmt_value(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_audit_summary(
    run_dir: Path,
    audit: dict[str, Any],
    label_audit: pd.DataFrame,
    metrics: pd.DataFrame,
    aggregates: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    included = label_audit[label_audit["included"].astype(bool)].copy() if not label_audit.empty else pd.DataFrame()
    low_count = included[
        included["low_val_positives"].astype(bool) | included["low_test_positives"].astype(bool)
    ].copy() if not included.empty else pd.DataFrame()
    skipped = label_audit[~label_audit["included"].astype(bool)].copy() if not label_audit.empty else pd.DataFrame()
    test_metrics = metrics[metrics["eval_split"].eq("test")].copy() if not metrics.empty else pd.DataFrame()
    if not test_metrics.empty:
        test_metrics = test_metrics.sort_values(["family", "label"])

    lines = [
        "# Robust ESM-C Embedding Classifier Audit",
        "",
        "## Evaluation Design",
        "",
        "- Features: frozen mean-pooled ESM-C embeddings.",
        "- Binary label scope: all trainable rows.",
        f"- Mechanism label scope: `{config['mechanism_scope']}` rows.",
        "- Model family: L2-regularized logistic regression with class-balanced loss.",
        "- Model selection: choose `C` on validation by validation AUPR.",
        "- Threshold selection: choose threshold on validation by F1.",
        "- Final model: refit selected model on train+validation, then evaluate once on held-out test.",
        "",
        "## Split Audit",
        "",
        f"- Embedded rows: `{audit.get('n_embedded_rows')}`.",
        f"- Split counts: `{audit.get('split_counts')}`.",
        f"- Cluster split leakage: `{audit.get('cluster_split_leakage')}`.",
        f"- Duplicate accessions: `{audit.get('duplicate_accession_rows')}`.",
        f"- Duplicate embedding groups spanning splits: `{audit.get('duplicate_embedding_groups_spanning_splits', 'not checked')}`.",
        "",
        "## Aggregate Metrics",
        "",
        markdown_table(
            aggregates,
            ["family", "eval_split", "n_labels", "micro_aupr", "micro_auroc", "macro_aupr", "macro_auroc", "macro_f1"],
        ),
        "",
        "## Test Metrics By Label",
        "",
        markdown_table(
            test_metrics,
            [
                "label",
                "family",
                "positives",
                "negatives",
                "aupr",
                "auroc",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
            ],
        ),
        "",
        "## Low-Count Labels",
        "",
    ]
    if low_count.empty:
        lines.append("No included labels had fewer than 5 validation or test positives.")
    else:
        lines.append(
            "These labels are included for exploratory signal, but their threshold-based metrics are unstable."
        )
        lines.append("")
        lines.append(
            markdown_table(
                low_count,
                ["label", "val_positives", "test_positives", "train_positives", "low_val_positives", "low_test_positives"],
            )
        )

    lines.extend(["", "## Skipped Labels", ""])
    if skipped.empty:
        lines.append("No labels were skipped.")
    else:
        lines.append(markdown_table(skipped, ["label", "train_positives", "val_positives", "test_positives", "skip_reason"]))

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- High AUROC with very few positives can still be unstable; check AUPR, precision, recall, and the positive counts.",
            "- Perfect test scores are less suspicious after the split audit, but this is still a weak-label POC, not a curated benchmark.",
            "- The saved `.joblib` artifacts are one model per label. A multi-label prediction call should run all label models on the same embedding.",
            "",
            "## Key Files",
            "",
            "- `per_label_metrics.tsv`",
            "- `aggregate_metrics.tsv`",
            "- `label_audit.tsv`",
            "- `model_selection.tsv`",
            "- `model_index.tsv`",
            "- `models/*.joblib`",
            "- `plots/test_precision_recall_curves.png`",
            "- `plots/test_roc_curves.png`",
            "- `plots/test_metric_bars.png`",
        ]
    )
    (run_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    c_grid = parse_c_grid(args.c_grid)
    plt = maybe_import_matplotlib(args.no_plots)

    run_dir = Path(args.output_dir) / args.run_name
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    predictions_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    label_matrix = pd.read_csv(args.label_matrix, sep="\t", dtype=str).fillna("")
    splits = pd.read_csv(args.split_assignments, sep="\t", dtype=str).fillna("")
    split_columns = ["accession", "cluster_id", "representative_accession", "split"]
    frame = label_matrix.merge(splits[split_columns], on="accession", how="left")
    frame = frame[frame["is_trainable"].astype(str).str.lower().eq("true")]
    frame = frame[frame["split"].isin(["train", "val", "test"])].copy()

    log(f"loading embeddings from {args.embedding_dir}")
    X, rows = load_rows_with_embeddings(frame, Path(args.embedding_dir), args.embedding_glob)
    split_values = rows["split"].astype(str).to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    test_mask = split_values == "test"

    audit = split_audit(frame, rows, Path(args.embedding_dir), args.skip_embedding_hash_audit)
    (run_dir / "split_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    label_columns = selected_label_columns(rows, args.task)
    if not label_columns:
        raise SystemExit(f"No labels selected for task={args.task}.")

    label_audit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[pd.DataFrame] = []
    curve_rows: list[dict[str, Any]] = []
    model_index_rows: list[dict[str, Any]] = []

    for column in label_columns:
        label = slugify(column)
        family, scope_name = label_scope(column, args.mechanism_scope)
        scope_mask = scope_mask_for_column(rows, column, args.mechanism_scope)
        label_train_mask = train_mask & scope_mask
        label_val_mask = val_mask & scope_mask
        label_test_mask = test_mask & scope_mask
        label_train_val_mask = label_train_mask | label_val_mask

        y = binary_series(rows, column).to_numpy(dtype=np.uint8)
        y_train = y[label_train_mask]
        y_val = y[label_val_mask]
        y_test = y[label_test_mask]
        counts = {
            "label": label,
            "label_column": column,
            "family": family,
            "scope": scope_name,
            "n_train": int(label_train_mask.sum()),
            "n_val": int(label_val_mask.sum()),
            "n_test": int(label_test_mask.sum()),
            "train_positives": int(y_train.sum()),
            "val_positives": int(y_val.sum()),
            "test_positives": int(y_test.sum()),
            "train_negatives": int((y_train == 0).sum()),
            "val_negatives": int((y_val == 0).sum()),
            "test_negatives": int((y_test == 0).sum()),
            "low_val_positives": bool(int(y_val.sum()) < 5),
            "low_test_positives": bool(int(y_test.sum()) < 5),
        }
        skip_reason = ""
        if counts["n_train"] == 0 or counts["n_val"] == 0 or counts["n_test"] == 0:
            skip_reason = "empty split after scope filter"
        elif counts["train_positives"] < args.min_train_positives:
            skip_reason = f"train positives < {args.min_train_positives}"
        elif counts["train_negatives"] < 1:
            skip_reason = "train negatives < 1"
        elif counts["val_positives"] < args.min_val_positives:
            skip_reason = f"val positives < {args.min_val_positives}"
        elif counts["test_positives"] < args.min_test_positives:
            skip_reason = f"test positives < {args.min_test_positives}"
        elif counts["val_negatives"] < 1 or counts["test_negatives"] < 1:
            skip_reason = "val/test negatives < 1"
        counts["included"] = not bool(skip_reason)
        counts["skip_reason"] = skip_reason or "none"
        label_audit_rows.append(counts)
        if skip_reason:
            log(f"skip label={label}: {skip_reason}")
            continue
        if args.dry_run:
            log(f"dry-run include label={label} family={family} scope={scope_name}")
            continue

        log(f"label={label} family={family} scope={scope_name} model selection")
        best_c, selection, val_scores = choose_model(
            X[label_train_mask],
            y_train,
            X[label_val_mask],
            y_val,
            c_grid,
            args.rank_metric,
            args.random_seed,
            args.max_iter,
        )
        selection.insert(0, "scope", scope_name)
        selection.insert(0, "family", family)
        selection.insert(0, "label", label)
        selection_rows.append(selection)

        threshold, threshold_value = threshold_scores(y_val, val_scores, args.threshold_metric)
        val_metrics = metrics_at_threshold(y_val, val_scores, threshold)
        val_row = {
            "label": label,
            "label_column": column,
            "family": family,
            "scope": scope_name,
            "eval_split": "val",
            "selected_C": best_c,
            "threshold_metric": args.threshold_metric,
            "threshold_metric_value": threshold_value,
            **val_metrics,
        }
        metric_rows.append(val_row)
        curve_rows.append({"label": label, "family": family, "eval_split": "val", "y_true": y_val, "y_score": val_scores})

        final_model = make_logreg(best_c, random_seed=args.random_seed, max_iter=args.max_iter)
        final_model.fit(X[label_train_val_mask], y[label_train_val_mask])
        test_scores = score_model(final_model, X[label_test_mask]).astype(np.float32)
        test_metrics = metrics_at_threshold(y_test, test_scores, threshold)
        test_row = {
            "label": label,
            "label_column": column,
            "family": family,
            "scope": scope_name,
            "eval_split": "test",
            "selected_C": best_c,
            "threshold_metric": args.threshold_metric,
            "threshold_metric_value": threshold_value,
            **test_metrics,
        }
        metric_rows.append(test_row)
        curve_rows.append(
            {"label": label, "family": family, "eval_split": "test", "y_true": y_test, "y_score": test_scores}
        )

        for split_name, mask, y_true, scores in [
            ("val", label_val_mask, y_val, val_scores),
            ("test", label_test_mask, y_test, test_scores),
        ]:
            pred = rows.loc[mask, ["protein_uid", "accession", "cluster_id", "representative_accession", "split"]].copy()
            pred["label"] = label
            pred["true"] = y_true
            pred["score"] = scores
            pred["pred"] = (scores >= threshold).astype(np.uint8)
            pred["threshold"] = threshold
            pred.to_csv(predictions_dir / f"{label}_{split_name}_predictions.tsv", sep="\t", index=False)

        model_path = models_dir / f"{label}.joblib"
        joblib.dump(
            {
                "label": label,
                "label_column": column,
                "family": family,
                "scope": scope_name,
                "selected_C": best_c,
                "threshold": threshold,
                "threshold_metric": args.threshold_metric,
                "pipeline": final_model,
                "trained_on": "train+val",
                "embedding_dir": args.embedding_dir,
                "n_features": int(X.shape[1]),
                "test_metrics": test_metrics,
            },
            model_path,
        )
        model_index_rows.append(
            {
                "label": label,
                "family": family,
                "scope": scope_name,
                "model_path": str(model_path),
                "selected_C": best_c,
                "threshold": threshold,
                "test_aupr": test_metrics["aupr"],
                "test_auroc": test_metrics["auroc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
            }
        )

    label_audit = pd.DataFrame(label_audit_rows)
    label_audit.to_csv(run_dir / "label_audit.tsv", sep="\t", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "per_label_metrics.tsv", sep="\t", index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_csv(run_dir / "model_selection.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "model_selection.tsv", sep="\t", index=False)
    pd.DataFrame(model_index_rows).to_csv(run_dir / "model_index.tsv", sep="\t", index=False)
    aggregates = aggregate_metrics(curve_rows, metric_rows) if metric_rows else pd.DataFrame()
    aggregates.to_csv(run_dir / "aggregate_metrics.tsv", sep="\t", index=False)

    config = {
        "label_matrix": args.label_matrix,
        "split_assignments": args.split_assignments,
        "embedding_dir": args.embedding_dir,
        "embedding_glob": args.embedding_glob,
        "task": args.task,
        "mechanism_scope": args.mechanism_scope,
        "min_train_positives": args.min_train_positives,
        "min_val_positives": args.min_val_positives,
        "min_test_positives": args.min_test_positives,
        "c_grid": c_grid,
        "rank_metric": args.rank_metric,
        "threshold_metric": args.threshold_metric,
        "random_seed": args.random_seed,
        "n_samples_with_embeddings": int(len(rows)),
        "n_features": int(X.shape[1]),
        "dry_run": bool(args.dry_run),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        log(f"dry run complete; wrote audits to {run_dir}")
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    if plt is not None and not metrics.empty:
        plot_metric_bars(plt, metrics, "val", plots_dir / "val_metric_bars.png")
        plot_metric_bars(plt, metrics, "test", plots_dir / "test_metric_bars.png")
        plot_prevalence(plt, label_audit, plots_dir / "test_label_prevalence.png")
        for split in ["val", "test"]:
            plot_curves(plt, curve_rows, split, "pr", plots_dir / f"{split}_precision_recall_curves.png")
            plot_curves(plt, curve_rows, split, "roc", plots_dir / f"{split}_roc_curves.png")

    write_audit_summary(run_dir, audit, label_audit, metrics, aggregates, config)

    payload = {
        "run_dir": str(run_dir),
        "included_labels": int(label_audit["included"].sum()) if not label_audit.empty else 0,
        "skipped_labels": int((~label_audit["included"]).sum()) if not label_audit.empty else 0,
        "split_audit": audit,
    }
    if not aggregates.empty:
        payload["aggregate_metrics"] = aggregates.to_dict(orient="records")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote robust model outputs to {run_dir}")


if __name__ == "__main__":
    main()
