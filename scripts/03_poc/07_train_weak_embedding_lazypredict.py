#!/usr/bin/env python3
"""Screen weak-label tRNA-modifier classifiers with LazyPredict.

This script uses the existing MMseqs cluster split. It never creates random
train/test splits, because homologous proteins should stay on the same side of
the evaluation boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
DEFAULT_LABEL_MATRIX = "data/processed/poc_weak/weak_trna_mod_label_matrix.tsv"
DEFAULT_SPLIT_ASSIGNMENTS = "data/processed/poc_weak/splits/mmseqs50/split_assignments.tsv"
DEFAULT_OUTPUT_DIR = "data/processed/poc_weak/ml_runs"
DEFAULT_RUN_NAME = "esmc6b_lazypredict"
DEFAULT_MIN_POSITIVE_COUNT = 20
DEFAULT_MIN_EVAL_POSITIVE_COUNT = 1
DEFAULT_MECHANISM_SCOPE = "mechanism_labeled"
DEFAULT_RANDOM_SEED = 42
DEFAULT_TOP_N_PLOTS = 20
DEFAULT_CONFUSION_TOP_N = 6


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
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--embedding-glob", default="*")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--task", choices=("binary", "mechanism", "all"), default="binary")
    parser.add_argument("--min-positive-count", type=int, default=DEFAULT_MIN_POSITIVE_COUNT)
    parser.add_argument("--min-val-positive-count", type=int, default=DEFAULT_MIN_EVAL_POSITIVE_COUNT)
    parser.add_argument("--min-test-positive-count", type=int, default=DEFAULT_MIN_EVAL_POSITIVE_COUNT)
    parser.add_argument(
        "--mechanism-scope",
        choices=("all_trainable", "trna_modifiers", "mechanism_labeled"),
        default=DEFAULT_MECHANISM_SCOPE,
        help=(
            "Rows used for mechanism one-vs-rest classifiers. "
            "mechanism_labeled avoids treating unrelated non-tRNA proteins as mechanism negatives."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--top-n-plots", type=int, default=DEFAULT_TOP_N_PLOTS)
    parser.add_argument("--confusion-top-n", type=int, default=DEFAULT_CONFUSION_TOP_N)
    parser.add_argument(
        "--rank-metric",
        default="Balanced Accuracy",
        help="LazyPredict metric used for sorting plots and selecting best models.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def strip_known_suffix(path: Path) -> str:
    stem = path.stem
    return re.sub(r"_hidden_layer_steps\d+$", "", stem)


def slugify(value: str) -> str:
    value = value.replace("target__", "").replace("mechanism__", "")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "label"


def build_embedding_index(embedding_dir: Path, pattern: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(embedding_dir.glob(pattern)):
        if path.suffix.lower() not in {".npy", ".pt"}:
            continue
        index[path.stem] = path
        index[strip_known_suffix(path)] = path
    return index


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


def load_embedding(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".pt":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local env.
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


def choose_label_columns(frame: pd.DataFrame, task: str, min_positive_count: int, mechanism_scope: str) -> list[str]:
    columns: list[str] = []
    if task in {"binary", "all"}:
        columns.append("target__trna_modifier")
    if task in {"mechanism", "all"}:
        scope_mask = mechanism_scope_mask(frame, mechanism_scope)
        train_frame = frame[scope_mask & frame["split"].astype(str).eq("train")]
        for column in [item for item in frame.columns if is_mechanism_column(item)]:
            positives = int(pd.to_numeric(train_frame[column], errors="coerce").fillna(0).sum())
            if positives >= min_positive_count:
                columns.append(column)
    if not columns:
        raise SystemExit(f"No labels selected for task={task} with min_positive_count={min_positive_count}.")
    return columns


def import_lazy_classifier():
    try:
        from lazypredict.Supervised import LazyClassifier
    except ImportError as exc:
        raise SystemExit(
            "LazyPredict is not installed. Install the ML extras in the py311 environment first:\n"
            "  /ocean/projects/bio250095p/azimic/venvs/trna-esmc-py311/bin/python -m pip install lazypredict matplotlib"
        ) from exc
    return LazyClassifier


def maybe_import_matplotlib(no_plots: bool):
    if no_plots:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Install it or rerun with --no-plots:\n"
            "  /ocean/projects/bio250095p/azimic/venvs/trna-esmc-py311/bin/python -m pip install matplotlib"
        ) from exc
    return plt


def normalize_lazy_table(table: pd.DataFrame, label: str, eval_split: str, n_train: int, n_eval: int) -> pd.DataFrame:
    frame = table.copy()
    frame.index.name = "model"
    frame = frame.reset_index()
    frame.insert(0, "eval_split", eval_split)
    frame.insert(0, "label", label)
    frame["n_train"] = int(n_train)
    frame["n_eval"] = int(n_eval)
    return frame


def prediction_metrics(y_true: np.ndarray, predictions: pd.DataFrame, label: str, eval_split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in predictions.columns:
        y_pred = predictions[model_name].to_numpy()
        rows.append(
            {
                "label": label,
                "eval_split": eval_split,
                "model": model_name,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "positives": int(y_true.sum()),
                "negatives": int((y_true == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def all_negative_baseline(y_true: np.ndarray) -> dict[str, float]:
    y_pred = np.zeros_like(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def write_prediction_table(
    out_path: Path,
    rows: pd.DataFrame,
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    label_name: str,
) -> None:
    metadata_columns = ["protein_uid", "accession", "cluster_id", "representative_accession", "split"]
    out = rows[metadata_columns].reset_index(drop=True).copy()
    out[f"true__{label_name}"] = y_true
    for model_name in predictions.columns:
        out[f"pred__{slugify(model_name)}"] = predictions[model_name].to_numpy()
    out.to_csv(out_path, sep="\t", index=False)


def run_lazy_split(
    LazyClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            custom_metric=None,
            predictions=True,
            random_state=random_seed,
        )
    except TypeError:
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
    models, predictions = clf.fit(X_train, X_eval, y_train, y_eval)
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    return models, predictions


def sort_by_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric in frame.columns:
        return frame.sort_values(metric, ascending=False)
    for fallback in ["Balanced Accuracy", "ROC AUC", "F1 Score", "accuracy", "balanced_accuracy", "f1"]:
        if fallback in frame.columns:
            return frame.sort_values(fallback, ascending=False)
    return frame


def plot_metric_bar(plt, frame: pd.DataFrame, metric: str, title: str, out_path: Path, top_n: int) -> None:
    if metric not in frame.columns or frame.empty:
        return
    data = sort_by_metric(frame, metric).head(top_n).iloc[::-1]
    fig_height = max(4.0, 0.35 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(data["model"].astype(str), data[metric].astype(float))
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_val_test_scatter(plt, joined: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    val_col = f"val__{metric}"
    test_col = f"test__{metric}"
    if val_col not in joined.columns or test_col not in joined.columns or joined.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(joined[val_col].astype(float), joined[test_col].astype(float), alpha=0.75)
    ax.set_xlabel(f"validation {metric}")
    ax.set_ylabel(f"test {metric}")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1, alpha=0.4)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_grid(
    plt,
    y_true: np.ndarray,
    predictions: pd.DataFrame,
    model_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    model_names = [name for name in model_names if name in predictions.columns]
    if not model_names:
        return
    n_cols = min(3, len(model_names))
    n_rows = int(np.ceil(len(model_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows), squeeze=False)
    for ax, model_name in zip(axes.ravel(), model_names):
        cm = confusion_matrix(y_true, predictions[model_name].to_numpy(), labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(model_name)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(model_names) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_best_by_label(plt, best: pd.DataFrame, metric: str, out_path: Path) -> None:
    metric_col = f"test__{metric}"
    if best.empty or metric_col not in best.columns:
        return
    data = best.sort_values(metric_col).copy()
    fig_height = max(4.0, 0.45 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(data["label"], data[metric_col].astype(float))
    for idx, (_, row) in enumerate(data.iterrows()):
        ax.text(float(row[metric_col]), idx, f"  {row['model']}", va="center", fontsize=8)
    ax.set_xlabel(f"test {metric}")
    ax.set_title(f"Best validation-ranked model per label: test {metric}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def prefixed_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = ["model"]
    metric_columns = [
        column for column in frame.columns if column not in {"label", "eval_split", "model", "n_train", "n_eval"}
    ]
    out = frame[keep + metric_columns].copy()
    return out.rename(columns={column: f"{prefix}__{column}" for column in metric_columns})


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)
    LazyClassifier = import_lazy_classifier()
    plt = maybe_import_matplotlib(args.no_plots)

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
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
    label_columns = choose_label_columns(rows, args.task, args.min_positive_count, args.mechanism_scope)
    split_values = rows["split"].astype(str).to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    test_mask = split_values == "test"
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise SystemExit("Need non-empty train, val, and test splits.")

    all_lazy_tables: list[pd.DataFrame] = []
    all_prediction_metrics: list[pd.DataFrame] = []
    best_rows: list[pd.Series] = []
    label_summary: list[dict[str, Any]] = []

    for column in label_columns:
        label_name = slugify(column)
        if is_mechanism_column(column):
            scope_name = args.mechanism_scope
            scope_mask = mechanism_scope_mask(rows, args.mechanism_scope).to_numpy(dtype=bool)
        else:
            scope_name = "all_trainable"
            scope_mask = np.ones(len(rows), dtype=bool)
        label_train_mask = train_mask & scope_mask
        label_val_mask = val_mask & scope_mask
        label_test_mask = test_mask & scope_mask
        label_train_val_mask = label_train_mask | label_val_mask
        if label_train_mask.sum() == 0 or label_val_mask.sum() == 0 or label_test_mask.sum() == 0:
            log(f"skip label={label_name}; scope={scope_name} has an empty train/val/test split")
            continue

        label_dir = run_dir / label_name
        label_plots_dir = label_dir / "plots"
        label_dir.mkdir(parents=True, exist_ok=True)
        if plt is not None:
            label_plots_dir.mkdir(parents=True, exist_ok=True)

        y = rows[column].astype(int).to_numpy(dtype=np.uint8)
        y_train = y[label_train_mask]
        y_val = y[label_val_mask]
        y_train_val = y[label_train_val_mask]
        y_test = y[label_test_mask]
        if len(np.unique(y_train)) < 2:
            log(f"skip label={label_name}; scope={scope_name} train split has one class only")
            continue
        if int(y_val.sum()) < args.min_val_positive_count:
            log(
                f"skip label={label_name}; scope={scope_name} val positives={int(y_val.sum())} "
                f"< min_val_positive_count={args.min_val_positive_count}"
            )
            continue
        if int(y_test.sum()) < args.min_test_positive_count:
            log(
                f"skip label={label_name}; scope={scope_name} test positives={int(y_test.sum())} "
                f"< min_test_positive_count={args.min_test_positive_count}"
            )
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[label_train_mask])
        X_val = scaler.transform(X[label_val_mask])
        train_val_scaler = StandardScaler()
        X_train_val = train_val_scaler.fit_transform(X[label_train_val_mask])
        X_test = train_val_scaler.transform(X[label_test_mask])

        val_baseline = all_negative_baseline(y_val)
        test_baseline = all_negative_baseline(y_test)

        label_summary.append(
            {
                "label": label_name,
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
                "val_all_negative_accuracy": val_baseline["accuracy"],
                "val_all_negative_balanced_accuracy": val_baseline["balanced_accuracy"],
                "test_all_negative_accuracy": test_baseline["accuracy"],
                "test_all_negative_balanced_accuracy": test_baseline["balanced_accuracy"],
            }
        )

        log(f"label={label_name} scope={scope_name} validation screen train->{int(label_val_mask.sum())} val")
        val_models, val_predictions = run_lazy_split(
            LazyClassifier, X_train, y_train, X_val, y_val, random_seed=args.random_seed
        )
        val_table = normalize_lazy_table(
            val_models, label_name, "val", int(label_train_mask.sum()), int(label_val_mask.sum())
        )
        val_pred_metrics = prediction_metrics(y_val, val_predictions, label_name, "val")
        val_table.to_csv(label_dir / "val_lazypredict_models.tsv", sep="\t", index=False)
        val_pred_metrics.to_csv(label_dir / "val_prediction_metrics.tsv", sep="\t", index=False)
        write_prediction_table(label_dir / "val_predictions.tsv", rows[label_val_mask], y_val, val_predictions, label_name)

        log(f"label={label_name} scope={scope_name} test screen train+val->{int(label_test_mask.sum())} test")
        test_models, test_predictions = run_lazy_split(
            LazyClassifier, X_train_val, y_train_val, X_test, y_test, random_seed=args.random_seed
        )
        test_table = normalize_lazy_table(
            test_models, label_name, "test", int(label_train_val_mask.sum()), int(label_test_mask.sum())
        )
        test_pred_metrics = prediction_metrics(y_test, test_predictions, label_name, "test")
        test_table.to_csv(label_dir / "test_lazypredict_models.tsv", sep="\t", index=False)
        test_pred_metrics.to_csv(label_dir / "test_prediction_metrics.tsv", sep="\t", index=False)
        write_prediction_table(label_dir / "test_predictions.tsv", rows[label_test_mask], y_test, test_predictions, label_name)

        joined = prefixed_metrics(val_table, "val").merge(prefixed_metrics(test_table, "test"), on="model", how="outer")
        joined.insert(0, "label", label_name)
        joined = sort_by_metric(joined, f"val__{args.rank_metric}" if f"val__{args.rank_metric}" in joined else args.rank_metric)
        joined.to_csv(label_dir / "val_test_model_comparison.tsv", sep="\t", index=False)
        if not joined.empty:
            best_rows.append(joined.iloc[0])

        if plt is not None:
            plot_metric_bar(
                plt,
                val_table,
                args.rank_metric,
                f"{label_name}: validation {args.rank_metric}",
                label_plots_dir / f"val_top_{slugify(args.rank_metric)}.png",
                args.top_n_plots,
            )
            plot_metric_bar(
                plt,
                test_table,
                args.rank_metric,
                f"{label_name}: test {args.rank_metric}",
                label_plots_dir / f"test_top_{slugify(args.rank_metric)}.png",
                args.top_n_plots,
            )
            plot_val_test_scatter(
                plt,
                joined,
                args.rank_metric,
                f"{label_name}: validation vs test {args.rank_metric}",
                label_plots_dir / f"val_vs_test_{slugify(args.rank_metric)}.png",
            )
            top_models = sort_by_metric(test_table, args.rank_metric)["model"].astype(str).head(args.confusion_top_n).tolist()
            plot_confusion_grid(
                plt,
                y_test,
                test_predictions,
                top_models,
                f"{label_name}: test confusion matrices",
                label_plots_dir / "test_confusion_top_models.png",
            )

        all_lazy_tables.extend([val_table, test_table])
        all_prediction_metrics.extend([val_pred_metrics, test_pred_metrics])

    all_lazy = pd.concat(all_lazy_tables, ignore_index=True) if all_lazy_tables else pd.DataFrame()
    all_pred_metrics = (
        pd.concat(all_prediction_metrics, ignore_index=True) if all_prediction_metrics else pd.DataFrame()
    )
    label_summary_frame = pd.DataFrame(label_summary)
    best_frame = pd.DataFrame(best_rows)

    all_lazy.to_csv(run_dir / "all_lazypredict_model_results.tsv", sep="\t", index=False)
    all_pred_metrics.to_csv(run_dir / "all_prediction_metrics.tsv", sep="\t", index=False)
    label_summary_frame.to_csv(run_dir / "label_summary.tsv", sep="\t", index=False)
    best_frame.to_csv(run_dir / "best_validation_ranked_models.tsv", sep="\t", index=False)

    completed_labels = label_summary_frame["label"].astype(str).tolist() if not label_summary_frame.empty else []
    config = {
        "label_matrix": args.label_matrix,
        "split_assignments": args.split_assignments,
        "embedding_dir": args.embedding_dir,
        "embedding_glob": args.embedding_glob,
        "task": args.task,
        "mechanism_scope": args.mechanism_scope,
        "min_positive_count": args.min_positive_count,
        "min_val_positive_count": args.min_val_positive_count,
        "min_test_positive_count": args.min_test_positive_count,
        "random_seed": args.random_seed,
        "rank_metric": args.rank_metric,
        "n_samples_with_embeddings": int(len(rows)),
        "n_features": int(X.shape[1]),
        "base_n_train": int(train_mask.sum()),
        "base_n_val": int(val_mask.sum()),
        "base_n_test": int(test_mask.sum()),
        "selected_labels": [slugify(column) for column in label_columns],
        "completed_labels": completed_labels,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if plt is not None and not best_frame.empty:
        plot_best_by_label(plt, best_frame, args.rank_metric, plots_dir / f"best_by_label_{slugify(args.rank_metric)}.png")

    print(json.dumps(config, indent=2, sort_keys=True))
    print(f"wrote LazyPredict outputs to {run_dir}")


if __name__ == "__main__":
    main()
