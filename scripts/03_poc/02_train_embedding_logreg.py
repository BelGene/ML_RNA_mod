#!/usr/bin/env python3
"""Train a simple multi-label logistic-regression POC from protein embeddings.

Expected inputs:
  - a label matrix from 01_build_modomics_trna_dataset.py
  - one embedding file per accession under --embedding-dir

Embedding filenames may be:
  ACCESSION.npy
  ACCESSION.pt
  ACCESSION_hidden_layer_steps10.pt
  ACCESSION_hidden_layer_steps10.npy

Per-residue tensors are mean-pooled into one vector per protein, matching the
simple frozen-ESM transfer-learning baseline used in the reference project.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler


DEFAULT_LABEL_MATRIX = "data/processed/poc/modomics_trna_label_matrix.tsv"
DEFAULT_OUTPUT_DIR = "data/processed/poc/ml_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-matrix", default=DEFAULT_LABEL_MATRIX)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="logreg")
    parser.add_argument("--embedding-glob", default="*")
    parser.add_argument("--min-positive-count", type=int, default=2)
    parser.add_argument("--cv-mode", choices=("kfold", "loocv"), default="kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--pickle-final-models", action="store_true")
    return parser.parse_args()


def strip_known_suffix(path: Path) -> str:
    stem = path.stem
    return re.sub(r"_hidden_layer_steps\d+$", "", stem)


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
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise RuntimeError("Loading .pt embeddings requires torch.") from exc
        tensor = torch.load(path, map_location="cpu")
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
        arr = np.asarray(tensor, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding file extension: {path}")

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


def load_feature_table(label_matrix: pd.DataFrame, embedding_dir: Path, pattern: str) -> tuple[np.ndarray, pd.DataFrame]:
    index = build_embedding_index(embedding_dir, pattern)
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for row in label_matrix.to_dict(orient="records"):
        accession = str(row["accession"])
        path = index.get(accession)
        if path is None:
            continue
        vectors.append(load_embedding(path))
        rows.append(row)
    if not vectors:
        raise RuntimeError(f"No embeddings matched accessions in {embedding_dir}")
    return np.stack(vectors, axis=0), pd.DataFrame(rows)


def build_cv_splits(n_samples: int, mode: str, n_splits: int, random_seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples < 2:
        raise ValueError("At least two embedded proteins are required.")
    if mode == "loocv":
        splitter = LeaveOneOut()
    else:
        effective_splits = min(int(n_splits), n_samples)
        if effective_splits < 2:
            raise ValueError("At least two folds are required.")
        splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_seed)
    dummy = np.zeros((n_samples, 1), dtype=np.uint8)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(dummy)]


def safe_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    metrics["micro_aupr"] = float(average_precision_score(y_true.ravel(), y_score.ravel()))
    per_label_aupr = []
    per_label_auroc = []
    for idx in range(y_true.shape[1]):
        if y_true[:, idx].sum() > 0:
            per_label_aupr.append(float(average_precision_score(y_true[:, idx], y_score[:, idx])))
        if len(np.unique(y_true[:, idx])) == 2:
            per_label_auroc.append(float(roc_auc_score(y_true[:, idx], y_score[:, idx])))
    metrics["macro_aupr"] = float(np.mean(per_label_aupr)) if per_label_aupr else None
    metrics["macro_auroc"] = float(np.mean(per_label_auroc)) if per_label_auroc else None
    if len(np.unique(y_true.ravel())) == 2:
        metrics["micro_auroc"] = float(roc_auc_score(y_true.ravel(), y_score.ravel()))
    else:
        metrics["micro_auroc"] = None
    return metrics


def fit_predict_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    C: float,
    random_seed: int,
) -> np.ndarray:
    scores = np.zeros_like(y, dtype=np.float32)
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        for label_idx in range(y.shape[1]):
            label_train = y_train[:, label_idx]
            unique = np.unique(label_train)
            if len(unique) < 2:
                scores[test_idx, label_idx] = float(unique[0])
                continue
            clf = LogisticRegression(
                C=C,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=random_seed,
            )
            clf.fit(X_train, label_train)
            scores[test_idx, label_idx] = clf.predict_proba(X_test)[:, 1]
    return scores


def fit_final_models(X: np.ndarray, y: np.ndarray, label_names: list[str], C: float, random_seed: int) -> dict[str, Any]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models: dict[str, Any] = {}
    for label_idx, label_name in enumerate(label_names):
        label_y = y[:, label_idx]
        unique = np.unique(label_y)
        if len(unique) < 2:
            models[label_name] = {"constant_score": float(unique[0])}
            continue
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_seed,
        )
        clf.fit(X_scaled, label_y)
        models[label_name] = clf
    return {"scaler": scaler, "label_names": label_names, "models": models}


def main() -> None:
    args = parse_args()
    label_matrix = pd.read_csv(args.label_matrix, sep="\t", dtype=str).fillna("")
    label_columns = [column for column in label_matrix.columns if column.startswith("enzyme_type__")]
    if not label_columns:
        raise SystemExit("No enzyme_type__ label columns found in label matrix.")

    X, rows = load_feature_table(label_matrix, Path(args.embedding_dir), args.embedding_glob)
    y_all = rows[label_columns].astype(int).to_numpy(dtype=np.uint8)
    positive_counts = y_all.sum(axis=0)
    keep_mask = positive_counts >= int(args.min_positive_count)
    kept_label_columns = [column for column, keep in zip(label_columns, keep_mask) if keep]
    y = y_all[:, keep_mask]
    if y.shape[1] == 0:
        raise SystemExit(f"No labels have at least {args.min_positive_count} positive examples.")

    splits = build_cv_splits(len(rows), args.cv_mode, args.n_splits, args.random_seed)
    scores = fit_predict_cv(X, y, splits, C=args.C, random_seed=args.random_seed)
    metrics = safe_metrics(y, scores)

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "n_samples_with_embeddings": int(len(rows)),
        "n_features": int(X.shape[1]),
        "labels": [column.removeprefix("enzyme_type__") for column in kept_label_columns],
        "positive_counts": {
            column.removeprefix("enzyme_type__"): int(count)
            for column, count, keep in zip(label_columns, positive_counts.tolist(), keep_mask)
            if keep
        },
        "cv_mode": args.cv_mode,
        "n_splits": len(splits),
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    predictions = rows[["protein_uid", "accession", "enzyme_type_labels"]].copy()
    for idx, column in enumerate(kept_label_columns):
        label = column.removeprefix("enzyme_type__")
        predictions[f"true__{label}"] = y[:, idx]
        predictions[f"score__{label}"] = scores[:, idx]
    predictions.to_csv(run_dir / "cv_predictions.tsv", sep="\t", index=False)

    if args.pickle_final_models:
        final = fit_final_models(X, y, kept_label_columns, C=args.C, random_seed=args.random_seed)
        with (run_dir / "final_models.pkl").open("wb") as handle:
            pickle.dump(final, handle)

    print(json.dumps(metrics_payload, indent=2, sort_keys=True))
    print(f"wrote outputs to {run_dir}")


if __name__ == "__main__":
    main()
