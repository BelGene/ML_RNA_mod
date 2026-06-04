#!/usr/bin/env python3
"""Train weak-label tRNA-modifier classifiers from precomputed embeddings."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
DEFAULT_LABEL_MATRIX = "data/processed/poc_weak/weak_trna_mod_label_matrix.tsv"
DEFAULT_SPLIT_ASSIGNMENTS = "data/processed/poc_weak/splits/mmseqs50/split_assignments.tsv"
DEFAULT_OUTPUT_DIR = "data/processed/poc_weak/ml_runs"
DEFAULT_RUN_NAME = "weak_trna_logreg"
DEFAULT_MIN_POSITIVE_COUNT = 20
DEFAULT_C = 0.1
DEFAULT_RANDOM_SEED = 42


def find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from script path.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
os.chdir(REPO_ROOT)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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
    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
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


def choose_label_columns(frame: pd.DataFrame, task: str, min_positive_count: int) -> list[str]:
    columns: list[str] = []
    if task in {"binary", "all"}:
        columns.append("target__trna_modifier")
    if task in {"mechanism", "all"}:
        mechanism_columns = [column for column in frame.columns if column.startswith("mechanism__")]
        for column in mechanism_columns:
            positives = int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
            if positives >= min_positive_count:
                columns.append(column)
    if not columns:
        raise SystemExit(f"No labels selected for task={task} with min_positive_count={min_positive_count}.")
    return columns


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


def safe_label_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_pred = (y_score >= threshold).astype(np.uint8)
    metrics: dict[str, float | None] = {
        "positives": int(y_true.sum()),
        "negatives": int((y_true == 0).sum()),
        "aupr": float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else None,
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else None,
    }
    metrics["auroc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    return metrics


def fit_label_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    C: float,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    unique = np.unique(y_train)
    if len(unique) < 2:
        constant = float(unique[0])
        return np.full(len(x_eval), constant, dtype=np.float32), {"constant_score": constant}
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_seed,
    )
    clf.fit(x_train_scaled, y_train)
    return clf.predict_proba(x_eval_scaled)[:, 1], {"scaler": scaler, "model": clf}


def main() -> None:
    args = parse_args()
    label_matrix = pd.read_csv(args.label_matrix, sep="\t", dtype=str).fillna("")
    splits = pd.read_csv(args.split_assignments, sep="\t", dtype=str).fillna("")
    split_columns = ["accession", "cluster_id", "representative_accession", "split"]
    frame = label_matrix.merge(splits[split_columns], on="accession", how="left")
    frame = frame[frame["is_trainable"].astype(str).str.lower().eq("true")]
    frame = frame[frame["split"].isin(["train", "val", "test"])]
    label_columns = choose_label_columns(frame, args.task, args.min_positive_count)

    X, rows = load_rows_with_embeddings(frame, Path(args.embedding_dir), args.embedding_glob)
    y = rows[label_columns].astype(int).to_numpy(dtype=np.uint8)
    split_values = rows["split"].astype(str).to_numpy()
    train_mask = split_values == "train"
    if train_mask.sum() == 0:
        raise SystemExit("No embedded train rows found.")

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = rows[["protein_uid", "accession", "cluster_id", "representative_accession", "split"]].copy()
    metrics: dict[str, Any] = {
        "task": args.task,
        "n_samples_with_embeddings": int(len(rows)),
        "n_train": int(train_mask.sum()),
        "n_val": int((split_values == "val").sum()),
        "n_test": int((split_values == "test").sum()),
        "n_features": int(X.shape[1]),
        "labels": [column.replace("target__", "").replace("mechanism__", "") for column in label_columns],
        "per_label": {},
    }
    models: dict[str, Any] = {}
    for label_idx, column in enumerate(label_columns):
        label_name = column.replace("target__", "").replace("mechanism__", "")
        y_train = y[train_mask, label_idx]
        for split in ["val", "test"]:
            eval_mask = split_values == split
            if eval_mask.sum() == 0:
                continue
            scores, fitted = fit_label_model(X[train_mask], y_train, X[eval_mask], C=args.C, random_seed=args.random_seed)
            predictions.loc[eval_mask, f"true__{label_name}"] = y[eval_mask, label_idx]
            predictions.loc[eval_mask, f"score__{label_name}"] = scores
            metrics["per_label"].setdefault(label_name, {})[split] = safe_label_metrics(y[eval_mask, label_idx], scores)
            if split == "test":
                models[label_name] = fitted

    predictions.to_csv(run_dir / "split_predictions.tsv", sep="\t", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.pickle_final_models:
        with (run_dir / "models.pkl").open("wb") as handle:
            pickle.dump(models, handle)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote outputs to {run_dir}")


if __name__ == "__main__":
    main()
