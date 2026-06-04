#!/usr/bin/env python3
"""Cluster weak-POC protein sequences and create train/validation/test splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
DEFAULT_FASTA = "data/processed/poc_weak/weak_trna_mod_sequences.faa"
DEFAULT_LABEL_MATRIX = "data/processed/poc_weak/weak_trna_mod_label_matrix.tsv"
DEFAULT_OUTPUT_DIR = "data/processed/poc_weak/splits/mmseqs50"
DEFAULT_MIN_SEQ_ID = 0.50
DEFAULT_COVERAGE = 0.80
DEFAULT_COV_MODE = 0
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_RANDOM_SEED = 42
DEFAULT_MMSEQS_VERBOSITY = 1


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
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--label-matrix", default=DEFAULT_LABEL_MATRIX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--min-seq-id", type=float, default=DEFAULT_MIN_SEQ_ID)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--cov-mode", type=int, default=DEFAULT_COV_MODE)
    parser.add_argument("--mmseqs-verbosity", type=int, default=DEFAULT_MMSEQS_VERBOSITY)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--force", action="store_true", help="Delete and rebuild an existing output directory.")
    parser.add_argument("--keep-mmseqs-workdir", action="store_true")
    return parser.parse_args()


def accession_from_fasta_id(fasta_id: str) -> str:
    return fasta_id.split("|", 1)[0]


def validate_fractions(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0; got {total:.4f}")
    if min(train, val, test) <= 0:
        raise ValueError("Split fractions must all be positive.")


def require_mmseqs(binary: str) -> str:
    resolved = shutil.which(binary) if not Path(binary).exists() else binary
    if not resolved and binary == "mmseqs":
        local_binary = REPO_ROOT / ".local_tools/mmseqs/bin/mmseqs"
        if local_binary.exists():
            resolved = str(local_binary)
    if not resolved:
        raise SystemExit(
            "mmseqs executable was not found. Install/load MMseqs2 or pass --mmseqs-bin /path/to/mmseqs.\n"
            "The repository environment.yml includes mmseqs2, but it must be active before running this script."
        )
    return resolved


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def run_mmseqs_cluster(
    mmseqs: str,
    fasta: Path,
    work_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    verbosity: int,
) -> Path:
    db = work_dir / "seqdb"
    cluster_db = work_dir / "clusters"
    tmp_dir = work_dir / "tmp"
    clusters_tsv = work_dir / "clusters.tsv"
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_command([mmseqs, "createdb", str(fasta), str(db), "-v", str(verbosity)])
    run_command(
        [
            mmseqs,
            "cluster",
            str(db),
            str(cluster_db),
            str(tmp_dir),
            "--min-seq-id",
            f"{min_seq_id:.3f}",
            "-c",
            f"{coverage:.3f}",
            "--cov-mode",
            str(cov_mode),
            "-v",
            str(verbosity),
        ]
    )
    run_command([mmseqs, "createtsv", str(db), str(db), str(cluster_db), str(clusters_tsv), "-v", str(verbosity)])
    return clusters_tsv


def load_cluster_membership(clusters_tsv: Path) -> pd.DataFrame:
    frame = pd.read_csv(clusters_tsv, sep="\t", names=["representative_fasta_id", "fasta_id"], dtype=str)
    frame["accession"] = frame["fasta_id"].map(accession_from_fasta_id)
    frame["representative_accession"] = frame["representative_fasta_id"].map(accession_from_fasta_id)
    frame["cluster_id"] = frame["representative_accession"]
    return frame[["cluster_id", "representative_fasta_id", "representative_accession", "fasta_id", "accession"]]


def cluster_stats(membership: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = membership.merge(labels[["accession", "target__trna_modifier"]], on="accession", how="left")
    merged["target__trna_modifier"] = pd.to_numeric(merged["target__trna_modifier"], errors="coerce").fillna(0).astype(int)
    stats = (
        merged.groupby("cluster_id", sort=True)
        .agg(
            cluster_size=("accession", "nunique"),
            positive_count=("target__trna_modifier", "sum"),
        )
        .reset_index()
    )
    stats["negative_count"] = stats["cluster_size"] - stats["positive_count"]
    stats["cluster_label"] = "mixed"
    stats.loc[stats["positive_count"].eq(0), "cluster_label"] = "negative"
    stats.loc[stats["positive_count"].eq(stats["cluster_size"]), "cluster_label"] = "positive"
    return stats


def assign_splits(
    stats: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    rng = random.Random(random_seed)
    target_fractions = {"train": train_fraction, "val": val_fraction, "test": test_fraction}
    total_size = int(stats["cluster_size"].sum())
    total_pos = int(stats["positive_count"].sum())
    total_neg = int(stats["negative_count"].sum())
    targets = {
        split: {
            "size": max(1.0, total_size * fraction),
            "positive": max(1.0, total_pos * fraction),
            "negative": max(1.0, total_neg * fraction),
        }
        for split, fraction in target_fractions.items()
    }
    running = {split: {"size": 0, "positive": 0, "negative": 0} for split in target_fractions}
    assignments: dict[str, str] = {}

    shuffled = stats.sample(frac=1.0, random_state=random_seed).sort_values(
        ["cluster_size", "positive_count"], ascending=[False, False], kind="mergesort"
    )
    for row in shuffled.to_dict(orient="records"):
        split_scores: list[tuple[float, float, str]] = []
        for split in target_fractions:
            size_deficit = (targets[split]["size"] - running[split]["size"]) / targets[split]["size"]
            if int(row["positive_count"]) and not int(row["negative_count"]):
                label_deficit = (targets[split]["positive"] - running[split]["positive"]) / targets[split]["positive"]
            elif int(row["negative_count"]) and not int(row["positive_count"]):
                label_deficit = (targets[split]["negative"] - running[split]["negative"]) / targets[split]["negative"]
            else:
                pos_deficit = (targets[split]["positive"] - running[split]["positive"]) / targets[split]["positive"]
                neg_deficit = (targets[split]["negative"] - running[split]["negative"]) / targets[split]["negative"]
                label_deficit = (pos_deficit + neg_deficit) / 2
            jitter = rng.random() * 1e-6
            split_scores.append((size_deficit + label_deficit + jitter, targets[split]["size"], split))
        _, _, best_split = max(split_scores, key=lambda item: (item[0], item[1]))
        assignments[str(row["cluster_id"])] = best_split
        running[best_split]["size"] += int(row["cluster_size"])
        running[best_split]["positive"] += int(row["positive_count"])
        running[best_split]["negative"] += int(row["negative_count"])

    out = stats.copy()
    out["split"] = out["cluster_id"].map(assignments)
    return out


def make_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in assignments.groupby("split", sort=True):
        rows.append(
            {
                "split": split,
                "rows": int(len(group)),
                "clusters": int(group["cluster_id"].nunique()),
                "positives": int(pd.to_numeric(group["target__trna_modifier"], errors="coerce").fillna(0).sum()),
                "negatives": int((pd.to_numeric(group["target__trna_modifier"], errors="coerce").fillna(0) == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("split", kind="mergesort")


def write_card(
    output: Path,
    membership: pd.DataFrame,
    split_assignments: pd.DataFrame,
    cluster_assignment: pd.DataFrame,
    summary: pd.DataFrame,
    parameters: dict[str, object],
) -> None:
    cluster_sizes = cluster_assignment["cluster_size"]
    payload = {
        "parameters": parameters,
        "n_sequences_clustered": int(len(membership)),
        "n_clusters": int(cluster_assignment["cluster_id"].nunique()),
        "cluster_size_min": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
        "cluster_size_median": float(cluster_sizes.median()) if len(cluster_sizes) else 0.0,
        "cluster_size_max": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "split_summary": summary.to_dict(orient="records"),
    }
    lines = [
        "# Cluster-Heldout Split Card",
        "",
        "Sequences were clustered with MMseqs2, then whole clusters were assigned to train, validation, or test splits.",
        "",
        "## Summary",
        "",
        f"- Clustered sequences: {payload['n_sequences_clustered']}",
        f"- Clusters: {payload['n_clusters']}",
        f"- Cluster size range: {payload['cluster_size_min']} to {payload['cluster_size_max']}",
        f"- Median cluster size: {payload['cluster_size_median']:.1f}",
        "",
        "## Split Counts",
        "",
        "| Split | Rows | Clusters | Positives | Negatives |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['split']} | {row['rows']} | {row['clusters']} | {row['positives']} | {row['negatives']} |"
        )
    lines.extend(["", "## JSON Summary", "", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_fractions(args.train_fraction, args.val_fraction, args.test_fraction)
    fasta = Path(args.fasta)
    label_matrix = Path(args.label_matrix)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mmseqs = require_mmseqs(args.mmseqs_bin)
    labels = pd.read_csv(label_matrix, sep="\t", dtype=str).fillna("")
    trainable_labels = labels[labels["is_trainable"].astype(str).str.lower().eq("true")].copy()

    work_dir = output_dir / "mmseqs_work"
    clusters_tsv = run_mmseqs_cluster(
        mmseqs=mmseqs,
        fasta=fasta,
        work_dir=work_dir,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        verbosity=args.mmseqs_verbosity,
    )
    membership = load_cluster_membership(clusters_tsv)
    stats = cluster_stats(membership, trainable_labels)
    cluster_assignment = assign_splits(
        stats,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )
    split_assignments = trainable_labels.merge(
        membership[["accession", "cluster_id", "representative_accession"]],
        on="accession",
        how="left",
    ).merge(cluster_assignment[["cluster_id", "split"]], on="cluster_id", how="left")
    split_assignments["split"] = split_assignments["split"].fillna("excluded")
    summary = make_summary(split_assignments[split_assignments["split"].ne("excluded")])

    membership.to_csv(output_dir / "cluster_membership.tsv", sep="\t", index=False)
    cluster_assignment.to_csv(output_dir / "cluster_assignments.tsv", sep="\t", index=False)
    split_assignments.to_csv(output_dir / "split_assignments.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "split_summary.tsv", sep="\t", index=False)
    write_card(
        output_dir / "cluster_split_card.md",
        membership=membership,
        split_assignments=split_assignments,
        cluster_assignment=cluster_assignment,
        summary=summary,
        parameters={
            "fasta": str(fasta),
            "label_matrix": str(label_matrix),
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "mmseqs_verbosity": args.mmseqs_verbosity,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "random_seed": args.random_seed,
        },
    )
    if not args.keep_mmseqs_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"wrote cluster membership to {output_dir / 'cluster_membership.tsv'}")
    print(f"wrote split assignments to {output_dir / 'split_assignments.tsv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
