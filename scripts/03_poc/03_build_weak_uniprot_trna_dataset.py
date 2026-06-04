#!/usr/bin/env python3
"""Build a weak-label UniProt/MODOMICS tRNA-modifier POC dataset.

This script does not embed proteins. It only prepares an automated
training table, label matrix, FASTA, and dataset card for a first embedding
proof of concept.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "data/processed/poc_weak"
DEFAULT_PROFILE = "small"
DEFAULT_RANDOM_SEED = 42
DEFAULT_UNIPROT_BATCH_SIZE = 50
DEFAULT_REQUEST_PAUSE_SECONDS = 0.2

DATASET_PROFILES = {
    "small": {
        "taxonomy_scope": "bacteria_archaea",
        "page_size": 250,
        "max_records_per_query": 250,
        "max_easy_negative_pool": 1000,
        "max_easy_negatives": 500,
        "easy_negative_ratio": 0.75,
    },
    "standard": {
        "taxonomy_scope": "bacteria_archaea",
        "page_size": 500,
        "max_records_per_query": 1000,
        "max_easy_negative_pool": 2500,
        "max_easy_negatives": 1000,
        "easy_negative_ratio": 1.0,
    },
    "full": {
        "taxonomy_scope": "all",
        "page_size": 500,
        "max_records_per_query": 5000,
        "max_easy_negative_pool": 5000,
        "max_easy_negatives": 2000,
        "easy_negative_ratio": 1.0,
    },
}


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

from rnmod.dataset.weak_uniprot_trna_dataset import DEFAULT_MODOMICS_URL, TAXONOMY_SCOPES, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modomics-url", default=DEFAULT_MODOMICS_URL)
    parser.add_argument(
        "--profile",
        choices=sorted(DATASET_PROFILES),
        default=DEFAULT_PROFILE,
        help="Preset size/source policy. Use 'small' for the first POC; 'full' reproduces the large weak-label screen.",
    )
    parser.add_argument("--taxonomy-scope", choices=sorted(TAXONOMY_SCOPES), default=None)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--max-records-per-query", type=int, default=None)
    parser.add_argument("--max-easy-negative-pool", type=int, default=None)
    parser.add_argument("--max-easy-negatives", type=int, default=None)
    parser.add_argument("--easy-negative-ratio", type=float, default=None)
    parser.add_argument("--uniprot-batch-size", type=int, default=DEFAULT_UNIPROT_BATCH_SIZE)
    parser.add_argument("--request-pause", type=float, default=DEFAULT_REQUEST_PAUSE_SECONDS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--no-modomics",
        action="store_true",
        help="Use only UniProt weak labels and skip MODOMICS gold anchors.",
    )
    return parser.parse_args()


def profile_value(args: argparse.Namespace, key: str) -> object:
    value = getattr(args, key)
    return DATASET_PROFILES[args.profile][key] if value is None else value


def main() -> None:
    args = parse_args()
    taxonomy_scope = str(profile_value(args, "taxonomy_scope"))
    page_size = int(profile_value(args, "page_size"))
    max_records_per_query = int(profile_value(args, "max_records_per_query"))
    max_easy_negative_pool = int(profile_value(args, "max_easy_negative_pool"))
    max_easy_negatives = int(profile_value(args, "max_easy_negatives"))
    easy_negative_ratio = float(profile_value(args, "easy_negative_ratio"))

    print(
        "building weak-label tRNA modifier dataset "
        f"with profile={args.profile}, taxonomy_scope={taxonomy_scope}, "
        f"max_records_per_query={max_records_per_query}..."
    )
    summary = build_dataset(
        output_dir=args.output_dir,
        modomics_url=args.modomics_url,
        dataset_profile=args.profile,
        taxonomy_scope=taxonomy_scope,
        include_modomics=not args.no_modomics,
        page_size=page_size,
        max_records_per_query=max_records_per_query,
        max_easy_negative_pool=max_easy_negative_pool,
        easy_negative_ratio=easy_negative_ratio,
        max_easy_negatives=max_easy_negatives,
        uniprot_batch_size=args.uniprot_batch_size,
        request_pause=args.request_pause,
        random_seed=args.random_seed,
    )
    print(f"wrote {summary.n_rows} rows to {summary.dataset_path}")
    print(f"wrote {summary.n_trainable} trainable rows")
    print(f"wrote {summary.n_sequences} FASTA sequences to {summary.fasta_path}")
    print(f"wrote label matrix to {summary.label_matrix_path}")
    print(f"wrote dataset card to {summary.dataset_card_path}")


if __name__ == "__main__":
    main()
