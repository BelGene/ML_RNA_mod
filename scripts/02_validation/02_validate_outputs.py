#!/usr/bin/env python3
"""Validate that the processed dataset outputs exist and are internally usable."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# Keep these repository-relative unless you intentionally change paths in
# configs/config.yaml.
DEFAULT_CONFIG = "configs/config.yaml"
MIN_MASTER_ROWS = 1

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from script path.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
os.chdir(REPO_ROOT)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rnmod.settings import load_config, processed_path


REQUIRED_OUTPUTS = [
    "rnmod_master.parquet",
    "rnmod_master.tsv.gz",
    "rnmod_sequences.faa",
    "rnmod_label_matrix.parquet",
    "source_manifest.tsv",
    "rnmod_dataset_card.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--min-master-rows", type=int, default=MIN_MASTER_ROWS)
    args = parser.parse_args()

    config = load_config(args.config)
    missing = [str(processed_path(config, filename)) for filename in REQUIRED_OUTPUTS if not processed_path(config, filename).exists()]
    if missing:
        raise SystemExit("Missing processed outputs:\n" + "\n".join(missing))

    master_path = processed_path(config, "rnmod_master.parquet")
    labels_path = processed_path(config, "rnmod_label_matrix.parquet")
    master = pd.read_parquet(master_path)
    labels = pd.read_parquet(labels_path)
    if len(master) < args.min_master_rows:
        raise SystemExit(f"Master table has {len(master)} rows; expected at least {args.min_master_rows}.")
    if len(labels) != len(master):
        raise SystemExit(f"Label matrix rows ({len(labels)}) do not match master rows ({len(master)}).")
    print(f"validated {len(master)} master rows and {len(labels)} label rows")


if __name__ == "__main__":
    main()
