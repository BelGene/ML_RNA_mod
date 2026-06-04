#!/usr/bin/env python3
"""Download raw public source files required by the dataset pipeline."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# Routine changes should be made in configs/config.yaml. These defaults are
# repository-relative so any clone can run without local path edits.
DEFAULT_CONFIG = "configs/config.yaml"
DEFAULT_MARKER = "data/raw/.sources_downloaded.ok"
DEFAULT_FORCE = False
DEFAULT_UNIPROT_PAGE_SIZE = 500
DEFAULT_TIMEOUT_SECONDS = 120

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


def find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from script path.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rnmod.ingest.uniprot import download_uniprot_raw, read_query_config


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_yaml(path: str | Path) -> dict:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_bytes_from_url(url: str, output: str | Path, force: bool) -> bool:
    if not url:
        return False
    out = repo_path(output)
    if out.exists() and not force:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    out.write_bytes(response.content)
    return True


def download_uniprot(config: dict, force: bool) -> bool:
    source_cfg = config.get("sources", {}).get("uniprot", {})
    if not source_cfg.get("enabled", True) or not source_cfg.get("fetch", False):
        return False
    raw_tsv = repo_path(source_cfg.get("raw_tsv", "data/raw/uniprot/uniprot_records.tsv"))
    if raw_tsv.exists() and not force:
        return False

    query_cfg = read_query_config(repo_path(source_cfg.get("query_config", "configs/queries_uniprot.yaml")))
    query_cfg.setdefault("page_size", DEFAULT_UNIPROT_PAGE_SIZE)
    download_uniprot_raw(query_cfg, raw_tsv, timeout=DEFAULT_TIMEOUT_SECONDS)
    return True


def download_configured_sources(config: dict, force: bool) -> list[str]:
    downloaded: list[str] = []
    if download_uniprot(config, force):
        downloaded.append("uniprot")

    sources = config.get("sources", {})
    rhea = sources.get("rhea", {})
    if rhea.get("enabled", True) and rhea.get("fetch", False):
        if write_bytes_from_url(rhea.get("url", ""), rhea.get("raw_tsv", ""), force):
            downloaded.append("rhea")

    go = sources.get("go", {})
    if go.get("enabled", True) and go.get("fetch", False):
        if write_bytes_from_url(go.get("url", ""), go.get("raw_obo", ""), force):
            downloaded.append("go")

    legacy = sources.get("legacy_pilot", {})
    if legacy.get("enabled", False):
        for key in ("positive_fasta", "negative_fasta", "metadata"):
            url = legacy.get(f"{key}_url", "")
            path = legacy.get(key, "")
            if write_bytes_from_url(url, path, force):
                downloaded.append(f"legacy_pilot:{key}")
    return downloaded


def write_marker(path: str | Path, downloaded: list[str]) -> None:
    marker = repo_path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    marker.write_text(f"downloaded_at_utc\t{timestamp}\nsources\t{';'.join(downloaded) or 'already_present_or_disabled'}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--force", action="store_true", default=DEFAULT_FORCE)
    args = parser.parse_args()

    config = load_yaml(args.config)
    downloaded = download_configured_sources(config, args.force)
    write_marker(args.marker, downloaded)


if __name__ == "__main__":
    main()
