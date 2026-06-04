from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

from pathlib import Path

import pandas as pd
import requests

from rnmod.settings import ensure_parent, load_config
from rnmod.utils.tables import write_parquet


GO_COLUMNS = ["go_id", "name", "namespace", "is_rna_modification_term", "source_database", "source_type"]


def _download(url: str, output: Path) -> None:
    if not url:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output.write_bytes(response.content)


def _parse_obo_terms(path: Path) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    current: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                if current:
                    terms.append(current)
                current = {}
                continue
            if not line or line.startswith("!"):
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                if key in {"id", "name", "namespace"}:
                    current[key] = value
        if current:
            terms.append(current)
    return terms


def ingest(config_path: str | Path, output: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    source_cfg = config["sources"]["go"]
    if not source_cfg.get("enabled", True):
        frame = pd.DataFrame(columns=GO_COLUMNS)
        write_parquet(frame, ensure_parent(output))
        return frame
    raw = Path(source_cfg["raw_obo"])
    if source_cfg.get("enabled", True) and source_cfg.get("fetch", False) and not raw.exists():
        _download(source_cfg.get("url", ""), raw)
    if not raw.exists():
        frame = pd.DataFrame(columns=GO_COLUMNS)
        write_parquet(frame, ensure_parent(output))
        return frame
    rows = []
    for term in _parse_obo_terms(raw):
        name = term.get("name", "")
        is_rna_mod = "rna" in name.lower() and ("modification" in name.lower() or "editing" in name.lower())
        rows.append(
            {
                "go_id": term.get("id", ""),
                "name": name,
                "namespace": term.get("namespace", ""),
                "is_rna_modification_term": bool(is_rna_mod),
                "source_database": "Gene Ontology",
                "source_type": "go_obo",
            }
        )
    frame = pd.DataFrame(rows, columns=GO_COLUMNS).sort_values("go_id", kind="mergesort")
    write_parquet(frame, ensure_parent(output))
    return frame
