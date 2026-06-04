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


RHEA_COLUMNS = ["rhea_id", "equation", "enzyme_class", "chemistry_hint", "source_database", "source_type"]


def _download(url: str, output: Path) -> None:
    if not url:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output.write_bytes(response.content)


def _chemistry_hint(equation: str) -> str:
    text = equation.lower()
    if "s-adenosyl-l-methionine" in text or "s-adenosylmethionine" in text:
        return "SAM_methyltransferase"
    if "pseudouridine" in text:
        return "pseudouridine_synthase"
    if "thiouridine" in text or "sulfur" in text:
        return "thiouridylase"
    if "adenosine" in text and "inosine" in text:
        return "deaminase"
    return "unknown"


def ingest(config_path: str | Path, output: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    source_cfg = config["sources"]["rhea"]
    if not source_cfg.get("enabled", True):
        frame = pd.DataFrame(columns=RHEA_COLUMNS)
        write_parquet(frame, ensure_parent(output))
        return frame
    raw = Path(source_cfg["raw_tsv"])
    if source_cfg.get("enabled", True) and source_cfg.get("fetch", False) and not raw.exists():
        _download(source_cfg.get("url", ""), raw)
    if not raw.exists():
        frame = pd.DataFrame(columns=RHEA_COLUMNS)
        write_parquet(frame, ensure_parent(output))
        return frame
    df = pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    for _, row in df.iterrows():
        equation = row.get("Equation", row.get("equation", ""))
        rhea_id = row.get("RHEA_ID", row.get("REACTION_ID", row.get("rhea_id", row.get("ID", ""))))
        rows.append(
            {
                "rhea_id": rhea_id,
                "equation": equation,
                "enzyme_class": row.get("EC", row.get("enzyme_class", "")),
                "chemistry_hint": _chemistry_hint(equation),
                "source_database": "Rhea",
                "source_type": "rhea_reaction_table",
            }
        )
    frame = pd.DataFrame(rows, columns=RHEA_COLUMNS)
    write_parquet(frame.sort_values("rhea_id", kind="mergesort"), ensure_parent(output))
    return frame
