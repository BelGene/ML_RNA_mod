from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

from pathlib import Path
from typing import Iterable

import pandas as pd

from rnmod.labels.assign_labels import apply_family_label_rules
from rnmod.labels.evidence import evidence_score
from rnmod.schemas import MASTER_COLUMNS, ProteinRecord
from rnmod.settings import ensure_parent
from rnmod.utils.tables import write_parquet


def empty_records_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MASTER_COLUMNS)


def records_to_frame(records: Iterable[dict[str, object]], evidence_config: dict | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in records:
        labeled = apply_family_label_rules(dict(raw))
        labeled["evidence_score"] = evidence_score(str(labeled.get("evidence_level", "unknown")), evidence_config)
        record = ProteinRecord(**labeled)
        rows.append(record.to_row())
    if not rows:
        return empty_records_frame()
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=MASTER_COLUMNS).sort_values(
        ["source_database", "accession", "source_record_id"], kind="mergesort"
    )


def write_records(records: Iterable[dict[str, object]], output: str | Path, evidence_config: dict | None = None) -> pd.DataFrame:
    frame = records_to_frame(records, evidence_config)
    write_parquet(frame, ensure_parent(output))
    return frame

