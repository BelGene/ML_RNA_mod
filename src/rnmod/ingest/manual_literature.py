from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

from pathlib import Path

import pandas as pd

from rnmod.ingest.common import write_records
from rnmod.settings import load_config


def ingest(config_path: str | Path, output: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    source_cfg = config["sources"]["manual_literature"]
    if not source_cfg.get("enabled", True):
        return write_records([], output)
    source = source_cfg["table"]
    table_path = Path(source)
    if not table_path.exists():
        return write_records([], output)
    df = pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        if not any(str(value).strip() for value in row.values):
            continue
        records.append(
            {
                "source_record_id": row.get("record_id") or row.get("accession") or f"manual_{len(records) + 1}",
                "accession": row.get("accession", ""),
                "protein_name": row.get("protein_name", ""),
                "gene_name": row.get("gene_name", ""),
                "organism": row.get("organism", ""),
                "taxon_id": row.get("taxon_id", ""),
                "sequence": row.get("sequence", ""),
                "label_status": row.get("label_status", "unknown") or "unknown",
                "role_labels": row.get("role_labels", "unknown"),
                "target_rna_labels": row.get("target_rna_labels", "unknown"),
                "modification_scope_labels": row.get("modification_scope_labels", "unknown"),
                "chemistry_labels": row.get("chemistry_labels", "unknown"),
                "exact_modification_labels": row.get("exact_modification_labels", ""),
                "site_bucket_labels": row.get("site_bucket_labels", "unknown"),
                "evidence_level": row.get("evidence_level", "manual_hypothesis") or "manual_hypothesis",
                "source_database": row.get("source_database", "manual_literature") or "manual_literature",
                "source_type": "manual_literature",
                "source_url": row.get("source_url", ""),
                "reference_pmid": row.get("reference_pmid", ""),
                "reference_doi": row.get("reference_doi", ""),
                "source_notes": row.get("notes", ""),
            }
        )
    return write_records(records, output)
