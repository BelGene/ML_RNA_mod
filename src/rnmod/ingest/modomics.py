from __future__ import annotations

from pathlib import Path

import pandas as pd

from rnmod.ingest.common import write_records
from rnmod.settings import load_config


def ingest(config_path: str | Path, output: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    table_path = Path(config["sources"]["modomics"]["manual_table"])
    if not table_path.exists():
        return write_records([], output)
    df = pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        if not row.get("record_id", "") and not row.get("accession", ""):
            continue
        target = row.get("target_rna", "unknown") or "unknown"
        modification = row.get("modification", "")
        records.append(
            {
                "source_record_id": row.get("record_id") or row.get("accession"),
                "accession": row.get("accession", ""),
                "protein_name": row.get("protein_name", ""),
                "gene_name": row.get("gene_name", ""),
                "organism": row.get("organism", ""),
                "taxon_id": row.get("taxon_id", ""),
                "sequence": row.get("sequence", ""),
                "label_status": "gold_positive",
                "role_labels": "writer",
                "target_rna_labels": target,
                "modification_scope_labels": "unknown",
                "exact_modification_labels": modification,
                "site_bucket_labels": "unknown",
                "evidence_level": row.get("evidence_level", "curated_pathway_database") or "curated_pathway_database",
                "source_database": "MODOMICS",
                "source_type": "manual_modomics_import",
                "reaction": row.get("reaction", ""),
                "source_notes": row.get("notes", ""),
            }
        )
    return write_records(records, output)

