from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

import json
from pathlib import Path

import pandas as pd

from rnmod.ingest.common import write_records
from rnmod.settings import load_config
from rnmod.utils.fasta import iter_fasta


def _seed_id_from_fasta_id(fasta_id: str) -> str:
    return fasta_id.split("|", 1)[0]


def _parse_header_fields(description: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    parts = description.split()
    header = parts[0] if parts else description
    for item in header.split("|")[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value.replace("_", " ")
    return fields


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {str(row["seed_id"]): dict(row) for _, row in frame.iterrows() if row.get("seed_id", "")}


def _record_from_seed(seed_id: str, sequence: str, header_fields: dict[str, str], metadata: dict[str, str], posneg: str) -> dict[str, object]:
    gene = metadata.get("gene_name") or header_fields.get("gene", "")
    family = metadata.get("family") or header_fields.get("family", "")
    substrate = metadata.get("substrate_class") or header_fields.get("substrate", "")
    evidence_level = metadata.get("evidence_level") or "annotated_homolog_day1"
    positive = posneg == "positive"
    label_status = "bronze_candidate"
    if positive and evidence_level in {"experimental_direct", "reviewed_swissprot", "curated_pathway_database"}:
        label_status = "silver_positive"
    if positive and evidence_level == "annotated_homolog_day1":
        label_status = "silver_positive"
    if not positive:
        label_status = "hard_negative"
    notes = metadata.get("notes", "")
    source_database = metadata.get("source_database", "EDL933_pilot_seed_library")
    return {
        "source_record_id": seed_id,
        "accession": metadata.get("accession") or header_fields.get("accession", ""),
        "protein_name": metadata.get("enzyme_class") or family,
        "gene_name": gene,
        "family": family,
        "organism": metadata.get("organism") or header_fields.get("organism", ""),
        "taxon_group": metadata.get("taxon_group", ""),
        "sequence": sequence,
        "label_status": label_status,
        "role_labels": "writer" if positive else "nonRNA_enzyme",
        "target_rna_labels": _target_from_substrate(substrate, positive),
        "modification_scope_labels": _scope_from_text(f"{family} {metadata.get('modification', '')}", positive),
        "chemistry_labels": _chemistry_from_text(f"{family} {metadata.get('enzyme_class', '')}", positive),
        "exact_modification_labels": metadata.get("modification", ""),
        "site_bucket_labels": _site_from_text(metadata.get("modification", "")),
        "evidence_level": evidence_level,
        "source_database": source_database,
        "source_type": "legacy_edl933_pilot_seed",
        "substrate_class": substrate,
        "legacy_seed_id": seed_id,
        "source_notes": notes,
        "provenance_json": json.dumps({"legacy_positive_or_negative": posneg, "metadata_source": "expanded_seed_metadata.tsv"}),
    }


def _target_from_substrate(substrate: str, positive: bool) -> str:
    text = substrate.lower()
    if not positive:
        if "dna" in text:
            return "DNA"
        if "protein" in text:
            return "protein"
        return "small_molecule" if text else "unknown"
    labels = []
    if "trna" in text:
        labels.append("tRNA")
    if "rrna" in text:
        labels.append("rRNA")
    if "mrna" in text:
        labels.append("mRNA")
    if len(labels) > 1:
        return "mixed_RNA"
    return labels[0] if labels else "unknown"


def _scope_from_text(text: str, positive: bool) -> str:
    lower = text.lower()
    if not positive:
        return "nonRNA_modification"
    if "pseudo" in lower:
        return "pseudouridylation"
    if "thiol" in lower or "s2" in lower:
        return "thiolation"
    if "deamin" in lower:
        return "deamination"
    if "queu" in lower or "preq" in lower:
        return "queuosine"
    if "lysidine" in lower or "tils" in lower:
        return "lysidine"
    if "miaa" in lower or "isopentenyl" in lower:
        return "isopentenylation"
    if "methyl" in lower or "rlm" in lower or "rsm" in lower or "trm" in lower:
        return "methylation"
    return "unknown"


def _chemistry_from_text(text: str, positive: bool) -> str:
    lower = text.lower()
    if not positive:
        if "dna" in lower or "restriction" in lower:
            return "DNA_methyltransferase"
        if "protein" in lower:
            return "protein_methyltransferase"
        if "deamin" in lower:
            return "deaminase"
        return "small_molecule_methyltransferase"
    if "spout" in lower:
        return "SPOUT_methyltransferase"
    if "radical" in lower or "miab" in lower or "rlmn" in lower:
        return "radical_SAM"
    if "pseudo" in lower or "rlu" in lower or "tru" in lower or "rsua" in lower:
        return "pseudouridine_synthase"
    if "thiol" in lower or "mnma" in lower or "ttca" in lower:
        return "thiouridylase"
    if "deamin" in lower or "tada" in lower:
        return "deaminase"
    if "que" in lower or "tgt" in lower:
        return "queuosine_pathway"
    if "methyl" in lower or "rlm" in lower or "rsm" in lower or "trm" in lower:
        return "SAM_methyltransferase"
    return "unknown"


def _site_from_text(text: str) -> str:
    lower = text.lower()
    if "34" in lower:
        return "wobble_position_34"
    if "37" in lower:
        return "position_37"
    if "1402" in lower or "decoding" in lower:
        return "rRNA_decoding_center"
    if "rrna" in lower:
        return "rRNA_other"
    if "trna" in lower:
        return "tRNA_body"
    return "unknown"


def ingest(config_path: str | Path, output: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    source_cfg = config["sources"]["legacy_pilot"]
    if not source_cfg.get("enabled", False):
        return write_records([], output)
    metadata = _load_metadata(Path(source_cfg["metadata"]))
    records: list[dict[str, object]] = []
    for posneg, key in (("positive", "positive_fasta"), ("negative", "negative_fasta")):
        fasta_path = Path(source_cfg[key])
        for seq_record in iter_fasta(fasta_path):
            seed_id = _seed_id_from_fasta_id(seq_record.id)
            header_fields = _parse_header_fields(seq_record.description)
            records.append(
                _record_from_seed(
                    seed_id=seed_id,
                    sequence=str(seq_record.seq),
                    header_fields=header_fields,
                    metadata=metadata.get(seed_id, {}),
                    posneg=posneg,
                )
            )
    return write_records(records, output)
