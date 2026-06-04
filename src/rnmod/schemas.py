from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

import hashlib
from typing import Any

try:
    from pydantic.v1 import BaseModel, Field, validator
except ImportError:  # pragma: no cover - for older environments with Pydantic v1 only.
    from pydantic import BaseModel, Field, validator

from rnmod.utils.checksums import sequence_md5, sequence_sha256
from rnmod.utils.fasta import clean_sequence
from rnmod.utils.tables import join_labels, split_labels


CONTROLLED_VOCAB: dict[str, set[str]] = {
    "label_status": {
        "gold_positive",
        "silver_positive",
        "bronze_candidate",
        "hard_negative",
        "easy_negative",
        "unknown",
        "conflicted",
    },
    "role_labels": {
        "writer",
        "eraser",
        "reader_or_cofactor",
        "sulfur_relay",
        "RNA_cap_processing",
        "nonRNA_enzyme",
        "unknown",
    },
    "target_rna_labels": {
        "tRNA",
        "rRNA",
        "mRNA",
        "tmRNA",
        "ncRNA",
        "mixed_RNA",
        "DNA",
        "protein",
        "small_molecule",
        "unknown",
    },
    "modification_scope_labels": {
        "methylation",
        "pseudouridylation",
        "deamination",
        "thiolation",
        "queuosine",
        "lysidine",
        "isopentenylation",
        "RNA_cap_or_terminal",
        "RNAylation",
        "nonRNA_modification",
        "unknown",
    },
    "chemistry_labels": {
        "SAM_methyltransferase",
        "SPOUT_methyltransferase",
        "FAD_dependent_methyltransferase",
        "radical_SAM",
        "pseudouridine_synthase",
        "deaminase",
        "thiouridylase",
        "queuosine_pathway",
        "tRNA_ligase_or_synthetase",
        "Nudix_hydrolase",
        "ADP_ribosyltransferase",
        "DNA_methyltransferase",
        "protein_methyltransferase",
        "small_molecule_methyltransferase",
        "unknown",
    },
    "site_bucket_labels": {
        "anticodon_loop",
        "wobble_position_34",
        "position_37",
        "tRNA_body",
        "rRNA_decoding_center",
        "rRNA_peptidyl_transferase_center",
        "rRNA_other",
        "RNA_5prime_cap_or_end",
        "protein_or_nonRNA",
        "unknown",
    },
}

LIST_FIELDS = [
    "role_labels",
    "target_rna_labels",
    "modification_scope_labels",
    "chemistry_labels",
    "exact_modification_labels",
    "site_bucket_labels",
    "cofactor_labels",
]

MASTER_COLUMNS = [
    "protein_uid",
    "source_record_id",
    "accession",
    "reviewed",
    "protein_name",
    "gene_name",
    "family",
    "organism",
    "taxon_id",
    "taxon_group",
    "sequence",
    "sequence_md5",
    "sequence_sha256",
    "sequence_length",
    "is_fragment",
    "is_hypothetical",
    "source_database",
    "source_type",
    "source_url",
    "reference_pmid",
    "reference_doi",
    "evidence_level",
    "evidence_score",
    "label_status",
    "role_labels",
    "target_rna_labels",
    "modification_scope_labels",
    "chemistry_labels",
    "exact_modification_labels",
    "site_bucket_labels",
    "cofactor_labels",
    "reaction",
    "substrate_class",
    "is_rna_modifier",
    "is_train_positive",
    "is_train_negative",
    "is_trainable",
    "train_exclusion_reason",
    "legacy_seed_id",
    "source_notes",
    "provenance_json",
    "duplicate_sequence_count",
    "representative_uid",
]


class ProteinRecord(BaseModel):
    source_record_id: str
    accession: str = ""
    reviewed: bool = False
    protein_name: str = ""
    gene_name: str = ""
    family: str = ""
    organism: str = ""
    taxon_id: str = ""
    taxon_group: str = ""
    sequence: str = ""
    is_fragment: bool = False
    is_hypothetical: bool = False
    source_database: str
    source_type: str = "curated_table"
    source_url: str = ""
    reference_pmid: str = ""
    reference_doi: str = ""
    evidence_level: str = "unknown"
    evidence_score: float = 0.0
    label_status: str = "unknown"
    role_labels: list[str] = Field(default_factory=lambda: ["unknown"])
    target_rna_labels: list[str] = Field(default_factory=lambda: ["unknown"])
    modification_scope_labels: list[str] = Field(default_factory=lambda: ["unknown"])
    chemistry_labels: list[str] = Field(default_factory=lambda: ["unknown"])
    exact_modification_labels: list[str] = Field(default_factory=list)
    site_bucket_labels: list[str] = Field(default_factory=lambda: ["unknown"])
    cofactor_labels: list[str] = Field(default_factory=list)
    reaction: str = ""
    substrate_class: str = ""
    legacy_seed_id: str = ""
    source_notes: str = ""
    provenance_json: str = "{}"

    @validator("sequence", pre=True)
    def normalize_sequence(cls, value: object) -> str:
        return clean_sequence(value)

    @validator("label_status")
    def validate_label_status(cls, value: str) -> str:
        value = value or "unknown"
        if value not in CONTROLLED_VOCAB["label_status"]:
            raise ValueError(f"Invalid label_status: {value}")
        return value

    @validator(*LIST_FIELDS, pre=True)
    def normalize_list_fields(cls, value: object) -> list[str]:
        labels = split_labels(value)
        return labels if labels else []

    @validator("role_labels", "target_rna_labels", "modification_scope_labels", "chemistry_labels", "site_bucket_labels")
    def validate_controlled_lists(cls, values: list[str], field: Any) -> list[str]:
        allowed = CONTROLLED_VOCAB[field.name]
        clean = values if values else ["unknown"]
        bad = sorted(set(clean) - allowed)
        if bad:
            raise ValueError(f"Invalid labels for {field.name}: {bad}")
        return sorted(set(clean))

    def to_row(self) -> dict[str, Any]:
        seq_md5 = sequence_md5(self.sequence)
        seq_sha256 = sequence_sha256(self.sequence)
        uid_source = seq_sha256 or f"{self.source_database}:{self.source_record_id}:{self.accession}"
        uid_hash = hashlib.sha256(uid_source.encode("utf-8")).hexdigest()[:16]
        accession_part = self.accession.replace("|", "_") if self.accession else "no_accession"
        protein_uid = f"rnmod:{accession_part}:{uid_hash}"
        row = self.dict()
        row["protein_uid"] = protein_uid
        row["sequence_md5"] = seq_md5
        row["sequence_sha256"] = seq_sha256
        row["sequence_length"] = len(self.sequence)
        row["is_rna_modifier"] = self.label_status in {"gold_positive", "silver_positive", "bronze_candidate"}
        row["is_train_positive"] = self.label_status in {"gold_positive", "silver_positive"}
        row["is_train_negative"] = self.label_status == "hard_negative"
        row["is_trainable"] = bool(row["is_train_positive"] or row["is_train_negative"])
        row["train_exclusion_reason"] = ""
        if not row["is_trainable"]:
            row["train_exclusion_reason"] = f"status={self.label_status}"
        if self.is_fragment:
            row["is_trainable"] = False
            row["train_exclusion_reason"] = "fragmentary_sequence"
        if self.is_hypothetical:
            row["is_trainable"] = False
            row["train_exclusion_reason"] = "hypothetical_or_unknown_product"
        if not self.sequence:
            row["is_trainable"] = False
            row["train_exclusion_reason"] = "missing_sequence"
        for field_name in LIST_FIELDS:
            row[field_name] = join_labels(row.get(field_name))
        row["duplicate_sequence_count"] = 1
        row["representative_uid"] = protein_uid
        return {column: row.get(column, "") for column in MASTER_COLUMNS}


def empty_master_frame() -> list[dict[str, Any]]:
    return []
