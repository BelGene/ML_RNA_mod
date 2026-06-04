from __future__ import annotations

from rnmod.labels.assign_labels import apply_family_label_rules
from rnmod.schemas import ProteinRecord


def test_trmd_gets_rna_modification_labels() -> None:
    row = apply_family_label_rules(
        {
            "source_record_id": "trmd",
            "source_database": "test",
            "protein_name": "tRNA (guanine-N(1)-)-methyltransferase",
            "gene_name": "trmD",
            "sequence": "MKKLL",
            "label_status": "unknown",
            "role_labels": "unknown",
            "target_rna_labels": "unknown",
            "modification_scope_labels": "unknown",
            "chemistry_labels": "unknown",
            "site_bucket_labels": "unknown",
        }
    )
    record = ProteinRecord(**row).to_row()
    assert record["label_status"] == "silver_positive"
    assert "tRNA" in record["target_rna_labels"]
    assert "methylation" in record["modification_scope_labels"]
    assert "position_37" in record["site_bucket_labels"]


def test_pseudouridine_family_labels() -> None:
    row = apply_family_label_rules(
        {
            "source_record_id": "rlu",
            "source_database": "test",
            "protein_name": "23S rRNA pseudouridine synthase RluD",
            "gene_name": "rluD",
            "sequence": "MKKLL",
            "label_status": "unknown",
        }
    )
    record = ProteinRecord(**row).to_row()
    assert record["label_status"] == "silver_positive"
    assert "pseudouridylation" in record["modification_scope_labels"]
    assert "pseudouridine_synthase" in record["chemistry_labels"]


def test_dna_methyltransferase_becomes_hard_negative() -> None:
    row = apply_family_label_rules(
        {
            "source_record_id": "dam",
            "source_database": "test",
            "protein_name": "DNA adenine methyltransferase",
            "gene_name": "dam",
            "sequence": "MKKLL",
            "label_status": "unknown",
        }
    )
    record = ProteinRecord(**row).to_row()
    assert record["label_status"] == "hard_negative"
    assert "DNA" in record["target_rna_labels"]
    assert "DNA_methyltransferase" in record["chemistry_labels"]
    assert record["is_train_negative"] is True


def test_hypothetical_protein_never_hard_negative() -> None:
    row = apply_family_label_rules(
        {
            "source_record_id": "hyp",
            "source_database": "test",
            "protein_name": "hypothetical protein DNA methyltransferase-like",
            "gene_name": "",
            "sequence": "MKKLL",
            "label_status": "unknown",
        }
    )
    record = ProteinRecord(**row).to_row()
    assert record["label_status"] != "hard_negative"
    assert record["is_trainable"] is False


def test_nudix_without_rna_cap_context_is_not_positive() -> None:
    row = apply_family_label_rules(
        {
            "source_record_id": "nudix",
            "source_database": "test",
            "protein_name": "Nudix hydrolase",
            "gene_name": "nudC",
            "sequence": "MKKLL",
            "label_status": "unknown",
        }
    )
    record = ProteinRecord(**row).to_row()
    assert record["label_status"] == "unknown"
    assert "Nudix_hydrolase" in record["chemistry_labels"]


def test_modb_rnaylation_candidate() -> None:
    row = apply_family_label_rules(
        {
            "source_record_id": "modb",
            "source_database": "test",
            "protein_name": "ModB ADP-ribosyltransferase involved in RNAylation",
            "gene_name": "modB",
            "sequence": "MKKLL",
            "label_status": "unknown",
        }
    )
    record = ProteinRecord(**row).to_row()
    assert record["label_status"] == "bronze_candidate"
    assert "RNAylation" in record["modification_scope_labels"]

