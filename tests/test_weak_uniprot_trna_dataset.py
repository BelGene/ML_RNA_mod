from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from rnmod.dataset.weak_uniprot_trna_dataset import (
    build_label_matrix,
    classify_record,
    select_length_matched_easy_negatives,
)


def load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cluster_split = load_script_module("cluster_split_sequences", "scripts/03_poc/04_cluster_split_sequences.py")


def test_modomics_anchor_is_gold_positive() -> None:
    row = classify_record(
        {
            "accession": "P12345",
            "protein_name": "tRNA pseudouridine synthase",
            "sequence": "MKKLL",
            "is_modomics_anchor": True,
            "source_label_hints": "",
            "source_query_ids": "",
        }
    )
    assert row["label_status"] == "gold_positive"
    assert row["is_trna_modifier"] is True
    assert row["is_trainable"] is True
    assert "pseudouridine_synthase" in row["mechanism_labels"]


def test_reviewed_trna_query_hit_is_weak_positive() -> None:
    row = classify_record(
        {
            "accession": "Q1",
            "protein_name": "tRNA (guanine-N(1)-)-methyltransferase",
            "sequence": "MKKLL",
            "is_modomics_anchor": False,
            "source_label_hints": "weak_positive",
            "source_query_ids": "trna_methyltransferase",
        }
    )
    assert row["label_status"] == "weak_positive"
    assert row["target_rna_labels"] == "tRNA"
    assert "methyltransferase" in row["mechanism_labels"]


def test_positive_and_negative_query_hit_is_excluded_as_conflict() -> None:
    row = classify_record(
        {
            "accession": "Q2",
            "protein_name": "RNA/DNA methyltransferase",
            "sequence": "MKKLL",
            "is_modomics_anchor": False,
            "source_label_hints": "weak_positive;hard_negative",
            "source_query_ids": "trna_methyltransferase;negative_dna_methyltransferase",
        }
    )
    assert row["label_status"] == "conflicted"
    assert row["is_trainable"] is False
    assert row["train_exclusion_reason"] == "status=conflicted"


def test_hard_negative_is_non_trna_modifier() -> None:
    row = classify_record(
        {
            "accession": "Q3",
            "protein_name": "DNA adenine methyltransferase",
            "sequence": "MKKLL",
            "is_modomics_anchor": False,
            "source_label_hints": "hard_negative",
            "source_query_ids": "negative_dna_methyltransferase",
        }
    )
    assert row["label_status"] == "hard_negative"
    assert row["is_trna_modifier"] is False
    assert row["negative_type"] == "hard_negative_near_family"
    assert row["mechanism_labels"] == ""


def test_missing_sequence_excludes_otherwise_valid_record() -> None:
    row = classify_record(
        {
            "accession": "Q4",
            "protein_name": "tRNA methyltransferase",
            "sequence": "",
            "is_modomics_anchor": False,
            "source_label_hints": "weak_positive",
            "source_query_ids": "trna_methyltransferase",
        }
    )
    assert row["label_status"] == "weak_positive"
    assert row["is_trainable"] is False
    assert row["train_exclusion_reason"] == "missing_sequence"


def test_label_matrix_has_binary_and_mechanism_targets() -> None:
    rows = pd.DataFrame(
        [
            classify_record(
                {
                    "accession": "P1",
                    "protein_name": "tRNA methyltransferase",
                    "sequence": "MKKLL",
                    "is_modomics_anchor": False,
                    "source_label_hints": "weak_positive",
                    "source_query_ids": "trna_methyltransferase",
                }
            ),
            classify_record(
                {
                    "accession": "N1",
                    "protein_name": "DNA methyltransferase",
                    "sequence": "MKKLL",
                    "is_modomics_anchor": False,
                    "source_label_hints": "hard_negative",
                    "source_query_ids": "negative_dna_methyltransferase",
                }
            ),
        ]
    )
    matrix = build_label_matrix(rows)
    assert int(matrix.loc[matrix["accession"].eq("P1"), "target__trna_modifier"].iloc[0]) == 1
    assert int(matrix.loc[matrix["accession"].eq("N1"), "target__trna_modifier"].iloc[0]) == 0
    assert "mechanism__methyltransferase" in matrix.columns


def test_easy_negative_selection_is_deterministic_and_limited() -> None:
    easy = pd.DataFrame(
        [
            {"accession": f"E{idx}", "sequence_length": length}
            for idx, length in enumerate([100, 120, 220, 260, 310, 410, 510, 700, 900, 1200])
        ]
    )
    positives = pd.DataFrame(
        [
            {"accession": "P1", "sequence_length": 115},
            {"accession": "P2", "sequence_length": 305},
            {"accession": "P3", "sequence_length": 520},
        ]
    )
    selected_a = select_length_matched_easy_negatives(easy, positives, target_count=4, random_seed=7)
    selected_b = select_length_matched_easy_negatives(easy, positives, target_count=4, random_seed=7)
    assert len(selected_a) == 4
    assert selected_a["accession"].tolist() == selected_b["accession"].tolist()


def test_cluster_split_assignment_balances_large_clusters() -> None:
    stats = pd.DataFrame(
        [
            {"cluster_id": "p1", "cluster_size": 100, "positive_count": 100, "negative_count": 0},
            {"cluster_id": "p2", "cluster_size": 80, "positive_count": 80, "negative_count": 0},
            {"cluster_id": "p3", "cluster_size": 60, "positive_count": 60, "negative_count": 0},
            {"cluster_id": "n1", "cluster_size": 90, "positive_count": 0, "negative_count": 90},
            {"cluster_id": "n2", "cluster_size": 70, "positive_count": 0, "negative_count": 70},
            {"cluster_id": "n3", "cluster_size": 50, "positive_count": 0, "negative_count": 50},
        ]
    )
    assigned = cluster_split.assign_splits(stats, 0.70, 0.15, 0.15, random_seed=42)
    summary = assigned.groupby("split")["cluster_size"].sum()
    assert summary["train"] > summary["val"]
    assert summary["train"] > summary["test"]
