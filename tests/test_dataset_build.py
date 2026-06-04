from __future__ import annotations

from pathlib import Path

import pandas as pd

from rnmod.dataset.build_master_table import _add_sequence_representatives, _label_matrix
from rnmod.dataset.dataset_cards import make_dataset_card
from rnmod.ingest.common import records_to_frame
from rnmod.utils.checksums import file_md5


def test_sequence_deduplication_marks_representative() -> None:
    frame = records_to_frame(
        [
            {
                "source_record_id": "a",
                "accession": "A",
                "source_database": "test",
                "protein_name": "tRNA methyltransferase TrmD",
                "gene_name": "trmD",
                "sequence": "MKKLL",
            },
            {
                "source_record_id": "b",
                "accession": "B",
                "source_database": "test",
                "protein_name": "tRNA methyltransferase TrmD",
                "gene_name": "trmD",
                "sequence": "MKKLL",
            },
        ]
    )
    deduped = _add_sequence_representatives(frame)
    assert set(deduped["duplicate_sequence_count"].astype(int)) == {2}
    assert deduped["representative_uid"].nunique() == 1


def test_label_matrix_matches_master_rows() -> None:
    frame = records_to_frame(
        [
            {
                "source_record_id": "a",
                "accession": "A",
                "source_database": "test",
                "protein_name": "DNA adenine methyltransferase",
                "gene_name": "dam",
                "sequence": "MKKLL",
            }
        ]
    )
    matrix = _label_matrix(frame)
    assert len(matrix) == len(frame)
    assert "target_rna__DNA" in matrix.columns
    assert int(matrix.loc[0, "target_rna__DNA"]) == 1


def test_manifest_checksum_generation(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    source.write_text("a\tb\n1\t2\n", encoding="utf-8")
    assert file_md5(source)


def test_dataset_card_counts_match_master(tmp_path: Path) -> None:
    frame = records_to_frame(
        [
            {
                "source_record_id": "a",
                "accession": "A",
                "source_database": "test",
                "protein_name": "tRNA methyltransferase TrmD",
                "gene_name": "trmD",
                "sequence": "MKKLL",
            }
        ]
    )
    manifest = pd.DataFrame([{"source_type": "manual", "source_name": "test"}])
    card = tmp_path / "card.md"
    make_dataset_card(frame, manifest, card)
    text = card.read_text(encoding="utf-8")
    assert "Protein records: 1" in text
    assert "| silver_positive | 1 |" in text

