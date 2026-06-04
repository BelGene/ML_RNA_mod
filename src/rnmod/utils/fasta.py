from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

from pathlib import Path
from typing import Iterator

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def read_fasta(path: str | Path) -> dict[str, SeqRecord]:
    fasta_path = Path(path)
    if not fasta_path.exists():
        return {}
    return {record.id: record for record in SeqIO.parse(str(fasta_path), "fasta")}


def iter_fasta(path: str | Path) -> Iterator[SeqRecord]:
    fasta_path = Path(path)
    if not fasta_path.exists():
        return iter(())
    return SeqIO.parse(str(fasta_path), "fasta")


def clean_sequence(sequence: object) -> str:
    text = "" if sequence is None else str(sequence)
    return "".join(text.split()).upper()


def write_unique_fasta(rows: list[dict[str, object]], path: str | Path) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            sequence = clean_sequence(row.get("sequence", ""))
            seq_hash = str(row.get("sequence_sha256", ""))
            if not sequence or not seq_hash or seq_hash in seen:
                continue
            seen.add(seq_hash)
            header = str(row.get("protein_uid") or row.get("accession") or seq_hash[:16])
            gene = str(row.get("gene_name", ""))
            label_status = str(row.get("label_status", ""))
            handle.write(f">{header}|gene={gene}|label_status={label_status}|sha256={seq_hash}\n")
            for idx in range(0, len(sequence), 70):
                handle.write(sequence[idx : idx + 70] + "\n")
            count += 1
    return count

