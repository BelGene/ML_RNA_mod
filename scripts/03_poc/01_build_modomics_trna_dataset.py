#!/usr/bin/env python3
"""Build a small curated tRNA-modifier POC dataset from MODOMICS.

This script intentionally avoids broad discovery. It uses the MODOMICS protein
table filtered to RNA type=tRNA as the curated positive source, then fetches
protein sequences and basic metadata from UniProt for rows with UniProt accessions.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


DEFAULT_MODOMICS_URL = "https://genesilico.pl/modomics/proteins?species=all&enzyme_type=all&rna_type=tRNA"
DEFAULT_OUTPUT_DIR = "data/processed/poc"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FIELDS = [
    "accession",
    "reviewed",
    "protein_name",
    "gene_names",
    "organism_name",
    "organism_id",
    "length",
    "sequence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modomics-url", default=DEFAULT_MODOMICS_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-label-count", type=int, default=2)
    parser.add_argument("--uniprot-batch-size", type=int, default=50)
    parser.add_argument("--request-pause", type=float, default=0.2)
    parser.add_argument(
        "--keep-predicted-labels",
        action="store_true",
        help="Keep MODOMICS labels ending in 'predicted' as separate classes.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "-"} else text


def normalize_label(value: str, keep_predicted: bool = False) -> str:
    label = clean_text(value).lower()
    if not keep_predicted:
        label = re.sub(r"\s+predicted$", "", label)
    label = label.replace("alpha-amino-alpha-carboxypropyltransferase", "aminocarboxypropyltransferase")
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return label


def split_enzyme_type(value: object, keep_predicted: bool = False) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    parts = re.split(r"[,/]", text)
    labels = [normalize_label(part, keep_predicted=keep_predicted) for part in parts]
    return sorted({label for label in labels if label and label != "other"})


def load_modomics_table(url: str) -> pd.DataFrame:
    frames = pd.read_html(url)
    if not frames:
        raise RuntimeError(f"No tables found at {url}")
    table = frames[0].copy()
    table.columns = [re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") for column in table.columns]
    table = table.rename(
        columns={
            "id": "modomics_id",
            "traditional_name": "traditional_name",
            "full_name": "modomics_full_name",
            "uniprot": "accession",
            "enzyme_type": "modomics_enzyme_type",
            "organism": "modomics_organism",
        }
    )
    for column in table.columns:
        table[column] = table[column].map(clean_text)
    table["accession"] = table["accession"].str.split().str[0]
    table = table[table["accession"].ne("")].copy()
    return table


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_uniprot_table(accessions: list[str], batch_size: int, pause: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fields = ",".join(UNIPROT_FIELDS)
    for batch in chunks(accessions, batch_size):
        query = "(" + " OR ".join(f"accession:{accession}" for accession in batch) + ")"
        response = requests.get(
            UNIPROT_SEARCH_URL,
            params={"query": query, "format": "tsv", "fields": fields, "size": batch_size},
            timeout=120,
        )
        response.raise_for_status()
        text = response.text.strip()
        if text:
            frames.append(pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False))
        time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["Entry"])
    table = pd.concat(frames, ignore_index=True).drop_duplicates("Entry", keep="first")
    return table


def aggregate_labels(values: Iterable[object], keep_predicted: bool) -> list[str]:
    labels: set[str] = set()
    for value in values:
        labels.update(split_enzyme_type(value, keep_predicted=keep_predicted))
    return sorted(labels)


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def semicolon_join(values: Iterable[object]) -> str:
    cleaned = sorted({clean_text(value) for value in values if clean_text(value)})
    return ";".join(cleaned)


def build_dataset(modomics: pd.DataFrame, uniprot: pd.DataFrame, keep_predicted: bool, source_url: str) -> pd.DataFrame:
    uniprot = uniprot.rename(
        columns={
            "Entry": "accession",
            "Reviewed": "reviewed",
            "Protein names": "uniprot_protein_name",
            "Gene Names": "gene_names",
            "Organism": "organism",
            "Organism (ID)": "taxon_id",
            "Length": "sequence_length",
            "Sequence": "sequence",
        }
    )
    grouped_rows = []
    for accession, group in modomics.groupby("accession", sort=True):
        labels = aggregate_labels(group["modomics_enzyme_type"], keep_predicted=keep_predicted)
        grouped_rows.append(
            {
                "accession": accession,
                "modomics_ids": semicolon_join(group["modomics_id"]),
                "traditional_names": semicolon_join(group["traditional_name"]),
                "modomics_full_names": semicolon_join(group["modomics_full_name"]),
                "modomics_enzyme_types": semicolon_join(group["modomics_enzyme_type"]),
                "enzyme_type_labels": ";".join(labels),
                "target_rna_labels": "tRNA",
                "source_database": "MODOMICS",
                "source_url": source_url,
                "label_status": "curated_positive",
                "modomics_organisms": semicolon_join(group["modomics_organism"]),
            }
        )
    dataset = pd.DataFrame(grouped_rows)
    dataset = dataset.merge(uniprot, on="accession", how="left")
    dataset["protein_uid"] = "modomics:" + dataset["accession"].astype(str)
    dataset["has_sequence"] = dataset["sequence"].fillna("").astype(str).ne("")
    dataset["is_trainable_label"] = dataset["enzyme_type_labels"].fillna("").astype(str).ne("")
    dataset["label_source"] = "MODOMICS protein table RNA type=tRNA"
    columns = [
        "protein_uid",
        "accession",
        "reviewed",
        "uniprot_protein_name",
        "gene_names",
        "organism",
        "taxon_id",
        "sequence_length",
        "sequence",
        "has_sequence",
        "label_status",
        "target_rna_labels",
        "enzyme_type_labels",
        "modomics_enzyme_types",
        "traditional_names",
        "modomics_full_names",
        "modomics_ids",
        "modomics_organisms",
        "source_database",
        "source_url",
        "label_source",
        "is_trainable_label",
    ]
    for column in columns:
        if column not in dataset:
            dataset[column] = ""
    return dataset.reindex(columns=columns).sort_values("accession", kind="mergesort")


def label_matrix(dataset: pd.DataFrame, min_label_count: int) -> pd.DataFrame:
    all_labels = sorted(
        {
            label
            for labels in dataset["enzyme_type_labels"].fillna("").astype(str)
            for label in labels.split(";")
            if label
        }
    )
    label_values = dataset["enzyme_type_labels"].fillna("").astype(str)
    counts = {label: int(label_values.apply(lambda value, label=label: label in value.split(";")).sum()) for label in all_labels}
    kept = [label for label in all_labels if counts[label] >= min_label_count]
    matrix = dataset[["protein_uid", "accession", "enzyme_type_labels", "has_sequence", "is_trainable_label"]].copy()
    for label in kept:
        matrix[f"enzyme_type__{label}"] = dataset["enzyme_type_labels"].fillna("").astype(str).apply(
            lambda value, label=label: int(label in value.split(";"))
        )
    return matrix


def write_fasta(dataset: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset.to_dict(orient="records"):
            sequence = clean_text(row.get("sequence", ""))
            if not sequence:
                continue
            accession = clean_text(row.get("accession", ""))
            labels = clean_text(row.get("enzyme_type_labels", ""))
            name = clean_text(row.get("traditional_names", "")) or accession
            handle.write(f">{accession}|labels={labels}|name={name}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_dataset_card(dataset: pd.DataFrame, matrix: pd.DataFrame, output: Path, source_url: str) -> None:
    label_columns = [column for column in matrix.columns if column.startswith("enzyme_type__")]
    counts = {column.removeprefix("enzyme_type__"): int(matrix[column].sum()) for column in label_columns}
    payload = {
        "source": "MODOMICS protein table filtered to RNA type=tRNA",
        "source_url": source_url,
        "n_rows": int(len(dataset)),
        "n_with_sequences": int(dataset["has_sequence"].astype(bool).sum()),
        "n_with_trainable_enzyme_type_labels": int(dataset["is_trainable_label"].astype(bool).sum()),
        "label_counts": counts,
    }
    lines = [
        "# MODOMICS tRNA Protein POC Dataset",
        "",
        "This proof-of-concept dataset uses MODOMICS as the curated positive source.",
        "Rows are filtered to RNA type=tRNA and deduplicated by UniProt accession.",
        "The prediction target is the MODOMICS enzyme type label.",
        "",
        f"- Source URL: {source_url}",
        f"- Proteins with UniProt accessions: {payload['n_rows']}",
        f"- Proteins with fetched UniProt sequences: {payload['n_with_sequences']}",
        f"- Proteins with enzyme type labels: {payload['n_with_trainable_enzyme_type_labels']}",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## JSON Summary", "", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modomics = load_modomics_table(args.modomics_url)
    accessions = sorted(modomics["accession"].dropna().astype(str).unique())
    uniprot = fetch_uniprot_table(accessions, batch_size=args.uniprot_batch_size, pause=args.request_pause)
    dataset = build_dataset(modomics, uniprot, keep_predicted=args.keep_predicted_labels, source_url=args.modomics_url)
    matrix = label_matrix(dataset, min_label_count=args.min_label_count)

    dataset_path = output_dir / "modomics_trna_proteins.tsv"
    matrix_path = output_dir / "modomics_trna_label_matrix.tsv"
    fasta_path = output_dir / "modomics_trna_sequences.faa"
    card_path = output_dir / "modomics_trna_dataset_card.md"

    dataset.to_csv(dataset_path, sep="\t", index=False)
    matrix.to_csv(matrix_path, sep="\t", index=False)
    write_fasta(dataset, fasta_path)
    write_dataset_card(dataset, matrix, card_path, args.modomics_url)

    print(f"wrote {len(dataset)} proteins to {dataset_path}")
    print(f"wrote {len(matrix.columns) - 5} label columns to {matrix_path}")
    print(f"wrote FASTA to {fasta_path}")
    print(f"wrote dataset card to {card_path}")


if __name__ == "__main__":
    main()
