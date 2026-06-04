from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# Routine POC settings are exposed by scripts/03_poc/03_build_weak_uniprot_trna_dataset.py.
# Keep automated weak-label rules here so they can be tested and reused.

import json
import re
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from rnmod.utils.fasta import clean_sequence
from rnmod.utils.tables import join_labels, split_labels


DEFAULT_MODOMICS_URL = "https://iimcb.genesilico.pl/modomics/proteins?species=all&enzyme_type=all&rna_type=tRNA"
DEFAULT_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

UNIPROT_FIELDS = [
    "accession",
    "reviewed",
    "id",
    "protein_name",
    "gene_names",
    "organism_name",
    "organism_id",
    "lineage",
    "length",
    "sequence",
    "go_id",
    "ec",
    "rhea",
    "keyword",
    "cc_function",
]

TAXONOMY_SCOPES = {
    "all": "",
    "bacteria": "taxonomy_id:2",
    "archaea": "taxonomy_id:2157",
    "bacteria_archaea": "(taxonomy_id:2 OR taxonomy_id:2157)",
}

TRAINABLE_STATUSES = {"gold_positive", "weak_positive", "hard_negative", "easy_negative"}

OUTPUT_COLUMNS = [
    "protein_uid",
    "accession",
    "reviewed",
    "uniprot_id",
    "protein_name",
    "gene_names",
    "organism",
    "taxon_id",
    "lineage",
    "sequence_length",
    "sequence",
    "has_sequence",
    "label_status",
    "is_trna_modifier",
    "is_trainable",
    "train_exclusion_reason",
    "negative_type",
    "mechanism_labels",
    "target_rna_labels",
    "label_rule_ids",
    "label_confidence",
    "source_databases",
    "source_query_ids",
    "source_label_hints",
    "modomics_ids",
    "modomics_traditional_names",
    "modomics_full_names",
    "modomics_enzyme_types",
    "modomics_organisms",
    "go_ids",
    "ec_numbers",
    "rhea_ids",
    "keywords",
    "function_comment",
    "source_url",
    "provenance_json",
]


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    label_hint: str
    query: str


@dataclass(frozen=True)
class BuildSummary:
    dataset_path: Path
    fasta_path: Path
    label_matrix_path: Path
    dataset_card_path: Path
    n_rows: int
    n_trainable: int
    n_sequences: int


POSITIVE_QUERY_SPECS = [
    QuerySpec(
        "trna_methyltransferase",
        "weak_positive",
        '(reviewed:true) AND (protein_name:tRNA) AND (protein_name:methyltransferase OR protein_name:methylase)',
    ),
    QuerySpec(
        "trna_pseudouridine_synthase",
        "weak_positive",
        '(reviewed:true) AND (protein_name:tRNA) AND (protein_name:pseudouridine OR protein_name:pseudouridylate)',
    ),
    QuerySpec(
        "trna_thiolation",
        "weak_positive",
        '(reviewed:true) AND (protein_name:tRNA) AND (protein_name:thiouridylase OR protein_name:thiolation OR protein_name:thiouridine)',
    ),
    QuerySpec(
        "trna_deaminase",
        "weak_positive",
        '(reviewed:true) AND (protein_name:tRNA) AND (protein_name:deaminase)',
    ),
    QuerySpec(
        "trna_queuosine_wyosine",
        "weak_positive",
        '(reviewed:true) AND (protein_name:tRNA) AND (protein_name:queuosine OR protein_name:wybutosine OR protein_name:wyosine)',
    ),
    QuerySpec(
        "trna_t6a_dihydrouridine",
        "weak_positive",
        '(reviewed:true) AND (protein_name:tRNA) AND (protein_name:threonylcarbamoyladenosine OR protein_name:dihydrouridine)',
    ),
]

HARD_NEGATIVE_QUERY_SPECS = [
    QuerySpec(
        "negative_dna_methyltransferase",
        "hard_negative",
        '(reviewed:true) AND (protein_name:"DNA methyltransferase" OR protein_name:"DNA adenine methyltransferase" OR gene:dam OR gene:dcm) NOT (protein_name:tRNA)',
    ),
    QuerySpec(
        "negative_protein_methyltransferase",
        "hard_negative",
        '(reviewed:true) AND (protein_name:"protein methyltransferase") NOT (protein_name:tRNA)',
    ),
    QuerySpec(
        "negative_nontrna_deaminase",
        "hard_negative",
        '(reviewed:true) AND (protein_name:"cytidine deaminase" OR protein_name:"nucleotide deaminase") NOT (protein_name:tRNA)',
    ),
    QuerySpec(
        "negative_rrna_modifier",
        "hard_negative",
        '(reviewed:true) AND (protein_name:rRNA) AND (protein_name:methyltransferase OR protein_name:pseudouridine OR protein_name:pseudouridylate) NOT (protein_name:tRNA)',
    ),
]

EASY_NEGATIVE_QUERY_SPEC = QuerySpec(
    "negative_easy_reviewed_nonrna",
    "easy_negative",
    '(reviewed:true) NOT (protein_name:RNA OR protein_name:tRNA OR protein_name:rRNA OR protein_name:methyltransferase OR protein_name:pseudouridine OR protein_name:deaminase OR keyword:"RNA modification")',
)

MECHANISM_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "methyltransferase",
        (
            "methyltransferase",
            "methylase",
            "methyltransfer",
            "2'-o-methyl",
            "c5-methyl",
            "n1-methyl",
            "n2-methyl",
            "n3-methyl",
            "n7-methyl",
        ),
    ),
    (
        "pseudouridine_synthase",
        (
            "pseudouridine synthase",
            "pseudouridylate synthase",
            "pseudouridyl",
            "uridine isomerase",
        ),
    ),
    (
        "deaminase",
        (
            "deaminase",
            "adenosine deaminase",
            "cytidine deaminase",
            "tada",
            "adat",
        ),
    ),
    (
        "thiolation",
        (
            "thiouridylase",
            "thiolation",
            "thiouridine",
            "2-thio",
            "mnma",
            "ttua",
            "ncsa",
        ),
    ),
    (
        "queuosine_pathway",
        (
            "queuosine",
            "preq",
            "queuine",
            "tRNA-guanine transglycosylase".lower(),
            "transglycosylase",
            "archaeosine",
            "tgt",
        ),
    ),
    (
        "wyosine_pathway",
        (
            "wybutosine",
            "wyosine",
            "yW-synthesizing".lower(),
            "tyw",
        ),
    ),
    (
        "t6a_pathway",
        (
            "threonylcarbamoyladenosine",
            "t(6)a",
            "t6a",
            "tsa",
            "keops",
            "sua5",
            "yrdc",
        ),
    ),
    (
        "dihydrouridine_synthase",
        (
            "dihydrouridine synthase",
            "dus",
        ),
    ),
    (
        "acetyltransferase",
        (
            "acetyltransferase",
            "acetylase",
        ),
    ),
]


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "-"} else text


def normalize_label(value: str) -> str:
    label = clean_text(value).lower()
    label = re.sub(r"\s+predicted$", "", label)
    label = label.replace("alpha-amino-alpha-carboxypropyltransferase", "aminocarboxypropyltransferase")
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return label


def split_modomics_enzyme_type(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    labels = [normalize_label(part) for part in re.split(r"[,/]", text)]
    return sorted({label for label in labels if label and label != "other"})


def semicolon_join(values: Iterable[object]) -> str:
    return join_labels(clean_text(value) for value in values)


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def _safe_bool(value: object) -> bool:
    return clean_text(value).lower() in {"true", "yes", "reviewed", "1"}


def _normalize_table_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out.columns = [re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") for column in out.columns]
    return out


def _column(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row:
            return clean_text(row.get(name, ""))
    return ""


def _frame_from_tsv(text: str) -> pd.DataFrame:
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame()
    return pd.read_csv(StringIO(stripped), sep="\t", dtype=str, keep_default_na=False)


def _raise_for_uniprot_status(response: requests.Response, query_id: str = "") -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = response.text.strip().splitlines()[-1:] or [str(exc)]
        prefix = f" for {query_id}" if query_id else ""
        raise RuntimeError(f"UniProt request failed{prefix}: {message[0]}") from exc


def apply_taxonomy_scope(query: str, taxonomy_scope: str) -> str:
    scope = TAXONOMY_SCOPES[taxonomy_scope]
    if not scope:
        return query
    return f"({query}) AND ({scope})"


def iter_batches(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_uniprot_queries(
    query_specs: list[QuerySpec],
    taxonomy_scope: str,
    endpoint: str = DEFAULT_UNIPROT_SEARCH_URL,
    page_size: int = 500,
    max_records_per_query: int = 5000,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    fields = ",".join(UNIPROT_FIELDS)
    frames: list[pd.DataFrame] = []
    for spec in query_specs:
        query = apply_taxonomy_scope(spec.query, taxonomy_scope)
        fetched = 0
        next_url: str | None = endpoint
        params = {"query": query, "format": "tsv", "fields": fields, "size": page_size}
        while next_url:
            if next_url == endpoint:
                response = requests.get(endpoint, params=params, timeout=120)
            else:
                response = requests.get(next_url, timeout=120)
            _raise_for_uniprot_status(response, spec.query_id)
            frame = _frame_from_tsv(response.text)
            if not frame.empty:
                remaining = max_records_per_query - fetched
                frame = frame.head(remaining).copy()
                frame["rnmod_query_id"] = spec.query_id
                frame["rnmod_label_hint"] = spec.label_hint
                frames.append(frame)
                fetched += len(frame)
            if fetched >= max_records_per_query:
                break
            next_url = response.links.get("next", {}).get("url")
            time.sleep(pause_seconds)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_uniprot_by_accessions(
    accessions: list[str],
    endpoint: str = DEFAULT_UNIPROT_SEARCH_URL,
    batch_size: int = 50,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    fields = ",".join(UNIPROT_FIELDS)
    frames: list[pd.DataFrame] = []
    for batch in iter_batches(accessions, batch_size):
        query = "(" + " OR ".join(f"accession:{accession}" for accession in batch) + ")"
        response = requests.get(
            endpoint,
            params={"query": query, "format": "tsv", "fields": fields, "size": batch_size},
            timeout=120,
        )
        _raise_for_uniprot_status(response)
        frame = _frame_from_tsv(response.text)
        if not frame.empty:
            frames.append(frame)
        time.sleep(pause_seconds)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("Entry", keep="first")


def normalize_uniprot_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    rows: list[dict[str, object]] = []
    for _, row in table.iterrows():
        accession = _column(row, "Entry", "accession", "Accession")
        if not accession:
            continue
        rows.append(
            {
                "accession": accession,
                "reviewed": _safe_bool(_column(row, "Reviewed", "reviewed")),
                "uniprot_id": _column(row, "Entry Name", "id", "entry_name"),
                "protein_name": _column(row, "Protein names", "protein_name"),
                "gene_names": _column(row, "Gene Names", "gene_names"),
                "organism": _column(row, "Organism", "organism_name"),
                "taxon_id": _column(row, "Organism (ID)", "organism_id"),
                "lineage": _column(row, "Taxonomic lineage", "lineage"),
                "sequence_length": _column(row, "Length", "length"),
                "sequence": clean_sequence(_column(row, "Sequence", "sequence")),
                "go_ids": _column(row, "Gene Ontology IDs", "go_id"),
                "ec_numbers": _column(row, "EC number", "ec"),
                "rhea_ids": _column(row, "Rhea IDs", "rhea"),
                "keywords": _column(row, "Keywords", "keyword"),
                "function_comment": _column(row, "Function [CC]", "cc_function"),
                "source_databases": "UniProtKB/Swiss-Prot",
                "source_query_ids": _column(row, "rnmod_query_id"),
                "source_label_hints": _column(row, "rnmod_label_hint"),
                "source_url": f"https://www.uniprot.org/uniprotkb/{accession}/entry",
                "is_modomics_anchor": False,
            }
        )
    return pd.DataFrame(rows)


def load_modomics_anchor_table(url: str = DEFAULT_MODOMICS_URL) -> pd.DataFrame:
    frames = pd.read_html(url)
    if not frames:
        raise RuntimeError(f"No MODOMICS protein tables found at {url}")
    table = _normalize_table_columns(frames[0])
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
    return table[table["accession"].ne("")].copy()


def build_modomics_anchor_records(
    modomics_url: str,
    batch_size: int,
    pause_seconds: float,
) -> pd.DataFrame:
    modomics = load_modomics_anchor_table(modomics_url)
    accessions = sorted(modomics["accession"].dropna().astype(str).unique())
    uniprot = normalize_uniprot_rows(fetch_uniprot_by_accessions(accessions, batch_size=batch_size, pause_seconds=pause_seconds))
    grouped_rows: list[dict[str, object]] = []
    for accession, group in modomics.groupby("accession", sort=True):
        labels: set[str] = set()
        for enzyme_type in group.get("modomics_enzyme_type", []):
            labels.update(split_modomics_enzyme_type(enzyme_type))
        grouped_rows.append(
            {
                "accession": accession,
                "modomics_ids": semicolon_join(group.get("modomics_id", [])),
                "modomics_traditional_names": semicolon_join(group.get("traditional_name", [])),
                "modomics_full_names": semicolon_join(group.get("modomics_full_name", [])),
                "modomics_enzyme_types": semicolon_join(group.get("modomics_enzyme_type", [])),
                "modomics_organisms": semicolon_join(group.get("modomics_organism", [])),
                "modomics_enzyme_type_labels": join_labels(labels),
                "source_databases": "MODOMICS",
                "source_url": modomics_url,
                "is_modomics_anchor": True,
            }
        )
    anchors = pd.DataFrame(grouped_rows)
    if uniprot.empty:
        return anchors
    merged = anchors.merge(
        uniprot.drop(columns=["source_databases", "source_url", "is_modomics_anchor"], errors="ignore"),
        on="accession",
        how="left",
    )
    has_uniprot_metadata = merged["sequence"].fillna("").astype(str).ne("") | merged["uniprot_id"].fillna("").astype(str).ne("")
    merged.loc[has_uniprot_metadata, "source_databases"] = "MODOMICS;UniProtKB/Swiss-Prot"
    merged["source_url"] = modomics_url
    merged["is_modomics_anchor"] = True
    return merged


def _aggregate_records(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["is_modomics_anchor", "modomics_enzyme_type_labels"])

    set_fields = [
        "source_databases",
        "source_query_ids",
        "source_label_hints",
        "modomics_ids",
        "modomics_traditional_names",
        "modomics_full_names",
        "modomics_enzyme_types",
        "modomics_organisms",
        "go_ids",
        "ec_numbers",
        "rhea_ids",
        "keywords",
    ]
    first_fields = [
        "reviewed",
        "uniprot_id",
        "protein_name",
        "gene_names",
        "organism",
        "taxon_id",
        "lineage",
        "sequence_length",
        "sequence",
        "function_comment",
        "source_url",
        "modomics_enzyme_type_labels",
    ]

    rows: list[dict[str, object]] = []
    for accession, group in table.groupby("accession", sort=True):
        row: dict[str, object] = {"accession": accession}
        for field in first_fields:
            row[field] = first_nonempty(group[field]) if field in group else ""
        for field in set_fields:
            values: list[str] = []
            if field in group:
                for value in group[field]:
                    values.extend(split_labels(value))
            row[field] = join_labels(values)
        row["reviewed"] = bool(any(_safe_bool(value) for value in group.get("reviewed", [])))
        row["is_modomics_anchor"] = bool(any(_safe_bool(value) for value in group.get("is_modomics_anchor", [])))
        rows.append(row)
    return pd.DataFrame(rows)


def _combined_text(row: dict[str, object]) -> str:
    parts = [
        row.get("protein_name", ""),
        row.get("gene_names", ""),
        row.get("modomics_traditional_names", ""),
        row.get("modomics_full_names", ""),
        row.get("modomics_enzyme_types", ""),
        row.get("keywords", ""),
        row.get("function_comment", ""),
    ]
    return " ".join(clean_text(part).lower() for part in parts if clean_text(part))


def infer_mechanism_labels(row: dict[str, object]) -> tuple[list[str], list[str]]:
    text = _combined_text(row)
    labels: list[str] = []
    rules: list[str] = []
    for label, patterns in MECHANISM_RULES:
        if any(pattern in text for pattern in patterns):
            labels.append(label)
            rules.append(f"mechanism:{label}")

    modomics_labels = split_labels(row.get("modomics_enzyme_type_labels", ""))
    for modomics_label in modomics_labels:
        if modomics_label == "methyltransferase":
            labels.append("methyltransferase")
            rules.append("mechanism:modomics_methyltransferase")
        elif modomics_label == "pseudouridine_synthase":
            labels.append("pseudouridine_synthase")
            rules.append("mechanism:modomics_pseudouridine_synthase")
        elif modomics_label in {"deaminase", "thiolase", "sulfurtransferase", "acetyltransferase"}:
            mapped = "thiolation" if modomics_label in {"thiolase", "sulfurtransferase"} else modomics_label
            labels.append(mapped)
            rules.append(f"mechanism:modomics_{modomics_label}")
        elif modomics_label in {"transglycosylase", "amidinotransferase", "agmatidine_synthase"}:
            labels.append("queuosine_pathway")
            rules.append(f"mechanism:modomics_{modomics_label}")
        elif modomics_label in {"threonylcarbamoyladenosine_synthetase", "threonylcarbamoyltransferase", "kinase"}:
            labels.append("t6a_pathway")
            rules.append(f"mechanism:modomics_{modomics_label}")
        elif modomics_label == "dihydrouridine_synthase":
            labels.append("dihydrouridine_synthase")
            rules.append("mechanism:modomics_dihydrouridine_synthase")
    return sorted(set(labels)), sorted(set(rules))


def classify_record(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    hints = set(split_labels(out.get("source_label_hints", "")))
    query_ids = set(split_labels(out.get("source_query_ids", "")))
    is_modomics = bool(out.get("is_modomics_anchor"))
    sequence = clean_sequence(out.get("sequence", ""))
    text = _combined_text(out)
    has_trna_context = "trna" in text or "transfer rna" in text
    positive_hint = "weak_positive" in hints
    hard_negative_hint = "hard_negative" in hints
    easy_negative_hint = "easy_negative" in hints

    mechanism_labels, mechanism_rules = infer_mechanism_labels(out)
    label_rules: list[str] = list(mechanism_rules)
    negative_type = ""
    confidence = 0.0

    if is_modomics:
        label_status = "gold_positive"
        is_trna_modifier = True
        confidence = 1.0
        label_rules.append("positive:modomics_trna_table")
    elif positive_hint and hard_negative_hint:
        label_status = "conflicted"
        is_trna_modifier = False
        confidence = 0.0
        label_rules.append("exclude:positive_and_negative_query_match")
    elif positive_hint and has_trna_context:
        label_status = "weak_positive"
        is_trna_modifier = True
        confidence = 0.70
        label_rules.append("positive:reviewed_uniprot_trna_text_query")
    elif hard_negative_hint:
        label_status = "hard_negative"
        is_trna_modifier = False
        confidence = 0.80
        negative_type = "hard_negative_near_family"
        label_rules.append("negative:reviewed_uniprot_near_family_query")
    elif easy_negative_hint:
        label_status = "easy_negative"
        is_trna_modifier = False
        confidence = 0.55
        negative_type = "easy_negative_reviewed_nonrna"
        label_rules.append("negative:reviewed_uniprot_nonrna_query")
    else:
        label_status = "unknown"
        is_trna_modifier = False
        confidence = 0.0
        label_rules.append("exclude:no_usable_automatic_label")

    is_trainable = label_status in TRAINABLE_STATUSES
    train_exclusion_reason = ""
    if not is_trainable:
        train_exclusion_reason = f"status={label_status}"
    if not sequence:
        is_trainable = False
        train_exclusion_reason = "missing_sequence"

    if not is_trna_modifier:
        mechanism_labels = []
    if is_trna_modifier and not mechanism_labels:
        label_rules.append("mechanism:unknown")

    out["protein_uid"] = f"weak_trna:{out.get('accession', '')}"
    out["sequence"] = sequence
    out["has_sequence"] = bool(sequence)
    out["label_status"] = label_status
    out["is_trna_modifier"] = bool(is_trna_modifier)
    out["is_trainable"] = bool(is_trainable)
    out["train_exclusion_reason"] = train_exclusion_reason
    out["negative_type"] = negative_type
    out["mechanism_labels"] = join_labels(mechanism_labels)
    out["target_rna_labels"] = "tRNA" if is_trna_modifier else "non_tRNA"
    out["label_rule_ids"] = join_labels(label_rules)
    out["label_confidence"] = confidence
    out["provenance_json"] = json.dumps(
        {
            "source_query_ids": sorted(query_ids),
            "source_label_hints": sorted(hints),
            "is_modomics_anchor": is_modomics,
            "has_trna_context": has_trna_context,
        },
        sort_keys=True,
    )
    return out


def _length_bin(length: object) -> str:
    try:
        value = int(float(clean_text(length)))
    except ValueError:
        return "unknown"
    if value < 150:
        return "lt150"
    if value < 250:
        return "150_249"
    if value < 350:
        return "250_349"
    if value < 500:
        return "350_499"
    if value < 750:
        return "500_749"
    if value < 1000:
        return "750_999"
    return "gte1000"


def select_length_matched_easy_negatives(
    easy_negatives: pd.DataFrame,
    positives: pd.DataFrame,
    target_count: int,
    random_seed: int,
) -> pd.DataFrame:
    if easy_negatives.empty or target_count <= 0:
        return easy_negatives.head(0).copy()
    target_count = min(target_count, len(easy_negatives))
    if positives.empty:
        return easy_negatives.sample(n=target_count, random_state=random_seed).sort_values("accession", kind="mergesort")

    candidates = easy_negatives.copy()
    reference = positives.copy()
    candidates["_length_bin"] = candidates["sequence_length"].map(_length_bin)
    reference["_length_bin"] = reference["sequence_length"].map(_length_bin)
    proportions = reference["_length_bin"].value_counts(normalize=True)

    selected_parts: list[pd.DataFrame] = []
    selected_accessions: set[str] = set()
    remaining = target_count
    for idx, (length_bin, proportion) in enumerate(proportions.items()):
        if remaining <= 0:
            break
        desired = int(round(float(proportion) * target_count))
        if idx == len(proportions) - 1:
            desired = remaining
        desired = max(0, min(desired, remaining))
        pool = candidates[candidates["_length_bin"].eq(length_bin) & ~candidates["accession"].isin(selected_accessions)]
        if pool.empty or desired == 0:
            continue
        sample_n = min(desired, len(pool))
        sample = pool.sample(n=sample_n, random_state=random_seed + idx)
        selected_parts.append(sample)
        selected_accessions.update(sample["accession"].astype(str))
        remaining -= sample_n

    if remaining > 0:
        pool = candidates[~candidates["accession"].isin(selected_accessions)]
        if not pool.empty:
            selected_parts.append(pool.sample(n=min(remaining, len(pool)), random_state=random_seed + 1000))

    if not selected_parts:
        return candidates.head(0).drop(columns=["_length_bin"], errors="ignore")
    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.drop(columns=["_length_bin"], errors="ignore").sort_values("accession", kind="mergesort")


def build_label_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "protein_uid",
        "accession",
        "label_status",
        "is_trainable",
        "is_trna_modifier",
        "mechanism_labels",
        "negative_type",
        "source_query_ids",
        "label_rule_ids",
    ]
    matrix = dataset[base_columns].copy()
    matrix["target__trna_modifier"] = dataset["is_trna_modifier"].astype(bool).astype(int)

    mechanism_labels = sorted(
        {
            label
            for value in dataset.loc[dataset["is_trna_modifier"].astype(bool), "mechanism_labels"]
            for label in split_labels(value)
        }
    )
    for label in mechanism_labels:
        matrix[f"mechanism__{label}"] = dataset["mechanism_labels"].apply(lambda value, label=label: int(label in split_labels(value)))
    matrix["has_mechanism_label"] = dataset["mechanism_labels"].astype(str).ne("").astype(int)
    return matrix


def write_fasta(dataset: pd.DataFrame, output: str | Path) -> int:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in dataset[dataset["is_trainable"].astype(bool)].to_dict(orient="records"):
            sequence = clean_sequence(row.get("sequence", ""))
            if not sequence:
                continue
            header = "|".join(
                [
                    clean_text(row.get("accession", "")),
                    f"status={clean_text(row.get('label_status', ''))}",
                    f"target={int(bool(row.get('is_trna_modifier')))}",
                    f"mechanisms={clean_text(row.get('mechanism_labels', '')) or 'none'}",
                ]
            )
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
            count += 1
    return count


def write_dataset_card(
    dataset: pd.DataFrame,
    label_matrix: pd.DataFrame,
    output: str | Path,
    parameters: dict[str, object],
) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainable = dataset[dataset["is_trainable"].astype(bool)]
    positives = trainable[trainable["is_trna_modifier"].astype(bool)]
    negatives = trainable[~trainable["is_trna_modifier"].astype(bool)]

    def counts(column: str, frame: pd.DataFrame = dataset) -> dict[str, int]:
        if column not in frame or frame.empty:
            return {}
        return {str(key): int(value) for key, value in frame[column].replace("", "missing").value_counts().sort_index().items()}

    mechanism_counts = {
        column.removeprefix("mechanism__"): int(label_matrix[column].sum())
        for column in label_matrix.columns
        if column.startswith("mechanism__")
    }
    query_counts: dict[str, int] = {}
    for value in dataset["source_query_ids"]:
        for query_id in split_labels(value):
            query_counts[query_id] = query_counts.get(query_id, 0) + 1

    payload = {
        "n_rows": int(len(dataset)),
        "n_trainable": int(len(trainable)),
        "n_trainable_positives": int(len(positives)),
        "n_trainable_negatives": int(len(negatives)),
        "n_with_sequences": int(dataset["has_sequence"].astype(bool).sum()),
        "label_status_counts": counts("label_status"),
        "negative_type_counts": counts("negative_type", negatives),
        "mechanism_counts": mechanism_counts,
        "query_counts": dict(sorted(query_counts.items())),
        "parameters": parameters,
    }

    lines = [
        "# Weak UniProt tRNA Modifier POC Dataset",
        "",
        "This dataset is an automated proof-of-concept training set for testing whether protein embeddings can separate tRNA modification proteins from non-tRNA controls.",
        "",
        "Labels are intentionally weak: positives come from MODOMICS tRNA anchors and reviewed UniProt protein-name queries; negatives come from reviewed near-family and non-RNA queries. No manual curation is applied.",
        "",
        "## Summary",
        "",
        f"- Rows: {payload['n_rows']}",
        f"- Trainable rows: {payload['n_trainable']}",
        f"- Trainable positives: {payload['n_trainable_positives']}",
        f"- Trainable negatives: {payload['n_trainable_negatives']}",
        f"- Rows with sequences: {payload['n_with_sequences']}",
        "",
        "## Label Status Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in payload["label_status_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Mechanism Label Counts", "", "| Label | Count |", "|---|---:|"])
    for label, count in sorted(mechanism_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Source Query Counts", "", "| Query | Count |", "|---|---:|"])
    for query_id, count in payload["query_counts"].items():
        lines.append(f"| {query_id} | {count} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is not a curated benchmark. It is a weak-label POC source for deciding whether the embedding approach has signal.",
            "- Protein-name queries miss some valid proteins and may include annotation errors.",
            "- Easy negatives are reviewed non-RNA proteins and may make the binary task easier than real discovery.",
            "- Cluster-based train/test splits are still required before interpreting model performance.",
            "",
            "## JSON Summary",
            "",
            "```json",
            json.dumps(payload, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(
    output_dir: str | Path,
    modomics_url: str = DEFAULT_MODOMICS_URL,
    dataset_profile: str = "",
    taxonomy_scope: str = "all",
    include_modomics: bool = True,
    page_size: int = 500,
    max_records_per_query: int = 5000,
    max_easy_negative_pool: int = 5000,
    easy_negative_ratio: float = 1.0,
    max_easy_negatives: int = 2000,
    uniprot_batch_size: int = 50,
    request_pause: float = 0.2,
    random_seed: int = 42,
) -> BuildSummary:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    query_frames = [
        fetch_uniprot_queries(
            POSITIVE_QUERY_SPECS + HARD_NEGATIVE_QUERY_SPECS,
            taxonomy_scope=taxonomy_scope,
            page_size=page_size,
            max_records_per_query=max_records_per_query,
            pause_seconds=request_pause,
        )
    ]
    easy_frame = fetch_uniprot_queries(
        [EASY_NEGATIVE_QUERY_SPEC],
        taxonomy_scope=taxonomy_scope,
        page_size=page_size,
        max_records_per_query=max_easy_negative_pool,
        pause_seconds=request_pause,
    )
    query_frames.append(easy_frame)
    uniprot_records = normalize_uniprot_rows(pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame())

    frames = [uniprot_records]
    if include_modomics:
        frames.append(build_modomics_anchor_records(modomics_url, batch_size=uniprot_batch_size, pause_seconds=request_pause))

    combined = _aggregate_records(pd.concat(frames, ignore_index=True, sort=False))
    classified = pd.DataFrame([classify_record(row) for row in combined.to_dict(orient="records")])
    if classified.empty:
        classified = pd.DataFrame(columns=OUTPUT_COLUMNS)

    positives = classified[classified["is_trainable"].astype(bool) & classified["is_trna_modifier"].astype(bool)]
    easy = classified[classified["label_status"].eq("easy_negative") & classified["is_trainable"].astype(bool)]
    target_easy_count = min(max_easy_negatives, int(round(len(positives) * float(easy_negative_ratio))))
    selected_easy = select_length_matched_easy_negatives(easy, positives, target_easy_count, random_seed=random_seed)
    keep_accessions = set(selected_easy["accession"].astype(str))
    keep = ~classified["label_status"].eq("easy_negative") | classified["accession"].astype(str).isin(keep_accessions)
    classified = classified.loc[keep].copy()
    classified = classified.reindex(columns=OUTPUT_COLUMNS)
    classified = classified.sort_values(["label_status", "accession"], kind="mergesort")

    label_matrix = build_label_matrix(classified)

    dataset_path = output_root / "weak_trna_mod_proteins.tsv"
    fasta_path = output_root / "weak_trna_mod_sequences.faa"
    label_matrix_path = output_root / "weak_trna_mod_label_matrix.tsv"
    dataset_card_path = output_root / "weak_trna_mod_dataset_card.md"

    classified.to_csv(dataset_path, sep="\t", index=False)
    label_matrix.to_csv(label_matrix_path, sep="\t", index=False)
    n_sequences = write_fasta(classified, fasta_path)
    write_dataset_card(
        classified,
        label_matrix,
        dataset_card_path,
        parameters={
            "dataset_profile": dataset_profile,
            "modomics_url": modomics_url,
            "taxonomy_scope": taxonomy_scope,
            "include_modomics": include_modomics,
            "page_size": page_size,
            "max_records_per_query": max_records_per_query,
            "max_easy_negative_pool": max_easy_negative_pool,
            "easy_negative_ratio": easy_negative_ratio,
            "max_easy_negatives": max_easy_negatives,
            "random_seed": random_seed,
        },
    )
    return BuildSummary(
        dataset_path=dataset_path,
        fasta_path=fasta_path,
        label_matrix_path=label_matrix_path,
        dataset_card_path=dataset_card_path,
        n_rows=len(classified),
        n_trainable=int(classified["is_trainable"].astype(bool).sum()),
        n_sequences=n_sequences,
    )
