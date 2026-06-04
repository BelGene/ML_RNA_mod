from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rnmod.dataset.dataset_cards import make_dataset_card
from rnmod.labels.build_negatives import enforce_negative_policy
from rnmod.schemas import CONTROLLED_VOCAB, MASTER_COLUMNS
from rnmod.settings import interim_path, load_config, processed_path
from rnmod.utils.checksums import file_md5, file_sha256
from rnmod.utils.fasta import write_unique_fasta
from rnmod.utils.tables import read_table, split_labels, write_parquet, write_tsv


PROTEIN_INTERIM_FILES = {
    "uniprot": "uniprot_records.parquet",
    "modomics": "modomics_records.parquet",
    "ecocyc": "ecocyc_records.parquet",
    "manual_literature": "manual_literature_records.parquet",
    "legacy_pilot": "legacy_pilot_records.parquet",
}

NONPROTEIN_INTERIM_FILES = {
    "rhea": "rhea_reactions.parquet",
    "go": "go_terms.parquet",
}


def protein_interim_paths(config: dict[str, Any]) -> list[Path]:
    return [interim_path(config, source, filename) for source, filename in PROTEIN_INTERIM_FILES.items()]


def nonprotein_interim_paths(config: dict[str, Any]) -> list[Path]:
    return [interim_path(config, source, filename) for source, filename in NONPROTEIN_INTERIM_FILES.items()]


def _refresh_training_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_rna_modifier"] = out["label_status"].isin(["gold_positive", "silver_positive", "bronze_candidate"])
    out["is_train_positive"] = out["label_status"].isin(["gold_positive", "silver_positive"])
    out["is_train_negative"] = out["label_status"].eq("hard_negative")
    out["is_trainable"] = out["is_train_positive"] | out["is_train_negative"]
    out["train_exclusion_reason"] = ""
    out.loc[~out["is_trainable"], "train_exclusion_reason"] = "status=" + out.loc[~out["is_trainable"], "label_status"].astype(str)
    out.loc[out["is_fragment"].astype(str).str.lower().eq("true"), ["is_trainable", "train_exclusion_reason"]] = [
        False,
        "fragmentary_sequence",
    ]
    out.loc[out["is_hypothetical"].astype(str).str.lower().eq("true"), ["is_trainable", "train_exclusion_reason"]] = [
        False,
        "hypothetical_or_unknown_product",
    ]
    out.loc[out["sequence"].astype(str).eq(""), ["is_trainable", "train_exclusion_reason"]] = [False, "missing_sequence"]
    return out


def _load_protein_tables(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        table = read_table(path)
        if table.empty:
            continue
        missing = [column for column in MASTER_COLUMNS if column not in table.columns]
        for column in missing:
            table[column] = ""
        frames.append(table.reindex(columns=MASTER_COLUMNS))
    if not frames:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["source_database", "source_record_id", "accession", "sequence_sha256"],
        keep="first",
    )
    return merged


def _add_sequence_representatives(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.sort_values(["source_database", "accession", "source_record_id", "protein_uid"], kind="mergesort")
    valid = out["sequence_sha256"].astype(str).ne("")
    counts = out.loc[valid].groupby("sequence_sha256")["protein_uid"].transform("count")
    representatives = out.loc[valid].groupby("sequence_sha256")["protein_uid"].transform("first")
    out.loc[valid, "duplicate_sequence_count"] = counts.astype(int)
    out.loc[valid, "representative_uid"] = representatives
    out.loc[~valid, "duplicate_sequence_count"] = 0
    out.loc[~valid, "representative_uid"] = out.loc[~valid, "protein_uid"]
    return out


def _label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "protein_uid",
        "accession",
        "sequence_sha256",
        "label_status",
        "evidence_level",
        "evidence_score",
        "is_rna_modifier",
        "is_train_positive",
        "is_train_negative",
        "is_trainable",
        "train_exclusion_reason",
    ]
    matrix = df[base_columns].copy() if not df.empty else pd.DataFrame(columns=base_columns)
    for field, vocab in CONTROLLED_VOCAB.items():
        if field == "label_status":
            continue
        prefix = field.replace("_labels", "")
        for label in sorted(vocab):
            column = f"{prefix}__{label}"
            if df.empty:
                matrix[column] = []
            else:
                matrix[column] = df[field].apply(lambda value, label=label: int(label in split_labels(value)))
    matrix["exact_modification_labels"] = df.get("exact_modification_labels", pd.Series(dtype=str))
    return matrix


def _manifest_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []

    def add(name: str, source_type: str, path_or_url: str | Path, notes: str = "") -> None:
        value = str(path_or_url)
        path = Path(value)
        is_url = value.startswith("http://") or value.startswith("https://")
        rows.append(
            {
                "source_name": name,
                "source_type": source_type,
                "path_or_url": value,
                "access_date": now,
                "exists": "" if is_url else path.exists(),
                "md5": "" if is_url else file_md5(path),
                "sha256": "" if is_url else file_sha256(path),
                "version": "",
                "notes": notes,
            }
        )

    sources = config.get("sources", {})
    uniprot_cfg = sources.get("uniprot", {})
    add("uniprot_raw_tsv", "raw_or_cached_download", uniprot_cfg.get("raw_tsv", ""), f"fetch={uniprot_cfg.get('fetch', False)}")
    query_cfg = uniprot_cfg.get("query_config", "")
    if query_cfg:
        add("uniprot_query_config", "config", query_cfg)
    for name, cfg_key in [("rhea_raw_tsv", "rhea"), ("go_raw_obo", "go")]:
        cfg = sources.get(cfg_key, {})
        path = cfg.get("raw_tsv", cfg.get("raw_obo", ""))
        add(name, "raw_or_manual_source", path)
    for source_name in ["modomics", "ecocyc"]:
        cfg = sources.get(source_name, {})
        add(f"{source_name}_manual_table", "manual_import", cfg.get("manual_table", ""))
    manual_cfg = sources.get("manual_literature", {})
    add("manual_literature_seeds", "manual_curated_table", manual_cfg.get("table", ""))
    legacy_cfg = sources.get("legacy_pilot", {})
    for key in ["positive_fasta", "negative_fasta", "metadata"]:
        add(f"legacy_pilot_{key}", "legacy_input", legacy_cfg.get(key, ""))
    for path in protein_interim_paths(config) + nonprotein_interim_paths(config):
        add(Path(path).stem, "normalized_interim", path)
    return rows


def build(config_path: str | Path = "configs/config.yaml") -> dict[str, Path]:
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    master = _load_protein_tables(protein_interim_paths(config))
    master = enforce_negative_policy(master)
    master = _refresh_training_flags(master)
    master = _add_sequence_representatives(master)
    master = master.reindex(columns=MASTER_COLUMNS)
    master = master.sort_values(
        ["label_status", "is_trainable", "source_database", "accession", "protein_uid"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )

    master_path = processed_path(config, "rnmod_master.parquet")
    master_tsv_path = processed_path(config, "rnmod_master.tsv.gz")
    fasta_path = processed_path(config, "rnmod_sequences.faa")
    label_path = processed_path(config, "rnmod_label_matrix.parquet")
    manifest_path = processed_path(config, "source_manifest.tsv")
    card_path = processed_path(config, "rnmod_dataset_card.md")

    write_parquet(master, master_path)
    write_tsv(master, master_tsv_path)
    write_unique_fasta(master.to_dict(orient="records"), fasta_path)
    write_parquet(_label_matrix(master), label_path)
    manifest = pd.DataFrame(_manifest_rows(config))
    write_tsv(manifest, manifest_path)
    make_dataset_card(master, manifest, card_path, title=config.get("project_title", "tRNA Modification Protein Prediction Dataset Pipeline"))
    return {
        "master": master_path,
        "master_tsv": master_tsv_path,
        "fasta": fasta_path,
        "labels": label_path,
        "manifest": manifest_path,
        "card": card_path,
    }
