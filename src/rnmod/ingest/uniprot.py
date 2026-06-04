from __future__ import annotations

import time
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml

from rnmod.ingest.common import write_records
from rnmod.labels.evidence import status_from_evidence
from rnmod.settings import load_config


def _read_query_config(path: str | Path) -> dict:
    query_path = Path(path)
    if not query_path.exists():
        return {"queries": [], "fields": []}
    with query_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"queries": [], "fields": []}


def _download_uniprot(config: dict, query_config: dict, raw_tsv: Path) -> None:
    endpoint = query_config.get("uniprot_endpoint", "https://rest.uniprot.org/uniprotkb/search")
    fields = ",".join(query_config.get("fields", []))
    raw_tsv.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for query in query_config.get("queries", []):
        params = {
            "query": query["query"],
            "format": "tsv",
            "fields": fields,
            "size": 500,
        }
        url = endpoint + "?" + urlencode(params)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            continue
        temp = pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False)
        temp["rnmod_query_id"] = query.get("query_id", "")
        temp["rnmod_label_hint"] = query.get("label_hint", "")
        frames.append(temp)
        time.sleep(0.2)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(raw_tsv, sep="\t", index=False)
    else:
        raw_tsv.write_text("", encoding="utf-8")


def _column(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row:
            return str(row.get(name, ""))
    return ""


def ingest(config_path: str | Path, output: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    source_cfg = config["sources"]["uniprot"]
    query_config = _read_query_config(source_cfg.get("query_config", "configs/queries_uniprot.yaml"))
    raw_tsv = Path(source_cfg.get("raw_tsv", "data/raw/uniprot/uniprot_records.tsv"))

    if source_cfg.get("fetch", False):
        _download_uniprot(config, query_config, raw_tsv)

    if not raw_tsv.exists() or raw_tsv.stat().st_size == 0:
        return write_records([], output)

    df = pd.read_csv(raw_tsv, sep="\t", dtype=str, keep_default_na=False)
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        accession = _column(row, "Entry", "accession", "Accession")
        reviewed_text = _column(row, "Reviewed", "reviewed")
        reviewed = reviewed_text.lower() in {"reviewed", "true", "yes"}
        label_hint = _column(row, "rnmod_label_hint", "label_hint")
        protein_name = _column(row, "Protein names", "protein_name")
        gene_name = _column(row, "Gene Names", "gene_names", "gene_name").split(" ")[0]
        sequence = _column(row, "Sequence", "sequence")
        evidence_level = "reviewed_swissprot" if reviewed else "unreviewed_homolog"
        label_status = status_from_evidence(label_hint == "positive", evidence_level, reviewed)
        if label_hint == "negative":
            label_status = "hard_negative"
        records.append(
            {
                "source_record_id": accession or f"uniprot_{len(records) + 1}",
                "accession": accession,
                "reviewed": reviewed,
                "protein_name": protein_name,
                "gene_name": gene_name,
                "organism": _column(row, "Organism", "organism_name"),
                "taxon_id": _column(row, "Organism (ID)", "organism_id"),
                "sequence": sequence,
                "label_status": label_status,
                "evidence_level": evidence_level,
                "source_database": "UniProtKB/Swiss-Prot" if reviewed else "UniProtKB/TrEMBL",
                "source_type": "uniprot_tsv",
                "source_url": f"https://www.uniprot.org/uniprotkb/{accession}/entry" if accession else "",
                "reaction": _column(row, "Rhea IDs", "rhea"),
                "source_notes": (
                    f"go={_column(row, 'Gene Ontology IDs', 'go_id')}; "
                    f"ec={_column(row, 'EC number', 'ec')}; "
                    f"interpro={_column(row, 'InterPro', 'interpro')}; "
                    f"keywords={_column(row, 'Keywords', 'keyword')}; "
                    f"function={_column(row, 'Function [CC]', 'cc_function')}"
                ),
            }
        )
    return write_records(records, output)
