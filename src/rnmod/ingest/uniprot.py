from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yaml

from rnmod.ingest.common import write_records
from rnmod.labels.evidence import status_from_evidence
from rnmod.settings import load_config


DEFAULT_PAGE_SIZE = 500
DEFAULT_TIMEOUT_SECONDS = 120
REQUEST_PAUSE_SECONDS = 0.2


def read_query_config(path: str | Path) -> dict:
    query_path = Path(path)
    if not query_path.exists():
        return {"queries": [], "fields": []}
    with query_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"queries": [], "fields": []}


def _raise_for_uniprot_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = response.text.strip().splitlines()[-1:] or [str(exc)]
        raise RuntimeError(f"UniProt request failed: {message[0]}") from exc


def _frame_from_tsv(text: str) -> pd.DataFrame:
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame()
    return pd.read_csv(StringIO(stripped), sep="\t", dtype=str, keep_default_na=False)


def download_uniprot_raw(query_config: dict, raw_tsv: str | Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
    endpoint = query_config.get("uniprot_endpoint", "https://rest.uniprot.org/uniprotkb/search")
    fields = ",".join(query_config.get("fields", []))
    page_size = int(query_config.get("page_size", DEFAULT_PAGE_SIZE))
    raw_tsv = Path(raw_tsv)
    raw_tsv.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for query in query_config.get("queries", []):
        params = {
            "query": query["query"],
            "format": "tsv",
            "fields": fields,
            "size": page_size,
        }
        next_url: str | None = endpoint
        while next_url:
            if next_url == endpoint:
                response = requests.get(endpoint, params=params, timeout=timeout)
            else:
                response = requests.get(next_url, timeout=timeout)
            _raise_for_uniprot_status(response)
            temp = _frame_from_tsv(response.text)
            if not temp.empty:
                temp["rnmod_query_id"] = query.get("query_id", "")
                temp["rnmod_label_hint"] = query.get("label_hint", "")
                frames.append(temp)
            next_url = response.links.get("next", {}).get("url")
            time.sleep(REQUEST_PAUSE_SECONDS)
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
    if not source_cfg.get("enabled", True):
        return write_records([], output)
    query_config = read_query_config(source_cfg.get("query_config", "configs/queries_uniprot.yaml"))
    raw_tsv = Path(source_cfg.get("raw_tsv", "data/raw/uniprot/uniprot_records.tsv"))

    if source_cfg.get("fetch", False) and not raw_tsv.exists():
        download_uniprot_raw(query_config, raw_tsv)

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
