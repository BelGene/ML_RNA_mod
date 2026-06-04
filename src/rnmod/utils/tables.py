from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def split_labels(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    for sep in (";", "|", ","):
        if sep in text:
            return [item.strip() for item in text.split(sep) if item.strip()]
    return [text]


def join_labels(values: Iterable[str] | str | None) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        values = split_labels(values)
    clean = sorted({str(value).strip() for value in values if str(value).strip()})
    return ";".join(clean)


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix == ".gz" or table_path.name.endswith(".tsv.gz"):
        return pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if out.name.endswith(".gz") else None
    df.to_csv(out, sep="\t", index=False, compression=compression)

