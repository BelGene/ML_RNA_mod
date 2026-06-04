from __future__ import annotations

import pandas as pd


def enforce_negative_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Prevent weak or unknown records from becoming model negatives."""
    if df.empty:
        return df
    out = df.copy()
    text = (
        out.get("protein_name", "").astype(str).str.lower()
        + " "
        + out.get("source_notes", "").astype(str).str.lower()
        + " "
        + out.get("family", "").astype(str).str.lower()
    )
    hypothetical = text.str.contains("hypothetical|uncharacterized|unknown protein|domain of unknown function", regex=True)
    out.loc[hypothetical & (out["label_status"] == "hard_negative"), "label_status"] = "unknown"
    return out

