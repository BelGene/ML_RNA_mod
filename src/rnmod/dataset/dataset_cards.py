from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

from pathlib import Path

import pandas as pd

from rnmod.utils.tables import split_labels


def _value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df:
        return pd.Series(dtype=int)
    return df[column].fillna("").replace("", "missing").value_counts().sort_index()


def _multilabel_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if df.empty or column not in df:
        return counts
    for value in df[column]:
        for label in split_labels(value):
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _markdown_counts(title: str, counts: pd.Series | dict[str, int]) -> list[str]:
    lines = [f"## {title}", ""]
    if isinstance(counts, pd.Series):
        iterable = counts.items()
    else:
        iterable = counts.items()
    rows = list(iterable)
    if not rows:
        lines.append("No records.")
        lines.append("")
        return lines
    lines.extend(["| Label | Count |", "|---|---:|"])
    for label, count in rows:
        lines.append(f"| {label} | {int(count)} |")
    lines.append("")
    return lines


def make_dataset_card(
    master: pd.DataFrame,
    manifest: pd.DataFrame,
    output: str | Path,
    title: str = "tRNA Modification Protein Prediction Dataset Pipeline",
) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_records = len(master)
    n_sequences = int(master["sequence_sha256"].astype(str).replace("", pd.NA).dropna().nunique()) if not master.empty else 0
    n_trainable = int(master["is_trainable"].astype(bool).sum()) if not master.empty else 0
    n_train_pos = int(master["is_train_positive"].astype(bool).sum()) if not master.empty else 0
    n_train_neg = int(master["is_train_negative"].astype(bool).sum()) if not master.empty else 0

    lines = [
        f"# {title} Dataset Card",
        "",
        "This dataset is a curated MVP benchmark for bacterial and phage/prophage-associated RNA-modification proteins. It is intended for later protein language model embedding and classifier development, not for making functional claims about unvalidated proteins.",
        "",
        "## Summary",
        "",
        f"- Protein records: {n_records}",
        f"- Unique protein sequences: {n_sequences}",
        f"- Trainable records: {n_trainable}",
        f"- Trainable positives: {n_train_pos}",
        f"- Trainable hard negatives: {n_train_neg}",
        "",
        "## Positive And Negative Policy",
        "",
        "Gold and silver positives are used as positive training labels. Only curated hard negatives are used as negative training labels. Bronze candidates are retained for candidate discovery and benchmarking but excluded from supervised training by default.",
        "",
        "Unknown, hypothetical, fragmentary, conflicted, and sequence-missing records are excluded from training. Hypothetical proteins are never promoted to hard negatives.",
        "",
    ]
    lines.extend(_markdown_counts("Label Status Counts", _value_counts(master, "label_status")))
    lines.extend(_markdown_counts("Evidence Level Counts", _value_counts(master, "evidence_level")))
    lines.extend(_markdown_counts("Source Contributions", _value_counts(master, "source_database")))
    lines.extend(_markdown_counts("Target RNA Labels", _multilabel_counts(master, "target_rna_labels")))
    lines.extend(_markdown_counts("Chemistry Labels", _multilabel_counts(master, "chemistry_labels")))
    lines.extend(_markdown_counts("Manifest Sources", _value_counts(manifest, "source_type")))
    lines.extend(
        [
            "## Limitations",
            "",
            "- This MVP is source-quality limited: manual MODOMICS and EcoCyc imports are hooks until curated exports are supplied.",
            "- UniProt live fetching is disabled by default to keep the build reproducible and lightweight.",
            "- The legacy EDL933 pilot seed library is useful for bootstrapping but should be replaced or supplemented with broader curated bacterial and phage homologs.",
            "- Broad enzyme superfamilies such as SAM-dependent methyltransferases, Nudix hydrolases, radical-SAM proteins, and deaminases require family, motif, and structural validation before RNA specificity is inferred.",
            "- Sequence redundancy is tracked by exact SHA256 identity; orthology and phylogenetic leakage control are future phases.",
            "",
            "## Citation Notes",
            "",
            "Cite the original databases and papers used for each source when this dataset is used in analysis. Recommended source classes include UniProtKB/Swiss-Prot, Rhea, Gene Ontology, MODOMICS, EcoCyc/BioCyc, and manually curated literature records.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
