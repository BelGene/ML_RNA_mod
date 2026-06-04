from __future__ import annotations

from copy import deepcopy

from rnmod.utils.tables import split_labels


def _text(row: dict[str, object]) -> str:
    parts = [
        row.get("protein_name", ""),
        row.get("gene_name", ""),
        row.get("family", ""),
        row.get("reaction", ""),
        row.get("substrate_class", ""),
        row.get("source_notes", ""),
    ]
    return " ".join(str(part).lower() for part in parts if part is not None)


def is_hypothetical_text(text: str) -> bool:
    bad_terms = ["hypothetical protein", "uncharacterized protein", "unknown protein", "domain of unknown function"]
    return any(term in text.lower() for term in bad_terms)


def apply_family_label_rules(row: dict[str, object]) -> dict[str, object]:
    out = deepcopy(row)
    text = _text(out)
    gene = str(out.get("gene_name", "")).lower()
    family = str(out.get("family", "")).lower()
    label_status = str(out.get("label_status", "") or "unknown")

    role = set(split_labels(out.get("role_labels"))) or {"unknown"}
    target = set(split_labels(out.get("target_rna_labels"))) or {"unknown"}
    scope = set(split_labels(out.get("modification_scope_labels"))) or {"unknown"}
    chemistry = set(split_labels(out.get("chemistry_labels"))) or {"unknown"}
    site = set(split_labels(out.get("site_bucket_labels"))) or {"unknown"}

    def replace_unknown(label_set: set[str], *values: str) -> set[str]:
        label_set.discard("unknown")
        label_set.update(values)
        return label_set

    if "dna methyltransferase" in text or "dna adenine methyl" in text or gene in {"dam", "dcm", "hsdm"} or "restriction" in text:
        role = replace_unknown(role, "nonRNA_enzyme")
        target = replace_unknown(target, "DNA")
        scope = replace_unknown(scope, "nonRNA_modification")
        chemistry = replace_unknown(chemistry, "DNA_methyltransferase")
        site = replace_unknown(site, "protein_or_nonRNA")
        if not is_hypothetical_text(text):
            label_status = "hard_negative"

    elif "protein methyltransferase" in text:
        role = replace_unknown(role, "nonRNA_enzyme")
        target = replace_unknown(target, "protein")
        scope = replace_unknown(scope, "nonRNA_modification")
        chemistry = replace_unknown(chemistry, "protein_methyltransferase")
        site = replace_unknown(site, "protein_or_nonRNA")
        if not is_hypothetical_text(text):
            label_status = "hard_negative"

    elif "small-molecule methyltransferase" in text or "small molecule methyltransferase" in text:
        role = replace_unknown(role, "nonRNA_enzyme")
        target = replace_unknown(target, "small_molecule")
        scope = replace_unknown(scope, "nonRNA_modification")
        chemistry = replace_unknown(chemistry, "small_molecule_methyltransferase")
        site = replace_unknown(site, "protein_or_nonRNA")
        if not is_hypothetical_text(text):
            label_status = "hard_negative"

    elif gene == "trmd" or "trmd-like" in family or "trna (guanine" in text:
        role = replace_unknown(role, "writer")
        target = replace_unknown(target, "tRNA")
        scope = replace_unknown(scope, "methylation")
        chemistry = replace_unknown(chemistry, "SAM_methyltransferase")
        site = replace_unknown(site, "position_37", "anticodon_loop")
        out["exact_modification_labels"] = "m1G37"
        label_status = "silver_positive" if label_status == "unknown" else label_status

    elif gene.startswith("rlu") or gene.startswith("tru") or "pseudouridine synthase" in text or "pseudouridyl" in text:
        role = replace_unknown(role, "writer")
        target = replace_unknown(target, "mixed_RNA" if "trna" in text and "rrna" in text else ("tRNA" if "trna" in text else "rRNA"))
        scope = replace_unknown(scope, "pseudouridylation")
        chemistry = replace_unknown(chemistry, "pseudouridine_synthase")
        site = replace_unknown(site, "tRNA_body" if "trna" in text else "rRNA_other")
        label_status = "silver_positive" if label_status == "unknown" else label_status

    elif gene.startswith("rsm") or gene.startswith("rlm") or "rrna" in text and "methyltransferase" in text:
        role = replace_unknown(role, "writer")
        target = replace_unknown(target, "rRNA")
        scope = replace_unknown(scope, "methylation")
        chemistry = replace_unknown(chemistry, "SAM_methyltransferase")
        site = replace_unknown(site, "rRNA_other")
        label_status = "silver_positive" if label_status == "unknown" else label_status

    elif gene in {"mnma", "ttca"} or "thiouridylase" in text or "thiolation" in text:
        role = replace_unknown(role, "writer")
        target = replace_unknown(target, "tRNA" if "trna" in text else "mixed_RNA")
        scope = replace_unknown(scope, "thiolation")
        chemistry = replace_unknown(chemistry, "thiouridylase")
        site = replace_unknown(site, "wobble_position_34" if "34" in text else "unknown")
        label_status = "silver_positive" if label_status == "unknown" else label_status

    elif gene in {"tada"} or "trna adenosine deaminase" in text:
        role = replace_unknown(role, "writer")
        target = replace_unknown(target, "tRNA")
        scope = replace_unknown(scope, "deamination")
        chemistry = replace_unknown(chemistry, "deaminase")
        site = replace_unknown(site, "anticodon_loop")
        label_status = "silver_positive" if label_status == "unknown" else label_status

    elif gene in {"tgt", "quea", "quec", "qued", "quee", "quef", "queg"} or "queuosine" in text or "preq" in text:
        role = replace_unknown(role, "writer")
        target = replace_unknown(target, "tRNA")
        scope = replace_unknown(scope, "queuosine")
        chemistry = replace_unknown(chemistry, "queuosine_pathway")
        site = replace_unknown(site, "wobble_position_34")
        label_status = "silver_positive" if label_status == "unknown" else label_status

    elif gene in {"nudc"} or "nudix" in text:
        chemistry = replace_unknown(chemistry, "Nudix_hydrolase")
        if "nad-rna" in text or "rna cap" in text or "decapping" in text:
            role = replace_unknown(role, "eraser", "RNA_cap_processing")
            target = replace_unknown(target, "mRNA")
            scope = replace_unknown(scope, "RNA_cap_or_terminal")
            site = replace_unknown(site, "RNA_5prime_cap_or_end")
            label_status = "bronze_candidate" if label_status == "unknown" else label_status
        elif label_status == "unknown":
            role = replace_unknown(role, "nonRNA_enzyme")
            target = replace_unknown(target, "small_molecule")
            scope = replace_unknown(scope, "nonRNA_modification")
            site = replace_unknown(site, "protein_or_nonRNA")

    elif gene == "modb" or "rnaylation" in text or "rna adp-ribosyl" in text:
        role = replace_unknown(role, "writer")
        target = replace_unknown(target, "mRNA")
        scope = replace_unknown(scope, "RNAylation")
        chemistry = replace_unknown(chemistry, "ADP_ribosyltransferase")
        site = replace_unknown(site, "RNA_5prime_cap_or_end")
        label_status = "bronze_candidate" if label_status == "unknown" else label_status

    if is_hypothetical_text(text):
        out["is_hypothetical"] = True
        if label_status == "hard_negative":
            label_status = "unknown"

    out["label_status"] = label_status
    out["role_labels"] = ";".join(sorted(role))
    out["target_rna_labels"] = ";".join(sorted(target))
    out["modification_scope_labels"] = ";".join(sorted(scope))
    out["chemistry_labels"] = ";".join(sorted(chemistry))
    out["site_bucket_labels"] = ";".join(sorted(site))
    return out

