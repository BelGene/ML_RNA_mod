from __future__ import annotations

from typing import Any


DEFAULT_EVIDENCE_SCORES = {
    "experimental_direct": 1.00,
    "reviewed_swissprot": 0.90,
    "reviewed_swissprot_query": 0.85,
    "curated_pathway_database": 0.85,
    "structural_supported": 0.80,
    "reviewed_homolog": 0.75,
    "annotated_homolog_day1": 0.60,
    "annotation_control_day1": 0.60,
    "unreviewed_homolog": 0.45,
    "manual_hypothesis": 0.35,
    "product_annotation_only": 0.20,
    "product_annotation_day1": 0.20,
    "unknown": 0.00,
}


def evidence_score(evidence_level: str, config: dict[str, Any] | None = None) -> float:
    rules = (config or {}).get("evidence_level_scores", DEFAULT_EVIDENCE_SCORES)
    return float(rules.get(evidence_level or "unknown", rules.get("unknown", 0.0)))


def status_from_evidence(is_positive: bool, evidence_level: str, reviewed: bool = False) -> str:
    if not is_positive:
        return "unknown"
    if evidence_level == "experimental_direct":
        return "gold_positive"
    if reviewed or evidence_level in {"reviewed_swissprot", "curated_pathway_database", "structural_supported"}:
        return "silver_positive"
    return "bronze_candidate"
