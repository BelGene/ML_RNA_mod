# Robust ESM-C Embedding Classifier Audit

## Evaluation Design

- Features: frozen mean-pooled ESM-C embeddings.
- Binary label scope: all trainable rows.
- Mechanism label scope: `mechanism_labeled` rows.
- Model family: L2-regularized logistic regression with class-balanced loss.
- Model selection: choose `C` on validation by validation AUPR.
- Threshold selection: choose threshold on validation by F1.
- Final model: refit selected model on train+validation, then evaluate once on held-out test.

## Split Audit

- Embedded rows: `2411`.
- Split counts: `{'train': 1687, 'val': 362, 'test': 362}`.
- Cluster split leakage: `False`.
- Duplicate accessions: `0`.
- Duplicate embedding groups spanning splits: `0`.

## Aggregate Metrics

| family | eval_split | n_labels | micro_aupr | micro_auroc | macro_aupr | macro_auroc | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary | val | 1 | 0.996 | 0.995 | 0.996 | 0.995 | 0.986 |
| binary | test | 1 | 0.990 | 0.992 | 0.990 | 0.992 | 0.988 |
| mechanism | val | 8 | 0.971 | 0.981 | 0.983 | 0.991 | 0.975 |
| mechanism | test | 8 | 0.955 | 0.973 | 0.802 | 0.978 | 0.711 |


## Test Metrics By Label

| label | family | positives | negatives | aupr | auroc | precision | recall | f1 | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trna_modifier | binary | 214 | 148 | 0.990 | 0.992 | 0.995 | 0.981 | 0.988 | 0.987 |
| deaminase | mechanism | 2 | 210 | 0.524 | 0.905 | 0.333 | 0.500 | 0.400 | 0.745 |
| dihydrouridine_synthase | mechanism | 17 | 195 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| methyltransferase | mechanism | 15 | 197 | 0.909 | 0.992 | 0.833 | 0.667 | 0.741 | 0.828 |
| pseudouridine_synthase | mechanism | 31 | 181 | 0.994 | 0.999 | 1.000 | 0.968 | 0.984 | 0.984 |
| queuosine_pathway | mechanism | 12 | 200 | 1.000 | 1.000 | 0.750 | 1.000 | 0.857 | 0.990 |
| t6a_pathway | mechanism | 17 | 195 | 0.884 | 0.976 | 0.667 | 0.824 | 0.737 | 0.894 |
| thiolation | mechanism | 122 | 90 | 0.996 | 0.993 | 0.991 | 0.951 | 0.971 | 0.970 |
| wyosine_pathway | mechanism | 1 | 211 | 0.111 | 0.962 | 0.000 | 0.000 | 0.000 | 0.500 |


## Low-Count Labels

These labels are included for exploratory signal, but their threshold-based metrics are unstable.

| label | val_positives | test_positives | train_positives | low_val_positives | low_test_positives |
| --- | --- | --- | --- | --- | --- |
| deaminase | 1 | 2 | 34 | True | True |
| dihydrouridine_synthase | 1 | 17 | 56 | True | False |
| wyosine_pathway | 1 | 1 | 12 | True | True |


## Skipped Labels

| label | train_positives | val_positives | test_positives | skip_reason |
| --- | --- | --- | --- | --- |
| acetyltransferase | 2 | 2 | 0 | train positives < 10 |


## Interpretation Notes

- High AUROC with very few positives can still be unstable; check AUPR, precision, recall, and the positive counts.
- Perfect test scores are less suspicious after the split audit, but this is still a weak-label POC, not a curated benchmark.
- The saved `.joblib` artifacts are one model per label. A multi-label prediction call should run all label models on the same embedding.

## Key Files

- `per_label_metrics.tsv`
- `aggregate_metrics.tsv`
- `label_audit.tsv`
- `model_selection.tsv`
- `model_index.tsv`
- `models/*.joblib`
- `plots/test_precision_recall_curves.png`
- `plots/test_roc_curves.png`
- `plots/test_metric_bars.png`
