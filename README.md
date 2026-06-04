# tRNA Modification Protein Prediction Dataset Pipeline

This repository builds a curated, evidence-aware benchmark dataset for training
machine learning models to predict RNA and especially tRNA modification
proteins. It is designed for bacterial, phage, prophage, and mobile-element
contexts.

The current deliverable is dataset construction: source download, source
normalization, label assignment, train/test policy fields, sequence FASTA
generation, and an auditable dataset card.

For a lightweight proof of concept, this repository also includes a simpler
MODOMICS-only path: retrieve curated tRNA-modification proteins, use MODOMICS
enzyme type as the prediction label, and train a small classifier from frozen
protein embeddings.

## What Is Tracked

Tracked:

- source code in `src/rnmod/`
- ordered pipeline scripts in numbered folders under `scripts/`
- Snakemake workflow in `workflow/Snakefile`
- configuration in `configs/`
- conda environments
- tests
- small manual curation tables in `data/manual/`

Ignored and regenerated:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `.snakemake/`
- `logs/`

## Quick Start

From the repository root:

```bash
conda env create -f environment.yml
conda activate ML_RNA_mod
scripts/01_pipeline/01_run_pipeline.sh
```

Validate generated outputs:

```bash
python scripts/02_validation/02_validate_outputs.py --config configs/config.yaml
```

## Simple POC Workflow

Use this path when the goal is only to prove that curated tRNA-modifier labels
can be learned from protein embeddings.

Build the MODOMICS tRNA protein dataset:

```bash
python scripts/03_poc/01_build_modomics_trna_dataset.py
```

This writes:

```text
data/processed/poc/modomics_trna_proteins.tsv
data/processed/poc/modomics_trna_sequences.faa
data/processed/poc/modomics_trna_label_matrix.tsv
data/processed/poc/modomics_trna_dataset_card.md
```

Then generate ESM3 embeddings for the FASTA sequences with your preferred
embedding script. Save one `.pt` or `.npy` file per UniProt accession, for
example:

```text
P53088_hidden_layer_steps10.pt
Q58428_hidden_layer_steps10.pt
```

Loading `.pt` embeddings requires `torch`; `.npy` embeddings only require NumPy.

Train the proof-of-concept classifier:

```bash
python scripts/03_poc/02_train_embedding_logreg.py \
  --embedding-dir path/to/esm3_embeddings \
  --output-dir data/processed/poc/ml_runs \
  --run-name modomics_trna_logreg
```

The trainer mean-pools each embedding and fits one-vs-rest logistic-regression
models for MODOMICS enzyme-type labels with cross-validation.

## Weak-Label UniProt POC Workflow

Use this path when you want a larger no-manual-curation proof of concept before
committing to curated database expansion. It combines MODOMICS tRNA anchors with
automated reviewed-UniProt text-query positives and reviewed non-tRNA controls.

Build the default small weak-label dataset:

```bash
python scripts/03_poc/03_build_weak_uniprot_trna_dataset.py
```

The default `small` profile caps live UniProt query results and focuses the
weak-label expansion on bacteria/archaea so the first embedding run stays
manageable. Larger runs are available with:

```bash
python scripts/03_poc/03_build_weak_uniprot_trna_dataset.py --profile standard
python scripts/03_poc/03_build_weak_uniprot_trna_dataset.py --profile full
```

This writes:

```text
data/processed/poc_weak/weak_trna_mod_proteins.tsv
data/processed/poc_weak/weak_trna_mod_sequences.faa
data/processed/poc_weak/weak_trna_mod_label_matrix.tsv
data/processed/poc_weak/weak_trna_mod_dataset_card.md
```

Primary label:

```text
target__trna_modifier
```

Optional broad mechanism labels are emitted as `mechanism__...` columns. These
labels are weak and should be evaluated with cluster-heldout splits before being
interpreted.

Create cluster-heldout splits:

```bash
python scripts/03_poc/04_cluster_split_sequences.py
```

This writes:

```text
data/processed/poc_weak/splits/mmseqs50/cluster_membership.tsv
data/processed/poc_weak/splits/mmseqs50/cluster_assignments.tsv
data/processed/poc_weak/splits/mmseqs50/split_assignments.tsv
data/processed/poc_weak/splits/mmseqs50/split_summary.tsv
data/processed/poc_weak/splits/mmseqs50/cluster_split_card.md
```

After generating embeddings for `weak_trna_mod_sequences.faa`, train the first
binary classifier:

```bash
python scripts/03_poc/05_train_weak_embedding_logreg.py \
  --embedding-dir path/to/embeddings \
  --task binary
```

For the recommended high-quality GPU embedding run with ESM-C 6B, use:

```bash
python scripts/03_poc/06_embed_sequences_esmc.py \
  --fasta data/processed/poc_weak/weak_trna_mod_sequences.faa \
  --output-dir data/processed/poc_weak/embeddings/esmc_6b \
  --model-name biohub/ESMC-6B \
  --max-tokens-per-batch 4096 \
  --device-map auto \
  --device cuda
```

On Bridges/Slurm, see `docs/POC_ESMC_BRIDGES.md` and
`scripts/03_poc/bridges_esmc_embed.sbatch`.

## Pipeline Order

1. `scripts/00_data_download/00_download_sources.py` downloads configured public raw sources.
2. Snakemake ingests each source into `data/interim/`.
3. `rnmod build-master` merges normalized records into processed model-ready outputs.
4. `scripts/02_validation/02_validate_outputs.py` checks final outputs.

Main generated outputs:

```text
data/processed/rnmod_master.parquet
data/processed/rnmod_master.tsv.gz
data/processed/rnmod_sequences.faa
data/processed/rnmod_label_matrix.parquet
data/processed/source_manifest.tsv
data/processed/rnmod_dataset_card.md
```

## Configuration

Edit `configs/config.yaml` for routine changes:

- source enable/disable flags
- public source URLs
- raw/interim/processed output paths
- legacy pilot input paths or URLs
- dataset behavior flags

No default config path points outside this repository.

## Label Policy

Use `gold_positive` and `silver_positive` records as positives. Use only curated
`hard_negative` rows as negatives. Keep `bronze_candidate` as candidate-only.
Exclude `unknown`, `conflicted`, fragmentary, hypothetical, and
sequence-missing records from supervised training.

Unknown, hypothetical, fragmentary, conflicted, and unlabeled proteins are never
used as hard negatives.

## Manual Curation Inputs

Tracked manual source tables:

```text
data/manual/literature_seeds.tsv
data/manual/modomics_import.tsv
data/manual/ecocyc_import.tsv
```

These are intentionally small and auditable. Larger public datasets are
downloaded or regenerated by the pipeline.

See `docs/REPRODUCIBILITY.md` for the full data and audit policy.
