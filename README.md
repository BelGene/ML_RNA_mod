# ML_RNA_mod

`ML_RNA_mod` builds a curated, evidence-aware dataset for bacterial and phage/prophage-associated RNA-modification proteins. This MVP creates the benchmark dataset needed before training protein language model classifiers.

The first deliverable is dataset construction only. Embeddings, classifiers, candidate ranking, and structure-informed rescues are future phases.

## Quick Start

From this directory:

```bash
conda env create -f environment.yml
conda activate ML_RNA_mod
snakemake --cores 8 --use-conda data/processed/rnmod_master.parquet
```

For local development without installing the package:

```bash
export PYTHONPATH=$PWD/src
rnmod build-master --config configs/config.yaml
pytest
```

## Output Files

`data/processed/rnmod_master.parquet`

Main machine-readable master table. It contains one row per curated protein record with accession, sequence hashes, organism, source provenance, RNA-modification labels, evidence level, confidence score, and training flags.

`data/processed/rnmod_master.tsv.gz`

Compressed tabular version of the master table for inspection in Excel, R, Python, or command-line tools.

`data/processed/rnmod_sequences.faa`

FASTA file of unique protein sequences from the master table. This is the sequence input for future MMseqs2, HMMER, and protein language model embedding jobs.

`data/processed/rnmod_label_matrix.parquet`

Model-ready labels keyed by `protein_uid`. It contains binary and multilabel targets such as RNA-modifier status, role labels, target RNA labels, chemistry labels, site buckets, and trainability flags.

`data/processed/source_manifest.tsv`

Reproducibility manifest. It records raw/manual/interim sources, path or URL, source type, access date, file checksum when available, and source notes.

`data/processed/rnmod_dataset_card.md`

Human-readable dataset summary. It reports dataset size, source contributions, label counts, positive/negative policy, leakage concerns, limitations, and citation notes.

`data/interim/*/*.parquet`

Normalized intermediate tables from each source. These are useful for debugging source-specific imports before they are merged into the master dataset.

`data/raw/`

Raw downloaded or manually supplied source files. The workflow preserves these files so checksums can be tracked.

`data/manual/literature_seeds.tsv`

Manual curation table for literature-backed proteins, including high-value prophage examples. Rows without sequences are retained as source records but are excluded from sequence FASTA generation until a protein sequence is supplied.

## Source Policy

Positive RNA-modification records are assembled from reviewed UniProt, Rhea/GO annotations, MODOMICS/EcoCyc manual imports, manual literature seeds, and the previous EDL933 pilot seed library. Hard negatives must be curated non-RNA enzymes such as DNA methyltransferases, restriction-modification methyltransferases, protein methyltransferases, small-molecule methyltransferases, or non-RNA deaminases.

Unknown, hypothetical, fragmentary, conflicted, and unlabeled proteins are never used as hard negatives.

## Training Policy

Use `gold_positive` and `silver_positive` records as positives. Use only curated `hard_negative` rows as negatives. Keep `bronze_candidate` as candidate-only. Exclude `unknown`, `conflicted`, fragmentary, hypothetical, and sequence-missing records from model training.

