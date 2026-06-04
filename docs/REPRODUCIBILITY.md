# Reproducibility And Audit Notes

## Repository Scope

The repository versions code, configuration, tests, environment files, and small
manual curation tables. It does not version raw downloads, interim parquet
files, processed model-ready datasets, logs, caches, or local Snakemake
environments.

## Rebuild Contract

A fresh clone should be able to rebuild the dataset with:

```bash
conda env create -f environment.yml
conda activate ML_RNA_mod
scripts/01_pipeline/01_run_pipeline.sh
```

The first rule calls `scripts/00_data_download/00_download_sources.py`, which downloads public
raw sources configured in `configs/config.yaml`.

## Manual Inputs

The tracked manual inputs are:

- `data/manual/literature_seeds.tsv`
- `data/manual/modomics_import.tsv`
- `data/manual/ecocyc_import.tsv`

These files are part of the audit trail and should be edited by pull request or
explicit commit.

## Ignored Outputs

Ignored runtime paths:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `.snakemake/`
- `logs/`

The generated `data/processed/source_manifest.tsv` records source paths, access
date, and checksums for files available at build time.
