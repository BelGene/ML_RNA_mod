# Numbered Pipeline Scripts

Run order:

1. `00_data_download/00_download_sources.py`: download configured public raw source files.
2. `01_pipeline/01_run_pipeline.sh`: run the Snakemake pipeline end to end.
3. `02_validation/02_validate_outputs.py`: validate generated processed outputs.

Proof-of-concept path:

1. `03_poc/01_build_modomics_trna_dataset.py`: build a small curated MODOMICS
   tRNA protein dataset with UniProt sequences and enzyme-type labels.
2. `03_poc/02_train_embedding_logreg.py`: train a mean-pooled embedding
   logistic-regression baseline from precomputed ESM-style embeddings.

Each script has a user-parameter section at the top. Routine path and source
changes should still be made in `configs/config.yaml`.
