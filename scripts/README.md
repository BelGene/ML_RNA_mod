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
3. `03_poc/03_build_weak_uniprot_trna_dataset.py`: build a larger no-manual
   weak-label UniProt/MODOMICS tRNA-modifier dataset for a binary first POC.
   The default `small` profile keeps the first run compact; use
   `--profile standard` or `--profile full` for larger screens.
4. `03_poc/04_cluster_split_sequences.py`: cluster weak-POC FASTA sequences
   with MMseqs2 and assign whole clusters to train/validation/test splits.
5. `03_poc/05_train_weak_embedding_logreg.py`: train split-aware binary or
   mechanism logistic-regression models from precomputed embeddings.
6. `03_poc/06_embed_sequences_esmc.py`: generate one mean-pooled ESM-C `.npy`
   embedding per FASTA record. Intended to run on a GPU machine.

Batch templates:

- `03_poc/bridges_esmc_embed.sbatch`: single-GPU Bridges/Slurm template for
  embedding the weak-POC FASTA with ESM-C 6B by default.

Each script has a user-parameter section at the top. Routine path and source
changes should still be made in `configs/config.yaml`.
