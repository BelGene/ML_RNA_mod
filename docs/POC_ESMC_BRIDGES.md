# Weak POC ESM-C Embeddings On Bridges

This runbook generates frozen ESM-C protein embeddings for the weak-label tRNA
modifier proof of concept. It does not train a neural network; it writes one
mean-pooled `.npy` vector per UniProt accession. The default Bridges batch
template uses `biohub/ESMC-6B`; use `biohub/ESMC-600M` if GPU memory is limited
or you want a faster first pass.

## Transfer

The simplest option is to copy the current repository state to Bridges, keeping
the generated weak-POC files but excluding local caches:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '.local_tools/' \
  --exclude '.snakemake/' \
  --exclude '__pycache__/' \
  /local/storage/alen/projects/4_tRNA_ML/ \
  USER@bridges2.psc.edu:/PATH/4_tRNA_ML/
```

Minimal files needed for embedding and downstream logistic regression:

```text
pyproject.toml
src/
scripts/03_poc/05_train_weak_embedding_logreg.py
scripts/03_poc/06_embed_sequences_esmc.py
scripts/03_poc/bridges_esmc_embed.sbatch
data/processed/poc_weak/weak_trna_mod_sequences.faa
data/processed/poc_weak/weak_trna_mod_label_matrix.tsv
data/processed/poc_weak/splits/mmseqs50/split_assignments.tsv
```

The current weak-POC processed directory is small enough to transfer as-is.

## Environment

On Bridges, create a GPU Python environment. Prefer a Bridges-supported PyTorch
module if one is available for your account. A direct virtualenv setup is:

```bash
cd /PATH/4_tRNA_ML
python -m venv "$SCRATCH/venvs/trna-esmc"
source "$SCRATCH/venvs/trna-esmc/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate huggingface_hub numpy pandas scikit-learn
pip install "esm @ git+https://github.com/Biohub/esm.git@main"
```

If the Hugging Face model download is gated for your account, authenticate once:

```bash
huggingface-cli login
```

## Smoke Test

Run a short interactive test before the full batch job:

```bash
source "$SCRATCH/venvs/trna-esmc/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"

python scripts/03_poc/06_embed_sequences_esmc.py \
  --fasta data/processed/poc_weak/weak_trna_mod_sequences.faa \
  --output-dir data/processed/poc_weak/embeddings/esmc_6b_smoke \
  --model-name biohub/ESMC-6B \
  --max-tokens-per-batch 2048 \
  --limit 20 \
  --device-map auto \
  --device cuda
```

## Batch Run

Edit `scripts/03_poc/bridges_esmc_embed.sbatch` before submitting:

- uncomment and set `#SBATCH -A YOUR_ALLOCATION_ID` if required
- set the GPU type in `--gres=gpu:h100-80:1` to the GPU type available to you
- if your system exposes A100s, this is often `--gres=gpu:a100:1`, but confirm
  with `sinfo` or your allocation notes
- keep `biohub/ESMC-6B` for the highest-quality public ESM-C representation on
  an 80 GB GPU; use the smaller command below for a faster run or a lower-memory
  GPU

Submit:

```bash
sbatch scripts/03_poc/bridges_esmc_embed.sbatch
```

Faster 600M fallback:

```bash
MODEL_NAME=biohub/ESMC-600M \
OUTPUT_DIR=data/processed/poc_weak/embeddings/esmc_600m \
MAX_TOKENS_PER_BATCH=8192 \
sbatch scripts/03_poc/bridges_esmc_embed.sbatch
```

Expected output:

```text
data/processed/poc_weak/embeddings/esmc_6b/*.npy
data/processed/poc_weak/embeddings/esmc_6b/embedding_manifest.tsv
data/processed/poc_weak/embeddings/esmc_6b/embedding_config.json
```

By default, the script truncates proteins longer than 2048 model tokens. In this
small weak-POC dataset that affects only one sequence.

## Bring Results Back

```bash
rsync -av \
  USER@bridges2.psc.edu:/PATH/4_tRNA_ML/data/processed/poc_weak/embeddings/esmc_6b/ \
  /local/storage/alen/projects/4_tRNA_ML/data/processed/poc_weak/embeddings/esmc_6b/
```

Then train the split-aware binary baseline:

```bash
python scripts/03_poc/05_train_weak_embedding_logreg.py \
  --embedding-dir data/processed/poc_weak/embeddings/esmc_6b \
  --task binary
```
