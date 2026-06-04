#!/usr/bin/env bash
# Submit from the repository root after creating the ESM-C Python environment.
#
# PSC Bridges-2 currently requires a GPU type in the GRES request. Edit the
# gpu type and allocation line to match your allocation before submitting.

#SBATCH -J trna-esmc
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH -o esmc_embed_%j.out
#SBATCH -e esmc_embed_%j.err
##SBATCH -A YOUR_ALLOCATION_ID

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

SCRATCH_BASE="${SCRATCH:-${PWD}}"
VENV_PATH="${ESMC_VENV:-${SCRATCH_BASE}/venvs/trna-esmc}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing Python environment: ${VENV_PATH}" >&2
  echo "Create it first or submit with ESMC_VENV=/path/to/venv." >&2
  exit 2
fi

source "${VENV_PATH}/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

python scripts/03_poc/06_embed_sequences_esmc.py \
  --fasta data/processed/poc_weak/weak_trna_mod_sequences.faa \
  --output-dir "${OUTPUT_DIR:-data/processed/poc_weak/embeddings/esmc_6b}" \
  --model-name "${MODEL_NAME:-biohub/ESMC-6B}" \
  --max-tokens-per-batch "${MAX_TOKENS_PER_BATCH:-4096}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --device-map "${DEVICE_MAP:-auto}" \
  --device cuda
