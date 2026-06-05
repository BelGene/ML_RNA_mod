#!/usr/bin/env bash
# Submit after creating the ESM-C Python environment under /ocean/projects/bio250095p.
#
# PSC Bridges-2 currently requires a GPU type in the GRES request. Edit the
# gpu type and allocation line to match your allocation before submitting.

#SBATCH -J trna-esmc
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH -o esmc_embed_%j.out
#SBATCH -e esmc_embed_%j.err
#SBATCH -A bio250095p

set -euo pipefail

module purge 2>/dev/null || true
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
unset LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH TRANSFORMERS_CACHE

PROJECT_ROOT="/ocean/projects/bio250095p/azimic"
REPO_ROOT="${PROJECT_ROOT}/4_tRNA_ML"
cd "${REPO_ROOT}"

LOCAL_JOB_DIR="/local/${USER}/${SLURM_JOB_ID}"
export TMPDIR="${LOCAL_JOB_DIR}"
if [[ "${TMPDIR}" != /local/* ]]; then
  echo "Refusing to run with TMPDIR outside /local: ${TMPDIR}" >&2
  echo "PSC H100 nodes must not use /tmp; use /local/\$USER/\$SLURM_JOB_ID for job-local temporary files." >&2
  exit 2
fi
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

VENV_PATH="${ESMC_VENV:-${PROJECT_ROOT}/venvs/trna-esmc-py311}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "Missing Python environment: ${VENV_PATH}" >&2
  echo "Create it first or submit with ESMC_VENV=/path/to/env." >&2
  exit 2
fi

export PATH="${VENV_PATH}/bin:/usr/local/bin:/usr/bin:/bin"
export CONDA_PREFIX="${VENV_PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/torch-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"

MODEL_ARG="${MODEL_NAME:-biohub/ESMC-6B}"
STAGE_MODEL_TO_LOCAL="${STAGE_MODEL_TO_LOCAL:-1}"
LOCAL_MIN_FREE_GB="${LOCAL_MIN_FREE_GB:-40}"
if [[ "${STAGE_MODEL_TO_LOCAL}" == "1" || "${STAGE_MODEL_TO_LOCAL}" == "true" ]]; then
  SOURCE_MODEL_DIR="${ESMC_MODEL_SNAPSHOT:-}"
  if [[ -z "${SOURCE_MODEL_DIR}" && "${MODEL_ARG}" == "biohub/ESMC-6B" ]]; then
    SOURCE_MODEL_DIR="$(find -L "${HF_HUB_CACHE}/models--biohub--ESMC-6B/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  fi
  if [[ -n "${SOURCE_MODEL_DIR}" && -d "${SOURCE_MODEL_DIR}" ]]; then
    FREE_LOCAL_BYTES="$(df -PB1 "${TMPDIR}" | awk 'NR==2 {print $4}')"
    REQUIRED_LOCAL_BYTES=$(( LOCAL_MIN_FREE_GB * 1024 * 1024 * 1024 ))
    echo "local_stage_free_bytes=${FREE_LOCAL_BYTES} required_bytes=${REQUIRED_LOCAL_BYTES}"
    if (( FREE_LOCAL_BYTES < REQUIRED_LOCAL_BYTES )); then
      echo "ERROR: insufficient /local space for ESM-C staging" >&2
      exit 2
    fi

    LOCAL_MODEL_DIR="${TMPDIR}/esmc_models/ESMC-6B"
    LOCAL_MODEL_TMP="${LOCAL_MODEL_DIR}.tmp"
    rm -rf "${LOCAL_MODEL_TMP}"
    mkdir -p "${LOCAL_MODEL_TMP}" "$(dirname "${LOCAL_MODEL_DIR}")"
    STAGE_START_EPOCH="$(date +%s)"
    echo "stage_start source=${SOURCE_MODEL_DIR} destination=${LOCAL_MODEL_DIR}"
    cp -aL "${SOURCE_MODEL_DIR}/." "${LOCAL_MODEL_TMP}/"
    rm -rf "${LOCAL_MODEL_DIR}"
    mv "${LOCAL_MODEL_TMP}" "${LOCAL_MODEL_DIR}"
    STAGE_END_EPOCH="$(date +%s)"
    STAGE_SECONDS=$(( STAGE_END_EPOCH - STAGE_START_EPOCH ))
    STAGE_BYTES="$(du -sb "${LOCAL_MODEL_DIR}" | awk '{print $1}')"
    echo "stage_complete seconds=${STAGE_SECONDS} bytes=${STAGE_BYTES}"
    MODEL_ARG="${LOCAL_MODEL_DIR}"
  else
    echo "WARNING: requested local staging but source model snapshot was not found; using ${MODEL_ARG}" >&2
  fi
fi

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

python -u scripts/03_poc/06_embed_sequences_esmc.py \
  --fasta data/processed/poc_weak/weak_trna_mod_sequences.faa \
  --output-dir "${OUTPUT_DIR:-data/processed/poc_weak/embeddings/esmc_6b}" \
  --model-name "${MODEL_ARG}" \
  --max-tokens-per-batch "${MAX_TOKENS_PER_BATCH:-4096}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --device-map "${DEVICE_MAP:-auto}" \
  --device cuda
