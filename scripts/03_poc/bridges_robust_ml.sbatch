#!/usr/bin/env bash
# CPU-only robust weak-POC classifier training from frozen ESM-C embeddings.

#SBATCH -J trna-robust-ml
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH -t 00:30:00
#SBATCH -o robust_ml_%j.out
#SBATCH -e robust_ml_%j.err
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
  exit 2
fi
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

VENV_PATH="${ESMC_VENV:-${PROJECT_ROOT}/venvs/trna-esmc-py311}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "Missing Python environment: ${VENV_PATH}" >&2
  exit 2
fi

export PATH="${VENV_PATH}/bin:/usr/local/bin:/usr/bin:/bin"
export CONDA_PREFIX="${VENV_PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/cache/matplotlib}"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

python -u scripts/03_poc/08_train_weak_embedding_robust_models.py \
  --embedding-dir "${EMBEDDING_DIR:-data/processed/poc_weak/embeddings/esmc_6b}" \
  --task "${TASK:-all}" \
  --run-name "${RUN_NAME:-esmc6b_robust_logreg}" \
  --mechanism-scope "${MECHANISM_SCOPE:-mechanism_labeled}" \
  --min-train-positives "${MIN_TRAIN_POSITIVES:-10}" \
  --min-val-positives "${MIN_VAL_POSITIVES:-1}" \
  --min-test-positives "${MIN_TEST_POSITIVES:-1}" \
  --c-grid "${C_GRID:-0.003,0.01,0.03,0.1,0.3,1,3,10}" \
  --rank-metric "${RANK_METRIC:-aupr}" \
  --threshold-metric "${THRESHOLD_METRIC:-f1}"
