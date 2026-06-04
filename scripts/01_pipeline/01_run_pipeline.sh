#!/usr/bin/env bash
# Run the complete tRNA modification protein prediction dataset pipeline.

set -euo pipefail

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# Override these with environment variables or edit configs/config.yaml.
CONFIG_FILE="${CONFIG_FILE:-configs/config.yaml}"
SNAKEFILE="${SNAKEFILE:-workflow/Snakefile}"
CORES="${CORES:-8}"
USE_CONDA="${USE_CONDA:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${USE_CONDA}" == "true" ]]; then
  snakemake -s "${SNAKEFILE}" --configfile "${CONFIG_FILE}" --cores "${CORES}" --use-conda
else
  snakemake -s "${SNAKEFILE}" --configfile "${CONFIG_FILE}" --cores "${CORES}"
fi
