#!/bin/bash
#SBATCH --job-name=conductance-neurons
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Default; you can override with sbatch --array=1-N at submit time:
#SBATCH --array=1-2
#SBATCH -o /home/yichenw/logs/conductance_neurons_%A_%a.out
#SBATCH -e /home/yichenw/logs/conductance_neurons_%A_%a.err
# email notifications (BEGIN,END,FAIL,ALL)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yichenw@uchicago.edu


set -euo pipefail

# ---- user knobs ----
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}/.."
JSONL_PATH="${PROJECT_ROOT}/jsonl/nu_x_realistic_candidate3.jsonl"
JULIA_PROJECT="${PROJECT_ROOT}"   # so --project=. works from $PROJECT_ROOT
# --------------------

echo "Master JobID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)  Cores: ${SLURM_CPUS_PER_TASK}  Mem: ${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-unknown}}"

# Make sure logs dir exists (if not already)
mkdir -p "${PROJECT_ROOT}/logs"

# Export Julia threads = cpus-per-task
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"

# Safety: ensure JSONL exists and is non-empty
if [[ ! -s "${JSONL_PATH}" ]]; then
  echo "ERROR: JSONL not found or empty at ${JSONL_PATH}" >&2
  exit 2
fi

# Extract the Nth line (1-indexed) for this array task
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JSONL_PATH}")"
if [[ -z "${LINE}" ]]; then
  echo "ERROR: No line for task ${SLURM_ARRAY_TASK_ID} in ${JSONL_PATH}" >&2
  exit 3
fi

# Optional: basic validation (requires jq if present)
if command -v jq >/dev/null 2>&1; then
  echo "${LINE}" | jq . >/dev/null || { echo "ERROR: line ${SLURM_ARRAY_TASK_ID} is not valid JSON"; exit 4; }
fi

# module purge || true
# module load julia || true

# Run the trial. We pass the whole JSON object string to --from-json.
# Runner.jl will create its own per-trial temp dir via run_from_json().
cd "${PROJECT_ROOT}"
julia --project="${JULIA_PROJECT}" src/Runner.jl --from-json "${LINE}"
