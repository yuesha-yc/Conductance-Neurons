#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# print PROJECT_ROOT
echo "Project root: ${PROJECT_ROOT}"
JSONL_PATH="${PROJECT_ROOT}/jsonl/nu_x_realistic_candidate4.jsonl"
SBATCH_FILE="${PROJECT_ROOT}/batch/run_array.sbatch"

if [[ ! -s "${JSONL_PATH}" ]]; then
  echo "ERROR: JSONL not found or empty at ${JSONL_PATH}" >&2
  exit 1
fi

N=$(wc -l < "${JSONL_PATH}")
echo "Submitting array of ${N} tasks from ${JSONL_PATH}"

# Override the --array value in the sbatch file with the exact N:
sbatch --array=1-"${N}" "${SBATCH_FILE}"
