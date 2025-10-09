#!/usr/bin/env bash
#SBATCH -J neuro_sweep
#SBATCH -o slurm-%A_%a.out
#SBATCH -t 02:00:00
#SBATCH -p compute
#SBATCH --array=1-1

set -euo pipefail
JSONL=${1:-}
if [[ -z "$JSONL" ]]; then
  echo "Usage: sbatch scripts/sbatch_array.sh sweeps/foo.jsonl"; exit 1
fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JSONL")
module load julia || true
julia --project=. -e 'include("src/Runner.jl"); Main.run_from_json("""'"$LINE"'""")'
