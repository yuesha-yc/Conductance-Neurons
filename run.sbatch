#!/bin/bash
#SBATCH --job-name=yichen            # Job name
#SBATCH --account=doiron-lab               # Account under which to submit
#SBATCH --partition=tier1q               # Partition/queue to submit to
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --time=00:30:30                  # Time limit
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=80gb                      # Memory per node
#SBATCH --output=/gpfs/data/doiron-lab/yichen/out/job_output.%j.txt  # Standard output file
#SBATCH --error=/gpfs/data/doiron-lab/yichen/out/job_error.%j.err    # Standard error file

module load julia

# Execute the Julia script with parameters
julia -t auto --project=. src/Runner.jl --from-toml experiments/conductance/single_lif/nu_x_realistic_candidate3.toml --local-threads 10