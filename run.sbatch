#!/bin/bash
#SBATCH --job-name=conductance-neurons            # Job name
#SBATCH --account=pi-bdoiron              # Account under which to submit
#SBATCH --partition=voltron               # Partition/queue to submit to
#SBATCH --qos=voltron
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --time=02:00:00                  # Time limit
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --output=/project/bdoiron/yichenw/out/job_output.%j.txt  # Standard output file
#SBATCH --error=/project/bdoiron/yichenw/out/job_error.%j.err    # Standard error file


# print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Time: $(date)"
echo "Current directory: $(pwd)"

# module load julia

# Run Pkg.instantiate() in Julia REPL and install JSON3 package

# julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Execute the Julia script with parameters
julia -t auto --project=. src/Runner.jl --from-toml experiments/conductance/single_lif/nu_x_realistic_candidate3.toml --local-threads 10