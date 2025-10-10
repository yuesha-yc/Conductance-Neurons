export JULIA_NUM_THREADS=7

cd ..
julia --project=. src/Runner.jl --from-toml experiments/conductance/single_lif/nu_x_linear.toml --local-threads 7