cd ..
julia --project=. src/Runner.jl --from-toml experiments/sin_waves_small.toml --local-threads 8