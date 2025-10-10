cd ..
julia -t auto --project=. src/Runner.jl --from-toml experiments/conductance/single_lif/nu_x_eta.toml --local-threads 10