cd ..
julia -t auto --project=. src/Runner.jl --from-toml experiments/conductance/single_lif/nu_x_linear_k1.toml --local-threads 10