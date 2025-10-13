# Example: generate trials JSONL from a spec
cd ..
julia --project=. src/Runner.jl \
  --from-toml experiments/conductance/single_lif/nu_x_realistic_candidate4.toml \
  --emit-jsonl jsonl/nu_x_realistic_candidate4.jsonl
