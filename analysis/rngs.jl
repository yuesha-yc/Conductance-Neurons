using Pkg; Pkg.activate(".."); Pkg.instantiate()

using Random, Distributions, Statistics

seed = 0
master = MersenneTwister(seed)
mk() = MersenneTwister(rand(master, UInt))  # derive independent sub-seeds
ind_e_rng = mk()  # independent E only
ind_i_rng = mk()  # independent I only
sh_e_rng = mk()   # shared E only
sh_i_rng = mk()   # shared I only

# @info rand(ind_e_rng, 5)
# @info rand(ind_e_rng, 5)
# @info rand(ind_i_rng, 5)
# @info rand(sh_e_rng, 5)
# @info rand(sh_i_rng, 5)

# generate poisson processes

# Poisson dists
dE = Poisson(1)
dI = Poisson(1)

# common input
dE_C = Poisson(1)
dI_C = Poisson(1)

for _ in 1:10
    s_e_ind = rand(ind_e_rng, dE)
    s_i_ind = rand(ind_e_rng, dI)
    s_e_c = rand(ind_e_rng, dE_C)
    s_i_c = rand(ind_e_rng, dI_C)
    @info "spikes" s_e_ind s_i_ind s_e_c s_i_c
end