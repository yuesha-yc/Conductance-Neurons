module Sweeps

using TOML, JSON3
export parse_experiment_toml, expand_trials, trials_to_jsonl

struct ExperimentSpec
    meta::Dict{String,Any}
    defaults::Dict{String,Any}
    fixed::Dict{String,Any}
    grid::Dict{String,Any}
    explicit::Vector{Dict{String,Any}}
end

function parse_experiment_toml(path::AbstractString)
    t = TOML.parsefile(path)
    meta = get(t, "experiment", Dict{String,Any}())
    defaults = get(t, "defaults", Dict{String,Any}())
    fixed = get(t, "fixed", Dict{String,Any}())
    grid = get(t, "grid", Dict{String,Any}())
    explicit = get(t, "trials", Vector{Dict{String,Any}}())
    # validations (lightweight)
    for (k,v) in grid
        isa(v, AbstractVector) || error("[grid.$k] must be an array")
    end
    return ExperimentSpec(meta, defaults, fixed, grid, explicit)
end

# cartesian product over a Dict of arrays, returning Vector{Dict}
function _cartesian(dict_of_arrays::Dict{String,Any})
    if isempty(dict_of_arrays)
        return Vector{Dict{String,Any}}()
    end
    keys_order = collect(keys(dict_of_arrays))
    values_arrays = [collect(dict_of_arrays[k]) for k in keys_order]
    idx = fill(1, length(keys_order))
    total = prod(length.(values_arrays))
    out = Vector{Dict{String,Any}}(undef, total)
    for i in 1:total
        d = Dict{String,Any}()
        for (j,k) in enumerate(keys_order)
            d[k] = values_arrays[j][idx[j]]
        end
        out[i] = d
        # increment idx
        for j in length(idx):-1:1
            idx[j] += 1
            if idx[j] <= length(values_arrays[j])
                break
            else
                idx[j] = 1
            end
        end
    end
    return out
end

function _merge_params(defaults::Dict, x::Dict, fixed::Dict)
    # defaults < x < fixed
    d = Dict{Symbol,Any}()
    for (k,v) in defaults; d[Symbol(k)] = v; end
    for (k,v) in x;        d[Symbol(k)] = v; end
    for (k,v) in fixed;    d[Symbol(k)] = v; end
    return d
end

function expand_trials(spec::ExperimentSpec)
    grid_trials = _cartesian(spec.grid)
    all_trials = vcat(grid_trials, spec.explicit)
    trials = Vector{NamedTuple}(undef, length(all_trials))
    base_seed = get(spec.meta, "base_seed", nothing)
    for (i, trialdict) in enumerate(all_trials)
        merged = _merge_params(spec.defaults, trialdict, spec.fixed)
        seed = isnothing(base_seed) ? i : (Int(base_seed) + i - 1)
        merged[:seed] = seed
        trials[i] = (; merged...)
    end
    return trials
end

function trials_to_jsonl(trials, outpath::AbstractString)
    mkpath(dirname(outpath))
    open(outpath, "w") do io
        for t in trials
            JSON3.write(io, Dict(pairs(t)))
            write(io, "\n")
        end
    end
    return outpath
end

end # module
