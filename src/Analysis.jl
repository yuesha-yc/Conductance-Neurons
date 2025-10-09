module Analysis

using JSON3, JLD2, Printf
export load_experiment, list_trials, load_trial, select, plot_series

const _plots_pkgid = Base.PkgId(Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots")

function _load_plots()
    try
        return Base.require(_plots_pkgid)
    catch e
        error("Plots.jl is required for plot_series but could not be loaded: $(sprint(showerror, e))")
    end
end

function _json_to_data(x)
    if x isa JSON3.Object
        d = Dict{String,Any}()
        for (k,v) in pairs(x)
            d[String(k)] = _json_to_data(v)
        end
        return d
    elseif x isa JSON3.Array
        return [_json_to_data(v) for v in x]
    else
        return x
    end
end

struct ExperimentHandle
    root::String
    runpath::String  # outputs/<rel>/<run_id>
end

"""Resolve latest run under outputs/<spec_rel_dir> that matches the TOML path."""
function load_experiment(spec_toml_path::AbstractString)
    # infer outputs subdir by mirroring experiments/ path
    rel = replace(spec_toml_path, r"^experiments/" => "") |> x -> replace(x, ".toml" => "")
    outdir = joinpath("outputs", rel)
    isdir(outdir) || error("No outputs for $spec_toml_path")
    # pick the newest run directory
    runs = filter(name -> isdir(joinpath(outdir, name)), readdir(outdir))
    isempty(runs) && error("No experiment runs found under $outdir")
    # newest by name (timestamp prefix)
    run = sort(runs)[end]
    return ExperimentHandle(outdir, joinpath(outdir, run))
end

function list_trials(exp::ExperimentHandle)
    mpath = joinpath(exp.runpath, "manifest.json")
    m = _json_to_data(JSON3.read(read(mpath, String)))
    return get(m, "trials", Vector{Any}())
end

function load_trial(exp::ExperimentHandle, trial_id::AbstractString)
    tdir = joinpath(exp.runpath, "trials", trial_id)
    params = _json_to_data(JSON3.read(read(joinpath(tdir, "params.json"), String)))
    meta = _json_to_data(JSON3.read(read(joinpath(tdir, "meta.json"), String)))
    out = Dict{String,Any}()
    if isfile(joinpath(tdir, "results.jld2"))
        @load joinpath(tdir, "results.jld2") out
    end
    return Dict("params"=>params, "meta"=>meta, "out"=>out)
end

function select(exp::ExperimentHandle, key::AbstractString; only::Symbol=:ok)
    rows = String[]; vals = Any[]
    for tr in list_trials(exp)
        if only == :ok && tr["status"] != "ok"; continue; end
        tdir = joinpath(exp.runpath, tr["relpath"])
        if isfile(joinpath(tdir, "results.jld2"))
            out = Dict{String,Any}(); @load joinpath(tdir, "results.jld2") out
            if haskey(out, :metrics) && haskey(out[:metrics], key)
                push!(rows, tr["trial_id"])
                push!(vals, out[:metrics][key])
            end
        end
    end
    return rows, vals
end

function plot_series(exp::ExperimentHandle, trial_id; x::AbstractString="t", y::AbstractString="V")
    plots = _load_plots()
    tdir = joinpath(exp.runpath, "trials", trial_id)
    out = Dict{String,Any}(); @load joinpath(tdir, "results.jld2") out
    xs = out[Symbol(x)]; ys = out[Symbol(y)]
    Base.invokelatest(plots.plot, xs, ys; xlabel=x, ylabel=y, title="$(basename(exp.runpath))/$trial_id")
end

end # module
