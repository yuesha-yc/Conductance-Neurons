module Analysis

using JSON3, JLD2, Printf, Plots, Logging
export load_experiment, list_trials, load_trial, select, plot_series, plot_param_scatter

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

function plot_param_scatter(exp::ExperimentHandle, x::AbstractString, y::AbstractString, label_key::AbstractString; only::Symbol=:ok)
    # Gather data across trials
    x_vals = Float64[]; y_vals = Float64[]; labels = String[]
    for tr in list_trials(exp)
        if only == :ok && tr["status"] != "ok"; continue; end
        tdir = joinpath(exp.runpath, tr["relpath"])
        if isfile(joinpath(tdir, "results.jld2"))
            out = Dict{String,Any}(); @load joinpath(tdir, "results.jld2") out
            # out keys are the keys of x and y
            if haskey(out, Symbol(x)) && haskey(out, Symbol(y))
                push!(x_vals, out[Symbol(x)])
                push!(y_vals, out[Symbol(y)])
                push!(labels, string(get(tr, label_key, "")))
            end
        end
    end
end

function plot_param_scatter(run_dir::AbstractString, x::AbstractString, y::AbstractString, label_key::AbstractString; kwargs...)
    isdir(run_dir) || error("Run directory not found: $run_dir")
    exp = ExperimentHandle(dirname(run_dir), run_dir)
    return plot_param_scatter(exp, x, y, label_key; kwargs...)
end

function plot_series(exp::ExperimentHandle, trial_id; x::AbstractString="t", y::AbstractString="V")
    tdir = joinpath(exp.runpath, "trials", trial_id)
    out = Dict{String,Any}(); @load joinpath(tdir, "results.jld2") out
    xs = out[Symbol(x)]; ys = out[Symbol(y)]
    plot(xs, ys; xlabel=x, ylabel=y, title="$(basename(exp.runpath))/$trial_id")
end

end # module
