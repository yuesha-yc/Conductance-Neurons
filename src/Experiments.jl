module Experiments

using Dates, JSON3, JLD2, Random, Logging, Printf
import Base: mkdir

export run_trial, make_exp_run_dir, write_manifest_skeleton, append_summary_row, finalize_manifest

"""Create experiment run directory (id = timestamp + 8-char sha-like tag)"""
function make_exp_run_dir(root::AbstractString, exp_relpath::AbstractString; tag::Union{Nothing,String}=nothing)
    ts = Dates.format(now(), "yyyymmdd-HHMMSS")
    # simple tag; for full reproducibility use content hash of spec file
    short = randstring(8)
    run_id = isnothing(tag) ? "$ts-$short" : "$ts-$short-$(tag)"
    outdir = joinpath(root, exp_relpath, run_id)
    mkpath(joinpath(outdir, "trials"))
    return (outdir=outdir, run_id=run_id)
end

function _write_json(path, obj)
    open(path, "w") do io
        JSON3.write(io, obj; indent=2)
    end
end

function write_manifest_skeleton(outdir; spec_path::AbstractString, n_trials::Int)
    manifest = Dict(
        "experiment_run_id" => splitdir(outdir)[2],
        "spec_path" => spec_path,
        "n_trials" => n_trials,
        "trials" => Vector{Any}(),
    )
    _write_json(joinpath(outdir, "manifest.json"), manifest)
end

function _update_manifest_trial!(outdir, trial_id; status::String, seed::Int, relpath::String)
    mpath = joinpath(outdir, "manifest.json")
    m = JSON3.read(read(mpath, String)) |> Dict
    push!(m["trials"], Dict("trial_id"=>trial_id, "status"=>status, "seed"=>seed, "relpath"=>relpath))
    _write_json(mpath, m)
end

function append_summary_row(outdir, trial_id, metrics::Dict)
    spath = joinpath(outdir, "summary.csv")
    header = "trial_id," * join(collect(keys(metrics)), ",") * "\n"
    row = trial_id * "," * join(string.(collect(values(metrics))), ",") * "\n"
    if !isfile(spath)
        open(spath, "w") do io
            write(io, header)
            write(io, row)
        end
    else
        open(spath, "a") do io
            write(io, row)
        end
    end
end

"""Mark manifest as finished (optional)."""
finalize_manifest(outdir) = nothing

"""
    run_trial(params::NamedTuple, trial_dir::AbstractString) -> Symbol

Seed RNG, call SimCore.simulate, write params/results/meta/log.
"""
function run_trial(params::NamedTuple, trial_dir::AbstractString)
    @info "Running trial" trial_dir
    mkpath(trial_dir)
    # params.json
    _write_json(joinpath(trial_dir, "params.json"), Dict(pairs(params)))

    # meta (start)
    meta = Dict(
        "status" => "running",
        "julia_version" => VERSION |> string,
        "started_at" => Dates.format(now(), Dates.ISODateTimeFormat),
    )
    _write_json(joinpath(trial_dir, "meta.json"), meta)

    # log.txt (minimal)
    open(joinpath(trial_dir, "log.txt"), "w") do logio
        try
            # seed
            if haskey(Dict(pairs(params)), :seed)
                Random.seed!(getfield(params, :seed))
            end
            # simulate
            using ..SimCore
            sp = SimCore.SimParams(; Dict(pairs(params))...)
            t0 = time();
            out = SimCore.simulate(sp)
            elapsed = time() - t0
            # write results
            @save joinpath(trial_dir, "results.jld2") out
            # update meta
            meta["status"] = "ok"
            meta["elapsed_sec"] = elapsed
            _write_json(joinpath(trial_dir, "meta.json"), meta)
            # summary
            if haskey(out, :metrics)
                append_summary_row(dirname(trial_dir), basename(trial_dir), out[:metrics])
            end
            write(logio, "OK\n")
            return :ok
        catch e
            write(logio, "ERROR: " * sprint(showerror, e) * "\n")
            meta["status"] = "error"
            meta["error"] = sprint(showerror, e)
            _write_json(joinpath(trial_dir, "meta.json"), meta)
            return :error
        end
    end
end

end # module
