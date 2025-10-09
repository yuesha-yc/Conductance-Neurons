#!/usr/bin/env julia

ENV["GKSwstype"] = "100"

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "Analysis.jl"))
using .Analysis, Plots

function resolve_run_dir()
    if !isempty(ARGS)
        candidate = ARGS[1]
        abs_candidate = isabspath(candidate) ? candidate : joinpath(ROOT, candidate)
        if isdir(abs_candidate) && isfile(joinpath(abs_candidate, "manifest.json"))
            return abs_candidate
        end
        base_dir = joinpath(ROOT, "outputs", candidate)
        if isdir(base_dir)
            if isfile(joinpath(base_dir, "manifest.json"))
                return base_dir
            end
            runs = filter(name -> isdir(joinpath(base_dir, name)), readdir(base_dir))
            isempty(runs) && error("No runs found under $base_dir")
            return joinpath(base_dir, sort(runs)[end])
        else
            error("Could not resolve run directory from input: $candidate")
        end
    end
    default_base = joinpath(ROOT, "outputs", "conductance", "pilot")
    runs = isdir(default_base) ? filter(name -> isdir(joinpath(default_base, name)), readdir(default_base)) : String[]
    isempty(runs) && error("No runs available under $default_base. Pass a run path as an argument.")
    return joinpath(default_base, sort(runs)[end])
end

function main()
    run_dir = resolve_run_dir()
    isdir(run_dir) || error("Run directory not found: $run_dir")

    exp = Analysis.ExperimentHandle(dirname(run_dir), run_dir)
    trials = Analysis.list_trials(exp)
    ok_idx = findfirst(tr -> get(tr, "status", "") == "ok", trials)
    ok_idx === nothing && error("No ok trials found for run at $run_dir")
    trial_id = trials[ok_idx]["trial_id"]

    plt = Analysis.plot_series(exp, trial_id)
    trace_path = joinpath(@__DIR__, "sample_plot.png")
    savefig(plt, trace_path)
    println("Saved plot for trial $trial_id to $trace_path")

    scatter = Analysis.plot_param_scatter(run_dir, "t", "V", "base_rate")
    scatter_path = joinpath(@__DIR__, "sample_scatter.png")
    savefig(scatter, scatter_path)
    println("Saved scatter plot to $scatter_path")
end

main()
