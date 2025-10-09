#!/usr/bin/env julia

ENV["GKSwstype"] = "100"

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "Analysis.jl"))
using .Analysis, Plots

function main()
    run_rel = "sin_waves/20251009-163705-25UYnxic-dummy"
    run_dir = joinpath(ROOT, "outputs", run_rel)
    isdir(run_dir) || error("Run directory not found: $run_dir")

    exp = Analysis.ExperimentHandle(dirname(run_dir), run_dir)
    trials = Analysis.list_trials(exp)
    ok_idx = findfirst(tr -> get(tr, "status", "") == "ok", trials)
    ok_idx === nothing && error("No ok trials found for run at $run_dir")
    trial_id = trials[ok_idx]["trial_id"]

    plt = Analysis.plot_series(exp, trial_id)
    out_path = joinpath(@__DIR__, "sample_plot.png")
    savefig(plt, out_path)
    println("Saved plot for trial $trial_id to $out_path")
end

main()
