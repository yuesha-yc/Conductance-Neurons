#!/usr/bin/env julia

ENV["GKSwstype"] = "100"

const ROOT = normpath(joinpath(@__DIR__, ".."))

using JSON3, JLD2, Plots, Statistics

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

"""
    plot_experiment(experiment_dir::String, xname::String, yname::String, key::String)

Recursively searches `experiment_dir` for subdirectories containing both
`params.json` and `results.jld2`. Extracts the variables `xname` and `yname`
(vectors) from result files, and the scalar parameter `key` from JSON files,
then plots y vs x for each distinct key value.
"""
function plot_experiment(experiment_dir::String, xname::String, yname::String, key::String)
    data = Dict{Any, Vector{Tuple{Vector{Float64}, Vector{Float64}}}}()

    for (root, dirs, files) in walkdir(experiment_dir)
        if "params.json" in files && "results.jld2" in files
            # --- Load parameters
            param_path = joinpath(root, "params.json")
            params = JSON3.read(open(param_path, "r"))
            key_value = haskey(params, key) ? params[key] : missing

            # --- Load results safely
            result_path = joinpath(root, "results.jld2")
            @load result_path out

            # the result should contain vectors named xname and yname
            if !(haskey(out, xname) && haskey(out, yname))
                @warn "Missing $xname or $yname in $result_path"
                continue
            end

            x = out[xname]
            y = out[yname]

            if !(x isa AbstractVector && y isa AbstractVector)
                @warn "x or y not a vector in $result_path"
                continue
            end

            if !haskey(data, key_value)
                data[key_value] = []
            end
            push!(data[key_value], (x, y))
        end
    end

    # --- Plot
    plt = plot(title="Experiment: $(basename(experiment_dir))",
            xlabel=xname, ylabel=yname, legend=:outertopright)

    for (key_value, pairs) in sort(collect(data))
        for (x, y) in pairs
            plot!(plt, x, y, label="key=$key_value", lw=2)
        end
    end

    # display(plt)

    # save the plot
    save_path = joinpath(experiment_dir, "summary_plot_$(yname)_vs_$(xname)_over_$(key).png")
    png(plt, save_path)
    @info "Plot saved to $save_path"
end


function main()
    # run_dir = resolve_run_dir()
    candidate = "sin_waves/20251009-181938-ynbzJ5HS-dummy"
    run_dir = joinpath(ROOT, "outputs", candidate)
    if isdir(run_dir)
        @info "Plotting finished for " run_dir
        plot_experiment(run_dir, "t", "V", "base_rate")
    else
        error("Run directory not found: $run_dir")
    end
end

main()
