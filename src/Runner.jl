#!/usr/bin/env julia

using .Sweeps, .Experiments
using Dates, Printf, JSON3

include("SimCore.jl"); using .SimCore
include("Experiments.jl"); using .Experiments
include("Sweeps.jl"); using .Sweeps
include("Analysis.jl")

function usage()
    println("""
NeuroPipeline Runner

USAGE:
  julia --project=. src/Runner.jl --from-toml <path> [--local-threads N]
  julia --project=. src/Runner.jl --from-toml <path> --emit-jsonl <out.jsonl>
  julia --project=. src/Runner.jl --from-json <params.json or JSON string>
""")
end

function run_from_json(json_str_or_path::AbstractString)
    s = isfile(json_str_or_path) ? read(json_str_or_path, String) : json_str_or_path
    params = JSON3.read(s) |> Dict |> NamedTuple
    trial_dir = mktempdir()
    return Experiments.run_trial(params, trial_dir)
end

function run_experiment_from_toml(path::AbstractString; local_threads::Int=0, outputs_root::String="outputs")
    spec = Sweeps.parse_experiment_toml(path)
    trials = Sweeps.expand_trials(spec)
    rel = replace(path, r"^experiments/" => "") |> x -> replace(x, ".toml" => "")
    tag = get(spec.meta, "tag", nothing)
    mkpath(joinpath(outputs_root, rel))
    info = Experiments.make_exp_run_dir(outputs_root, rel; tag=tag)
    outdir = info.outdir
    # copy spec
    mkpath(outdir); cp(path, joinpath(outdir, "spec.toml"), force=true)
    Experiments.write_manifest_skeleton(outdir; spec_path=path, n_trials=length(trials))

    if local_threads > 0
        # simple thread pool with a channel
        @info "Dispatching locally" n_trials=length(trials) threads=local_threads outdir
        ch = Channel{Tuple{Int,NamedTuple}}(length(trials))
        for (i,t) in enumerate(trials); put!(ch, (i,t)); end
        close(ch)
        Threads.@threads for _ in 1:local_threads
            for (i,t) in ch
                trial_id = @sprintf("%05d", i)
                tdir = joinpath(outdir, "trials", trial_id)
                st = Experiments.run_trial(t, tdir)
                Experiments._update_manifest_trial!(outdir, trial_id; status=String(st), seed=getfield(t, :seed), relpath=joinpath("trials", trial_id))
            end
        end
    else
        @info "No local threads specified; just expanded spec" n_trials=length(trials)
        println("Use --emit-jsonl to generate a JSONL for SLURM arrays.")
    end
    Experiments.finalize_manifest(outdir)
    println("Experiment run at: $outdir")
    return outdir
end

function emit_jsonl_from_toml(path::AbstractString, outpath::AbstractString)
    spec = Sweeps.parse_experiment_toml(path)
    trials = Sweeps.expand_trials(spec)
    Sweeps.trials_to_jsonl(trials, outpath)
    println("Wrote JSONL: $outpath  (trials=$(length(trials)))")
end

function main(args)
    if isempty(args); usage(); return; end
    i = 1
    from_toml = nothing; from_json = nothing; emit_jsonl = nothing
    local_threads = 0
    while i <= length(args)
        arg = args[i]
        if arg == "--from-toml"; from_toml = args[i+1]; i += 2
        elseif arg == "--from-json"; from_json = args[i+1]; i += 2
        elseif arg == "--emit-jsonl"; emit_jsonl = args[i+1]; i += 2
        elseif arg == "--local-threads"; local_threads = parse(Int, args[i+1]); i += 2
        else
            println("Unknown arg: $arg"); usage(); return
        end
    end
    if from_json !== nothing
        run_from_json(from_json)
    elseif from_toml !== nothing && emit_jsonl !== nothing
        emit_jsonl_from_toml(from_toml, emit_jsonl)
    elseif from_toml !== nothing
        run_experiment_from_toml(from_toml; local_threads=local_threads)
    else
        usage()
    end
end

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
