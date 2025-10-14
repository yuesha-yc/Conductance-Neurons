#!/usr/bin/env julia

using Dates, Printf, JSON3

include("SimCore.jl"); using .SimCore
include("Experiments.jl"); using .Experiments
include("Sweeps.jl"); using .Sweeps
include("Analysis.jl"); using .Analysis

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
    spec   = Sweeps.parse_experiment_toml(path)
    trials = Sweeps.expand_trials(spec)
    rel    = replace(path, r"^experiments/" => "") |> x -> replace(x, ".toml" => "")
    tag    = get(spec.meta, "tag", nothing)
    mkpath(joinpath(outputs_root, rel))
    info   = Experiments.make_exp_run_dir(outputs_root, rel; tag=tag)
    outdir = info.outdir

    mkpath(outdir); cp(path, joinpath(outdir, "spec.toml"), force=true)
    Experiments.write_manifest_skeleton(outdir; spec_path=path, n_trials=length(trials))

    @info "CPU / Threads" cpu_threads=Sys.CPU_THREADS julia_threads=Threads.nthreads()

    if local_threads > 0
        nworkers = min(local_threads, Threads.nthreads())
        if nworkers < local_threads
            @warn "Requested more local threads than Julia has; capping." requested=local_threads actually=nworkers
        end

        @info "Dispatching locally" n_trials=length(trials) workers=nworkers outdir

        ch = Channel{Tuple{Int,NamedTuple}}(length(trials))
        for (i,t) in enumerate(trials); put!(ch, (i,t)); end
        close(ch)

        # Optional: a lock if _update_manifest_trial! touches the same file
        manifest_lock = ReentrantLock()

        # record time
        start_ns = time_ns()
        @sync for _ in 1:nworkers
            Threads.@spawn begin
                for (i,t) in ch
                    trial_id = @sprintf("%05d", i)
                    tdir = joinpath(outdir, "trials", trial_id)
                    st = Experiments.run_trial(t, tdir)
                    Base.lock(manifest_lock) do
                        Experiments._update_manifest_trial!(outdir, trial_id;
                            status=String(st),
                            seed = hasproperty(t, :seed) ? getfield(t, :seed) : nothing,
                            relpath=joinpath("trials", trial_id))
                    end
                end
            end
        end
        elapsed_ns = time_ns() - start_ns
        println(@sprintf("Elapsed time for parallel experiments: %.3f s", elapsed_ns / 1e9))
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
    start_ns = time_ns()
    try
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
    finally
        elapsed_ns = time_ns() - start_ns
        println(@sprintf("Total elapsed time entirely: %.3f s", elapsed_ns / 1e9))
    end
end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == abspath(Base.source_path())
    main(ARGS)
end
