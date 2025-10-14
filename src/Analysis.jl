module Analysis

using JSON3, Random, Statistics

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

"""
mean_matched_ff(counts; nbins=20, n_reps=50, seed=0)

- counts: (T,P,R) where P are (unit×condition) pairs and R are repeats,
          or (T,U,C,R) which will be reshaped to (T,P,R) with P=U*C.
- returns: Vector{Float64} of length T (mean-matched FF per time).
"""
function mean_matched_ff(counts; nbins=20, n_reps=50, seed=0)
    A = counts
    nd = ndims(A)
    if nd == 4
        T, U, C, R = size(A)
        A = reshape(A, T, U*C, R)  # (T,P,R)
    elseif nd == 3
        T, P, R = size(A)
    else
        error("counts must be (T,P,R) or (T,U,C,R)")
    end
    T, P, R = size(A)

    # per-time, per-pair stats across repeats
    means = [mean(@view A[t,p,:]) for t in 1:T, p in 1:P]
    vars  = [var(@view A[t,p,:])  for t in 1:T, p in 1:P]

    # pooled positive means → common bins
    pool = [μ for μ in vec(means) if μ > 0]
    mn, mx = minimum(pool), maximum(pool)
    edges = collect(range(mn, mx, length=nbins+1))
    nb = length(edges) - 1

    # collect members per time/bin (indices into pairs)
    bins = [[Int[] for _ in 1:nb] for _ in 1:T]
    for t in 1:T, p in 1:P
        μ = means[t,p]
        if μ > 0
            b = clamp(searchsortedlast(edges, μ), 1, nb)
            push!(bins[t][b], p)
        end
    end

    # greatest common per-bin counts across times
    common = [minimum([length(bins[t][b]) for t in 1:T]) for b in 1:nb]

    # subsample repeatedly and average FF
    rng = MersenneTwister(seed)
    ff = zeros(Float64, T)
    for rep in 1:n_reps
        for t in 1:T
            sel = Int[]
            for b in 1:nb
                h = common[b]
                if h > 0
                    members = bins[t][b]
                    idx = (length(members) <= h) ? collect(1:length(members)) : randperm(rng, length(members))[1:h]
                    append!(sel, members[idx])
                end
            end
            if !isempty(sel)
                μs = means[t, sel]
                vs = vars[t,  sel]
                ff[t] += mean(vs ./ μs)
            end
        end
    end
    ff ./= n_reps
    return ff
end





end # module
