module SimCore

export SimParams, simulate

"""Minimal parameter carrier. Keep simple, JSON-compatible fields only."""
Base.@kwdef struct SimParams
    T::Float64 = 1000.0
    dt::Float64 = 0.1
    base_rate::Float64 = 100.0
    K_e::Int = 200
    K_i::Int = 50
end

"""
    simulate(p::SimParams) -> NamedTuple

Pure math kernel. No I/O, no global state. Replace with your actual model.
Returns numerics only (Vectors, numbers, small Dicts).
"""
function simulate(p::SimParams)
    n = Int(floor(p.T / p.dt))
    t = collect(0.0:p.dt:(n-1)*p.dt)
    # dummy signal standing in for a membrane potential time series
    V = @. -65.0 + 2.0*sin(2π * (p.base_rate/1000.0) * (t/1000.0))
    # tiny placeholder metrics
    mean_V = sum(V) / length(V)
    var_V  = sum((V .- mean_V).^2) / (length(V)-1)
    return (t=t, V=V, metrics=Dict("mean_V"=>mean_V, "var_V"=>var_V))
end

end # module
