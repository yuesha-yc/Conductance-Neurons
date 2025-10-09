module SimCore

export SimParams, simulate

"""Minimal parameter carrier. Keep simple, JSON-compatible fields only."""
Base.@kwdef struct SimParams
    T::Float64 = 1000.0
    dt::Float64 = 0.1
    base_rate::Float64 = 100.0
    K_e::Int = 200
    K_i::Int = 50
    seed::Int = 0
    model::String = "sin_waves"
    save_downsampled::Bool = false
    E_L::Float64 = -65.0
    note::String = ""
end

"""Dummy sine wave generator standing in for a membrane potential time series."""
function sin_waves(p::SimParams)
    n = Int(floor(p.T / p.dt))
    t = collect(0.0:p.dt:(n-1)*p.dt)
    V = @. -65.0 + 2.0*sin(2π * (p.base_rate/1000.0) * (t/1000.0))
    mean_V = sum(V) / length(V)
    var_V  = sum((V .- mean_V).^2) / (length(V)-1)
    return (t=t, V=V, metrics=Dict("mean_V"=>mean_V, "var_V"=>var_V))
end

"""
    simulate(p::SimParams) -> NamedTuple

Entry point for selecting the underlying simulation kernel based on `p.model`.
"""
function simulate(p::SimParams)
    model_key = lowercase(p.model)
    if isempty(model_key) || model_key == "sin_waves"
        return sin_waves(p)
    else
        error("Unsupported simulation model: $(p.model)")
    end
end

end # module
