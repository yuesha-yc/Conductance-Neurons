module SimCore

export AbstractSimParams, SimParams, make_params, simulate

"""Marker supertype for all simulation parameter carriers."""
abstract type AbstractSimParams end

"""Minimal parameter carrier for the sine-wave toy model."""
Base.@kwdef struct SimParams <: AbstractSimParams
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
function sin_waves(p::SimParams)::Dict{String,Any}
    n = Int(floor(p.T / p.dt))
    t = collect(0.0:p.dt:(n-1)*p.dt)
    V = @. -65.0 + 2.0*sin(2π * (p.base_rate/1000.0) * (t/1000.0))
    mean_V = sum(V) / length(V)
    var_V  = sum((V .- mean_V).^2) / (length(V)-1)
    return Dict("t"=>t, "V"=>V, "mean_V"=>mean_V, "var_V"=>var_V)
end

"""
    make_params(params::NamedTuple) -> AbstractSimParams

Select and build the appropriate parameter struct for a simulation model.
"""
function make_params(params::NamedTuple)::AbstractSimParams
    raw_model = get(params, :model, SimParams().model)
    model_key = Symbol(lowercase(String(raw_model)))
    return make_params(Val(model_key), params)
end

function make_params(::Val{:sin_waves}, params::NamedTuple)::SimParams
    return SimParams(; params...)
end

function make_params(::Val{model}, params::NamedTuple)::AbstractSimParams where {model}
    error("Unsupported simulation model: $(model)")
end

"""
    simulate(p::AbstractSimParams)

Entry point for selecting the underlying simulation kernel.
"""
simulate(params::NamedTuple)::Dict{String,Any} = simulate(make_params(params))

function simulate(p::AbstractSimParams)::Dict{String,Any}
    error("No simulate method defined for params of type $(typeof(p))")
end

simulate(p::SimParams)::Dict{String,Any} = sin_waves(p)

end # module
