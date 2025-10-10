module SimCore

export AbstractSimParams, SinParams, make_params, simulate

"""Marker supertype for all simulation parameter carriers."""
abstract type AbstractSimParams end

"""Parameter carrier for the sine-wave toy model."""
Base.@kwdef struct SinParams <: AbstractSimParams
    T::Float64 = 1000.0
    dt::Float64 = 0.1
    amplitude::Float64 = 2.0
    frequency::Float64 = 10.0
    phase_shift::Float64 = 0.0
    vertical_shift::Float64 = -65.0
    seed::Int = 0
    model::String = "sin_waves"
    save_downsampled::Bool = false
    note::String = ""
end

"""Dummy sine wave generator standing in for a membrane potential time series."""
function sin_waves(p::SinParams)::Dict{String,Any}
    n = Int(floor(p.T / p.dt))
    t = collect(0.0:p.dt:(n-1)*p.dt)
    V = @. p.vertical_shift + p.amplitude * sin(2π * p.frequency * (t/1000.0) + p.phase_shift)
    mean_V = sum(V) / length(V)
    var_V  = sum((V .- mean_V).^2) / (length(V)-1)
    return Dict("t"=>t, "V"=>V, "mean_V"=>mean_V, "var_V"=>var_V)
end

"""
    make_params(params::NamedTuple) -> AbstractSimParams

Select and build the appropriate parameter struct for a simulation model.
"""
function make_params(params::NamedTuple)::AbstractSimParams
    raw_model = get(params, :model, SinParams().model)
    model_key = Symbol(lowercase(String(raw_model)))
    return make_params(Val(model_key), params)
end

function make_params(::Val{:sin_waves}, params::NamedTuple)::SinParams
    return SinParams(; params...)
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

simulate(p::SinParams)::Dict{String,Any} = sin_waves(p)

end # module
