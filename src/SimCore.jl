module SimCore

using Random
using Statistics
using Printf
using Plots
using Distributions

export AbstractSimParams, SinParams, SingleConductanceLIF, make_params, simulate

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

"""Placeholder parameter carrier for the single-conductance LIF model."""
Base.@kwdef struct SingleConductanceLIF <: AbstractSimParams
    # general parameters
    seed::Int = 0
    model::String = "sin_waves"
    save_downsampled::Bool = false
    note::String = ""

    # time parameters
    t_0::Float64 = 0.0
    T::Float64 = 12000.0
    dt::Float64 = 0.1
    fano_window::Float64 = 100.0
    burn_in_time::Float64 = 2000.0

    # leaky neuron parameters
    g_L::Float64 = 1.0
    E_L::Float64 = -70.0
    C::Float64 = 2.0
    tau_m::Float64 = C / g_L

    # spiking parameters
    Vre::Float64 = -60.0
    Vth::Float64 = -54.0
    tau_ref::Float64 = 2.0  # ms

    # conductance synapse parameters
    tau_e::Float64 = 5.0
    tau_i::Float64 = 4.0
    E_i::Float64 = -80.0
    E_e::Float64 = 0.0

    # synaptic efficacies
    g::Float64 = 0.3
    a::Float64 = 0.04

    # synapse counts
    K::Int = 400
    gamma::Int = 5

    # presynaptic firing rates (Hz -> per ms)
    eta::Float64 = 1.5
    nu_x::Float64 = 10.0
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

function single_conductance_lif(p::SingleConductanceLIF)

    # Allocate all parameters from struct
    t_0 = p.t_0
    T = p.T
    dt = p.dt
    fano_window = p.fano_window
    burn_in_time = p.burn_in_time
    burn_in_steps = Int(round(burn_in_time / dt))

    g_L = p.g_L
    E_L = p.E_L
    C   = p.C
    tau_m = p.tau_m

    Vre = p.Vre
    Vth = p.Vth
    tau_ref = p.tau_ref
    ref_steps = Int(round(tau_ref / dt))
    refr_count = 0

    tau_e = p.tau_e
    tau_i = p.tau_i
    E_i = p.E_i
    E_e = p.E_e

    g = p.g
    a = p.a
    j_e = a
    j_i = a * g

    K = p.K
    gamma = p.gamma
    K_i = Int(gamma * K)
    K_e = K

    eta = p.eta
    nu_x = p.nu_x
    r_i = eta * nu_x / 1000    # per ms
    r_e = nu_x / 1000          # per ms

    # -------------------------
    # Time
    # -------------------------
    time_vec = collect(t_0:dt:(T - dt))   # length N
    N = length(time_vec)

    # -------------------------
    # Initialize state vectors
    # -------------------------
    V  = fill(0.0, N)
    V[1] = E_L
    g_e = zeros(N)
    g_i = zeros(N)
    S   = zeros(N)  # spike density (δ(t) as 1/dt at spike times)

    # -------------------------
    # Generate spike densities S_e, S_i
    # -------------------------

    p_spike_e = K_e * r_e * dt # spike probability per time step
    p_spike_i = K_i * r_i * dt
    S_e = rand(N) .< p_spike_e # boolean spike mask
    S_i = rand(N) .< p_spike_i
    S_e = S_e .* (1 / dt) # scale spikes to delta-like values
    S_i = S_i .* (1 / dt)

    # -------------------------
    # Time stepping
    # -------------------------
    for n in 1:(N-1)
        # conductance updates
        g_e[n+1] = g_e[n] + dt * ( ( -g_e[n] + g_L * tau_e * j_e * S_e[n] ) / tau_e )
        g_i[n+1] = g_i[n] + dt * ( ( -g_i[n] + g_L * tau_i * j_i * S_i[n] ) / tau_i )

        # refractory handling
        if refr_count > 0
            V[n+1] = Vre
            S[n+1] = 0.0
            refr_count -= 1
            continue
        end

        # membrane update
        V[n+1] = V[n] + dt * (1 / C) * (
            - g_L * (V[n] - E_L) - g_e[n] * (V[n] - E_e) - g_i[n] * (V[n] - E_i)
        )

        # spike check
        if V[n+1] >= Vth
            V[n+1] = Vre
            S[n+1] = 1.0 / dt
            refr_count = ref_steps
        end
    end

    # -------------------------
    # Currents
    # -------------------------
    I_e  = .- g_e .* (V .- E_e)
    I_i  = .- g_i .* (V .- E_i)
    I_tot = I_e .+ I_i

    # -------------------------
    # Slice after burn-in and record stats
    # -------------------------
    V_ss   = V[(burn_in_steps+1):end]
    S_ss   = S[(burn_in_steps+1):end]
    g_e_ss = g_e[(burn_in_steps+1):end]
    g_i_ss = g_i[(burn_in_steps+1):end]

    # compute fano factor over 100 ms windows
    window_size = Int(fano_window / dt)
    spike_counts = []
    for start_idx in 1:window_size:(length(S_ss) - window_size + 1)
        window = S_ss[start_idx:(start_idx + window_size - 1)]
        spike_count = sum(window) * dt
        push!(spike_counts, spike_count)
    end
    mean_spike_count = mean(spike_counts)
    var_spike_count  = var(spike_counts)
    fano_factor = var_spike_count / mean_spike_count

    # compute firing rate using S_ss spike density
    nu = mean(S_ss) * 1000.0  # in Hz


    return Dict(
        "t"=>time_vec,
        "V"=>V,
        "g_e"=>g_e,
        "g_i"=>g_i,
        "I_e"=>I_e,
        "I_i"=>I_i,
        "I_tot"=>I_tot,
        "S"=>S,
        "S_e"=>S_e,
        "S_i"=>S_i,
        "mean_V"=>mean(V_ss),
        "var_V"=>var(V_ss),
        "mean_g_e"=>mean(g_e_ss),
        "var_g_e"=>var(g_e_ss),
        "mean_g_i"=>mean(g_i_ss),
        "var_g_i"=>var(g_i_ss),
        "fano_factor"=>fano_factor,
        "nu"=>nu,
    )

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

function make_params(::Val{:single_conductance_lif}, params::NamedTuple)::SingleConductanceLIF
    return SingleConductanceLIF(; params...)
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
simulate(p::SingleConductanceLIF) = single_conductance_lif(p)

end # module
