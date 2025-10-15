module SimCore

using Random
using Statistics
using Printf
using Plots
using Distributions
using Base: @views

include("Analysis.jl"); using .Analysis

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
    buffer_time::Float64 = 1000.0  # ms of data to keep in ring buffer

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
    tau_e_decay::Float64 = 5.0
    tau_i_decay::Float64 = 4.0
    tau_e_rise::Float64 = 1.0
    tau_i_rise::Float64 = 1.0
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
    # --- unpack ---
    t_0, T, dt = p.t_0, p.T, p.dt
    burn_in_steps = Int(round(p.burn_in_time / dt))
    N = Int(floor((T - t_0)/dt))

    g_L, E_L, C = p.g_L, p.E_L, p.C
    Vre, Vth = p.Vre, p.Vth
    tau_ref = p.tau_ref; ref_steps = Int(round(tau_ref / dt)); refr_count = 0
    tau_e_decay, tau_i_decay = p.tau_e_decay, p.tau_i_decay
    tau_e_rise, tau_i_rise = p.tau_e_rise, p.tau_i_rise
    E_e, E_i = p.E_e, p.E_i
    a, g = p.a, p.g
    j_e, j_i = a, a * g
    K, gamma = p.K, p.gamma
    K_e = max(K, 1)
    K_i = max(Int(gamma * K), 1)
    r_e = p.nu_x / 1000       # per ms
    r_i = p.eta * r_e         # per ms
    fano_window = p.fano_window
    buffer_time = p.buffer_time

    # Poisson dists
    dE = Poisson(K_e * r_e * dt)
    dI = Poisson(K_i * r_i * dt)

    # precompute constants
    invC = 1 / C

    # --- ring buffers for LAST buffer_time ms ---
    Wlast = Int(buffer_time / dt)
    t_last = collect(0.0:dt:((Wlast-1)*dt))  # small
    Vbuf  = fill(Float32(E_L), Wlast)
    gebuf = fill(Float32(0), Wlast)
    gibuf = fill(Float32(0), Wlast)
    ring_idx = 1

    # --- state scalars (no full arrays) ---
    V  = E_L
    ge = 0.0
    gi = 0.0
    xe_rise = 0.0
    xe_decay = 0.0
    xi_rise = 0.0
    xi_decay = 0.0

    # --- running stats via Welford (after burn-in) ---
    n_samp = 0
    mean_V = 0.0; M2_V = 0.0
    mean_ge = 0.0; M2_ge = 0.0
    mean_gi = 0.0; M2_gi = 0.0
    mean_g0 = 0.0; M2_g0 = 0.0
    mean_Ie = 0.0; M2_Ie = 0.0
    mean_Ii = 0.0; M2_Ii = 0.0
    mean_It = 0.0; M2_It = 0.0

    # --- firing rate (from spike density) ---
    sum_S = 0.0

    # --- Fano factor via windowed counts over 100 ms ---
    w = Int(fano_window / dt)
    step_in_win = 0
    acc_counts = 0.0
    counts = Float32[]  # length about (N - burn_in) / w

    @inbounds for n in 1:N
        # presynaptic densities
        s_e = rand(dE) / dt
        s_i = rand(dI) / dt

        # conductances
        xe_rise += - dt * xe_rise / tau_e_rise + dt * s_e * j_e
        xe_decay += - dt * xe_decay / tau_e_decay + dt * s_e * j_e
        xi_rise += - dt * xi_rise / tau_i_rise + dt * s_i * j_i
        xi_decay += - dt * xi_decay / tau_i_decay + dt * s_i * j_i
        ge = (xe_decay - xe_rise) / (tau_e_decay - tau_e_rise)
        gi = (xi_decay - xi_rise) / (tau_i_decay - tau_i_rise)

        # refractory & spike reset
        S = 0.0
        if refr_count > 0
            V = Vre
            refr_count -= 1
        else
            V = V + dt * invC * ( -g_L*(V - E_L) - ge*(V - E_e) - gi*(V - E_i) )
            if V >= Vth
                V = Vre
                S = 1.0 / dt
                refr_count = ref_steps
            end
        end

        # after burn-in, update stats + windows
        if n > burn_in_steps
            # ring buffers for last buffer_time ms
            Vbuf[ring_idx]  = Float32(V)
            gebuf[ring_idx] = Float32(ge)
            gibuf[ring_idx] = Float32(gi)
            ring_idx += 1
            ring_idx > Wlast && (ring_idx = 1)

            # instantaneous currents
            Ie = -ge * (V - E_e)
            Ii = -gi * (V - E_i)
            It = Ie + Ii

            # Welford updates
            n_samp += 1
            let x = V
                δ = x - mean_V; mean_V += δ / n_samp; M2_V += δ*(x - mean_V)
            end
            let x = ge
                δ = x - mean_ge; mean_ge += δ / n_samp; M2_ge += δ*(x - mean_ge)
            end
            let x = gi
                δ = x - mean_gi; mean_gi += δ / n_samp; M2_gi += δ*(x - mean_gi)
            end
            # compute g0 = g_L + ge + gi
            let x = ge + gi + g_L
                δ = x - mean_g0; mean_g0 += δ / n_samp; M2_g0 += δ*(x - mean_g0)
            end
            let x = Ie
                δ = x - mean_Ie; mean_Ie += δ / n_samp; M2_Ie += δ*(x - mean_Ie)
            end
            let x = Ii
                δ = x - mean_Ii; mean_Ii += δ / n_samp; M2_Ii += δ*(x - mean_Ii)
            end
            let x = It
                δ = x - mean_It; mean_It += δ / n_samp; M2_It += δ*(x - mean_It)
            end

            # rate and Fano (counts per 100 ms)
            sum_S += S
            acc_counts += S * dt     # S is 1/dt at spikes ⇒ S*dt is count increment
            step_in_win += 1
            if step_in_win == w
                push!(counts, Float32(acc_counts))
                acc_counts = 0.0
                step_in_win = 0
            end
        end
    end

    # finalize stats
    var_V  = n_samp > 1 ? M2_V  / (n_samp - 1) : 0.0
    var_ge = n_samp > 1 ? M2_ge / (n_samp - 1) : 0.0
    var_gi = n_samp > 1 ? M2_gi / (n_samp - 1) : 0.0
    var_g0 = n_samp > 1 ? M2_g0 / (n_samp - 1) : 0.0
    var_Ie = n_samp > 1 ? M2_Ie / (n_samp - 1) : 0.0
    var_Ii = n_samp > 1 ? M2_Ii / (n_samp - 1) : 0.0
    var_It = n_samp > 1 ? M2_It / (n_samp - 1) : 0.0

    # Fano factor of window counts
    fano_factor = isempty(counts) ? NaN : (var(counts) / mean(counts))

    # firing rate (Hz)
    nu = (sum_S / n_samp) * 1000.0

    # unwrap ring buffers in chronological order
    function unwrap(buf)
        ring_idx == 1 ? copy(buf) :
            vcat(view(buf, ring_idx:Wlast), view(buf, 1:ring_idx-1))
    end

    V_buf = unwrap(Vbuf)
    gebuf = unwrap(gebuf)
    gibuf = unwrap(gibuf)
    
    # precompute Ie, Ii, It buffers here
    Ie_buf = .- gebuf .* (V_buf .- E_e)
    Ii_buf = .- gibuf .* (V_buf .- E_i)
    It_buf = Ie_buf .+ Ii_buf

    return Dict(
        "t" => t_last,
        "V" => V_buf,
        "g_e" => gebuf,
        "g_i" => gibuf,
        "I_e" => Ie_buf,
        "I_i" => Ii_buf,
        "I_tot" => It_buf,
        "fano_factor" => fano_factor,
        "nu" => nu,
        "mean_V" => mean_V, "var_V" => var_V,
        "mean_g_e" => mean_ge, "var_g_e" => var_ge,
        "mean_g_i" => mean_gi, "var_g_i" => var_gi,
        "mean_g_0" => mean_g0, "var_g_0" => var_g0,
        "mean_I_e" => mean_Ie, "var_I_e" => var_Ie,
        "mean_I_i" => mean_Ii, "var_I_i" => var_Ii,
        "mean_I_tot" => mean_It, "var_I_tot" => var_It,
    )
end

function single_conductance_lif_old(p::SingleConductanceLIF)

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

    tau_e = p.tau_e_decay
    tau_i = p.tau_i_decay
    E_i = p.E_i
    E_e = p.E_e

    g = p.g
    a = p.a
    j_e = a
    j_i = a * g

    K = p.K
    gamma = p.gamma
    # both should be min 1
    K_i = max(Int(gamma * K), 1)
    K_e = max(K, 1)

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
    S   = zeros(N)  # post-synaptic spike density (in kHz)

    # -------------------------
    # Generate presynaptic spike densities S_e, S_i
    # -------------------------
    λ_e = K_e * r_e
    λ_i = K_i * r_i
    Poisson_e = Poisson(λ_e * dt)
    Poisson_i = Poisson(λ_i * dt)

    # -------------------------
    # Time stepping
    # -------------------------
    decay_e = dt / tau_e
    decay_i = dt / tau_i
    drive_e = dt * g_L * j_e
    drive_i = dt * g_L * j_i
    invC = 1 / C

    @inbounds for n in 1:(N-1)
        # Determine the presynaptic spike density at time step n 
        s_e = rand(Poisson_e) / dt
        s_i = rand(Poisson_i) / dt

        # conductance updates
        ge = g_e[n]
        gi = g_i[n]
        g_e[n+1] = ge - decay_e * ge + drive_e * s_e
        g_i[n+1] = gi - decay_i * gi + drive_i * s_i

        # refractory handling
        if refr_count > 0
            V[n+1] = Vre
            S[n+1] = 0.0
            refr_count -= 1
            continue
        end

        # membrane update
        Vn = V[n]
        V[n+1] = Vn + dt * invC * (
            - g_L * (Vn - E_L) - g_e[n] * (Vn - E_e) - g_i[n] * (Vn - E_i)
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
    I_e_ss = I_e[(burn_in_steps+1):end]
    I_i_ss = I_i[(burn_in_steps+1):end]
    I_tot_ss = I_tot[(burn_in_steps+1):end]

    # compute fano factor over 100 ms windows
    spike_counts = spike_counts_from_spike_density(S_ss, dt; window_size_ms=100)
    fano_factor = fano(spike_counts)

    # compute firing rate using S_ss spike density
    nu = mean(S_ss) * 1000.0  # in Hz

    
    # extract the last 1000 ms of data to save
    V_short = V_ss[end-Int(1000/dt)+1:end]
    g_e_short = g_e_ss[end-Int(1000/dt)+1:end]
    g_i_short = g_i_ss[end-Int(1000/dt)+1:end]
    I_e_short = I_e_ss[end-Int(1000/dt)+1:end]
    I_i_short = I_i_ss[end-Int(1000/dt)+1:end]
    I_tot_short = I_tot_ss[end-Int(1000/dt)+1:end]
    time_vec = collect(0.0:dt:999.9)

    return Dict(
        "t"=>time_vec,
        "V"=>V_short,
        "g_e"=>g_e_short,
        "g_i"=>g_i_short,
        # "I_e"=>I_e_short,
        # "I_i"=>I_i_short,
        # "I_tot"=>I_tot_short,
        "fano_factor"=>fano_factor,
        "nu"=>nu,
        "mean_V"=>mean(V_ss),
        "var_V"=>var(V_ss),
        "mean_g_e"=>mean(g_e_ss),
        "var_g_e"=>var(g_e_ss),
        "mean_g_i"=>mean(g_i_ss),
        "var_g_i"=>var(g_i_ss),
        "mean_I_e"=>mean(I_e_ss),
        "var_I_e"=>var(I_e_ss),
        "mean_I_i"=>mean(I_i_ss),
        "var_I_i"=>var(I_i_ss),
        "mean_I_tot"=>mean(I_tot_ss),
        "var_I_tot"=>var(I_tot_ss),
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


# Utilities functions

# Generate a Poisson spike density with constant rate λ over time interval [0, T] with time step dt
function poisson_spike_density(λ::Float64, T::Float64, dt::Float64)
    d = Poisson(λ * dt)
    poisson_events = rand(d, round(Int, T/dt))
    spike_density = poisson_events ./ dt
    return spike_density
end

end # module
