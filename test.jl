"""
Dynamic Entry/Exit Game - Optimized Julia Implementation
Performance improvements:
- Vectorized operations with @simd/@inbounds
- Thread-based parallelization instead of distributed
- Pre-allocated arrays and views
- Type-stable operations
- Reduced allocations in hot paths
"""

using Random, Distributions, LinearAlgebra, Optim, Statistics, Printf
using Base.Threads

# Auto-restart with threads if needed
function check_and_restart_with_threads()
    if nthreads() == 1 && get(ENV, "JULIA_AUTO_RESTARTED", "0") == "0"
        # Detect number of physical cores
        n_cores = try
            if Sys.isapple()
                parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
            elseif Sys.islinux()
                parse(Int, strip(read(`nproc`, String)))
            else
                Sys.CPU_THREADS ÷ 2
            end
        catch
            Sys.CPU_THREADS ÷ 2
        end
        
        ENV["JULIA_AUTO_RESTARTED"] = "1"
        cmd = `$(Base.julia_cmd()) --threads $n_cores $PROGRAM_FILE $(ARGS)`
        run(cmd)
        exit(0)
    end
end

check_and_restart_with_threads()

mutable struct DynamicGame
    β::Float64; μ_mean::Float64; μ_std::Float64; γ_mean::Float64; γ_std::Float64
    entry_tax::Float64; N_max::Int; x_vals::Vector{Float64}; x_trans::Matrix{Float64}
    π::Matrix{Float64}; μ_cutoff::Matrix{Float64}; γ_cutoff::Matrix{Float64}; V_bar::Matrix{Float64}
    
    function DynamicGame(; β=0.9, μ_mean=5.0, μ_var=5.0, γ_mean=5.0, γ_var=5.0, entry_tax=0.0)
        x_vals = [-5.0, 0.0, 5.0]
        x_trans = [0.6 0.2 0.2; 0.2 0.6 0.2; 0.2 0.2 0.6]
        π = vcat(zeros(1, 3), ((10.0 .+ x_vals') ./ (reshape(1:5, :, 1) .+ 1)) .^ 2 .- 5)
        new(β, μ_mean, sqrt(μ_var), γ_mean, sqrt(γ_var), entry_tax, 5, x_vals, x_trans, π, copy(π), copy(π), copy(π))
    end
end

function solve_equilibrium!(g::DynamicGame; tol=1e-8, max_iter=5000, damp=0.5)
    g.V_bar, g.μ_cutoff, g.γ_cutoff = copy(g.π), copy(g.π), copy(g.π)
    
    for _ in 1:max_iter
        V_old, μ_old, γ_old = copy(g.V_bar), copy(g.μ_cutoff), copy(g.γ_cutoff)
        
        p_stay = clamp.(cdf.(Normal(), (g.μ_cutoff .- g.μ_mean) ./ g.μ_std), 0, 1)
        p_enter = clamp.(cdf.(Normal(), (g.γ_cutoff .- g.γ_mean) ./ g.γ_std), 0, 1)
        p_stay[1, :] .= 0; p_enter[6, :] .= 0
        
        Pr_d1, Pr_e1 = zeros(6, 6, 3), zeros(6, 6, 3)
        for N in 0:5, j in 1:3
            if N > 0
                pe = p_enter[N+1, j]
                for k in 0:(N-1)
                    w, Na = pdf(Binomial(N-1, p_stay[N+1, j]), k), 1 + k
                    Pr_d1[N+1, Na+1, j] += w * (1 - pe)
                    Na < 5 && (Pr_d1[N+1, Na+2, j] += w * pe)
                end
            end
            N < 5 && for k in 0:N
                Pr_e1[N+1, min(1+k, 5)+1, j] += pdf(Binomial(N, N > 0 ? p_stay[N+1, j] : 0), k)
            end
        end
        
        EV = g.x_trans' * g.V_bar'
        Ψ1 = g.β * dropdims(sum(Pr_d1 .* reshape(EV', 1, 6, 3), dims=2), dims=2)
        Ψ2 = g.β * dropdims(sum(Pr_e1 .* reshape(EV', 1, 6, 3), dims=2), dims=2)
        
        μ_new, γ_new = g.π .+ Ψ1, Ψ2 .- g.entry_tax
        z = (μ_new .- g.μ_mean) ./ g.μ_std
        V_new = zeros(6, 3)
        @inbounds for i in 2:6, j in 1:3
            V_new[i,j] = (1-cdf(Normal(),z[i,j]))*g.μ_mean + g.μ_std*pdf(Normal(),z[i,j]) + cdf(Normal(),z[i,j])*μ_new[i,j]
        end
        
        g.V_bar = damp*V_new + (1-damp)*V_old
        g.μ_cutoff = damp*μ_new + (1-damp)*μ_old
        g.γ_cutoff = damp*γ_new + (1-damp)*γ_old
        
        max(maximum(abs.(g.V_bar-V_old)), maximum(abs.(g.μ_cutoff-μ_old)), maximum(abs.(g.γ_cutoff-γ_old))) < tol && return true
    end
    false
end

function simulate(g::DynamicGame; T=10000, seed=42)
    rng = MersenneTwister(seed)
    d_cut = (g.μ_cutoff .- g.μ_mean) ./ g.μ_std
    e_cut = (g.γ_cutoff .- g.γ_mean) ./ g.γ_std
    N, xj, N_hist = 0, 2, zeros(Int, T)
    
    @inbounds for t in 1:T
        N_hist[t] = N
        n_stay = N > 0 ? sum(randn(rng, N) .<= d_cut[N+1, xj]) : 0
        e = N < 5 ? Int(randn(rng) <= e_cut[N+1, xj]) : 0
        N = n_stay + e
        xj = searchsortedfirst(cumsum(g.x_trans[xj, :]), rand(rng))
    end
    N_hist
end

function estimate_ccps(eq::DynamicGame; T=10000, seed=42)
    rng = MersenneTwister(seed)
    d_cut = (eq.μ_cutoff .- eq.μ_mean) ./ eq.μ_std
    e_cut = (eq.γ_cutoff .- eq.γ_mean) ./ eq.γ_std
    stay_cnt, tot_cnt, ent_cnt, ent_opp = Dict{Tuple{Int,Float64},Int}(), Dict{Tuple{Int,Float64},Int}(), Dict{Tuple{Int,Float64},Int}(), Dict{Tuple{Int,Float64},Int}()
    N, xj = 0, 2
    
    for _ in 1:T
        xv = eq.x_vals[xj]
        ns = N > 0 ? sum(randn(rng, N) .<= d_cut[N+1, xj]) : 0
        stay_cnt[(N, xv)] = get(stay_cnt, (N, xv), 0) + ns
        tot_cnt[(N, xv)] = get(tot_cnt, (N, xv), 0) + N
        
        if N < 5
            e = Int(randn(rng) <= e_cut[N+1, xj])
            ent_cnt[(N, xv)] = get(ent_cnt, (N, xv), 0) + e
            ent_opp[(N, xv)] = get(ent_opp, (N, xv), 0) + 1
            N = ns + e
        else
            N = ns
        end
        xj = searchsortedfirst(cumsum(eq.x_trans[xj, :]), rand(rng))
    end
    
    d_hat = Dict((N,x) => clamp(get(stay_cnt,(N,x),0)/get(tot_cnt,(N,x),1), 1e-3, 1-1e-3) for N in 1:5 for x in eq.x_vals if haskey(tot_cnt,(N,x)))
    e_hat = Dict((N,x) => clamp(get(ent_cnt,(N,x),0)/get(ent_opp,(N,x),1), 1e-3, 1-1e-3) for N in 0:4 for x in eq.x_vals if haskey(ent_opp,(N,x)))
    for N in 1:5, x in eq.x_vals; haskey(d_hat,(N,x)) || (d_hat[(N,x)] = 0.5); end
    for N in 0:4, x in eq.x_vals; haskey(e_hat,(N,x)) || (e_hat[(N,x)] = 0.5); end
    d_hat, e_hat
end

function _antithetic(rng, dims, n_sim, norm=true)
    half = max(1, n_sim ÷ 2)
    z = norm ? randn(rng, dims[1], half, dims[3:end]...) : rand(rng, dims[1], half, dims[3:end]...)
    out = cat(z, norm ? -z : 1.0 .- z, dims=2)
    size(out,2) < n_sim && (out = cat(out, (norm ? randn : rand)(rng, dims[1], 1, dims[3:end]...), dims=2))
    out
end

# Type-stable array-based CCP representation
struct CCPArrays
    d_arr::Matrix{Float64}
    e_arr::Matrix{Float64}
    d_thr::Matrix{Float64}
    e_thr::Matrix{Float64}
end

function ccps_to_arrays(d_hat, e_hat, eq::DynamicGame)
    d_arr, e_arr = fill(0.5, 6, 3), fill(0.5, 6, 3)
    for N in 1:5, (j,xv) in enumerate(eq.x_vals)
        haskey(d_hat,(N,xv)) && (d_arr[N+1,j] = d_hat[(N,xv)])
    end
    for N in 0:4, (j,xv) in enumerate(eq.x_vals)
        haskey(e_hat,(N,xv)) && (e_arr[N+1,j] = e_hat[(N,xv)])
    end
    d_arr, e_arr = clamp.(d_arr, 1e-6, 1-1e-6), clamp.(e_arr, 1e-6, 1-1e-6)
    d_thr, e_thr = quantile.(Normal(), d_arr), quantile.(Normal(), e_arr)
    CCPArrays(d_arr, e_arr, d_thr, e_thr)
end

function bbl_estimate(d_hat, e_hat, eq::DynamicGame; n_sim=50, horizon=100, seed=42, n_starts=3, start_from=nothing)
    states_inc = [(N,j) for N in 1:5 for j in 1:3]
    states_ent = [(N,j) for N in 0:4 for j in 1:3]
    
    ccps = ccps_to_arrays(d_hat, e_hat, eq)
    x_cum = cumsum(eq.x_trans, dims=2)
    
    function make_simulator(hor, ns, sd)
        rng = MersenneTwister(sd)
        β_pows = eq.β .^ (0:hor)
        
        # Pre-allocate all random draws
        μ_other = _antithetic(rng, (15, ns, hor, 5), ns)
        μ_firm1 = _antithetic(rng, (15, ns, hor), ns)
        γ_entry = _antithetic(rng, (15, ns, hor), ns)
        demand_u_inc = _antithetic(rng, (15, ns, hor), ns, false)
        μ_inc_initial = _antithetic(rng, (15, ns, 5), ns)
        μ_firmE = _antithetic(rng, (15, ns, hor), ns)
        μ_otherE = _antithetic(rng, (15, ns, hor, 5), ns)
        γ_entryE = _antithetic(rng, (15, ns, hor), ns)
        demand_u_ent = _antithetic(rng, (15, ns, hor), ns, false)
        ns_eff = size(μ_other, 2)
        
        # Vectorized next state function
        @inline function next_x_vec!(x_vec, u_vec, x_out)
            @inbounds @simd for i in eachindex(x_vec)
                x_out[i] = searchsortedfirst(@view(x_cum[x_vec[i],:]), u_vec[i])
            end
        end
        
        # Optimized incumbent simulation
        function sim_inc_state(si::Int, N0::Int, xi0::Int, θ)
            μ_m, σ_μ = θ[1], θ[2]
            
            # Pre-allocate working arrays
            N = fill(N0, ns_eff)
            x = fill(xi0, ns_eff)
            alive = trues(ns_eff)
            pdv = fill(eq.π[N0+1, xi0], ns_eff)
            
            # Initial transition (vectorized)
            if N0 > 1
                stay_other = zeros(Int, ns_eff)
                @inbounds for i in 1:ns_eff
                    count = 0
                    @simd for k in 1:(N0-1)
                        count += μ_other[si,i,1,k] <= ccps.d_thr[N0+1,xi0]
                    end
                    stay_other[i] = count
                end
            else
                stay_other = zeros(Int, ns_eff)
            end
            
            ent0 = N0 < 5 ? zeros(Int, ns_eff) : zeros(Int, ns_eff)
            if N0 < 5
                @inbounds @simd for i in 1:ns_eff
                    ent0[i] = γ_entry[si,i,1] <= ccps.e_thr[N0+1,xi0]
                end
            end
            
            @inbounds @simd for i in 1:ns_eff
                N[i] = min(1 + stay_other[i] + ent0[i], 5)
            end
            
            x_temp = similar(x)
            next_x_vec!(x, @view(demand_u_inc[si,:,1]), x_temp)
            x .= x_temp
            
            # Main simulation loop (optimized)
            for t in 1:(hor-1)
                any(alive) || break
                
                # Check survival (vectorized where possible)
                @inbounds for i in 1:ns_eff
                    if alive[i]
                        if μ_firm1[si,i,t] > ccps.d_thr[N[i]+1,x[i]]
                            # Firm exits
                            pdv[i] += β_pows[t] * (μ_m + σ_μ * μ_firm1[si,i,t])
                            alive[i] = false
                        else
                            # Firm stays
                            pdv[i] += β_pows[t] * eq.π[N[i]+1,x[i]]
                        end
                    end
                end
                
                # Update states for survivors
                @inbounds for i in 1:ns_eff
                    if alive[i]
                        # Count staying incumbents
                        stay_o = 0
                        if N[i] > 1
                            @simd for k in 1:(N[i]-1)
                                stay_o += μ_other[si,i,t,k] <= ccps.d_thr[N[i]+1,x[i]]
                            end
                        end
                        
                        # Entry decision
                        ent = (N[i] < 5 && γ_entry[si,i,t] <= ccps.e_thr[N[i]+1,x[i]]) ? 1 : 0
                        
                        N[i] = min(stay_o + 1 + ent, 5)
                    end
                end
                
                # Update demand state
                next_x_vec!(x, @view(demand_u_inc[si,:,t]), x_temp)
                @inbounds @simd for i in 1:ns_eff
                    alive[i] && (x[i] = x_temp[i])
                end
            end
            
            return mean(pdv)
        end
        
        # Optimized entrant simulation
        function sim_ent_state(si::Int, N0::Int, xi0::Int, θ)
            μ_m, σ_μ = θ[1], θ[2]
            
            # Initial incumbent staying
            stay_inc = zeros(Int, ns_eff)
            if N0 > 0
                @inbounds for i in 1:ns_eff
                    count = 0
                    @simd for k in 1:N0
                        count += μ_inc_initial[si,i,k] <= ccps.d_thr[N0+1,xi0]
                    end
                    stay_inc[i] = count
                end
            end
            
            N = similar(stay_inc)
            @inbounds @simd for i in 1:ns_eff
                N[i] = min(stay_inc[i] + 1, 5)
            end
            
            x = fill(xi0, ns_eff)
            x_temp = similar(x)
            next_x_vec!(x, @view(demand_u_ent[si,:,1]), x_temp)
            x .= x_temp
            
            alive = trues(ns_eff)
            pdv = zeros(ns_eff)
            
            for t in 1:(hor-1)
                any(alive) || break
                
                @inbounds for i in 1:ns_eff
                    if alive[i]
                        if μ_firmE[si,i,t] > ccps.d_thr[N[i]+1,x[i]]
                            pdv[i] += β_pows[t] * (μ_m + σ_μ * μ_firmE[si,i,t])
                            alive[i] = false
                        else
                            pdv[i] += β_pows[t] * eq.π[N[i]+1,x[i]]
                        end
                    end
                end
                
                @inbounds for i in 1:ns_eff
                    if alive[i]
                        stay_o = 0
                        if N[i] > 1
                            @simd for k in 1:(N[i]-1)
                                stay_o += μ_otherE[si,i,t,k] <= ccps.d_thr[N[i]+1,x[i]]
                            end
                        end
                        ent = (N[i] < 5 && γ_entryE[si,i,t] <= ccps.e_thr[N[i]+1,x[i]]) ? 1 : 0
                        N[i] = min(stay_o + 1 + ent, 5)
                    end
                end
                
                next_x_vec!(x, @view(demand_u_ent[si,:,t]), x_temp)
                @inbounds @simd for i in 1:ns_eff
                    alive[i] && (x[i] = x_temp[i])
                end
            end
            
            return mean(pdv)
        end
        
        function simulate_Lambda(θ)
            Λ_vals = Vector{Float64}(undef, 15)
            ΛE_vals = Vector{Float64}(undef, 15)
            
            # Use threading for state simulations
            if nthreads() > 1
                @threads for si in 1:15
                    N0, xi0 = states_inc[si]
                    Λ_vals[si] = sim_inc_state(si, N0, xi0, θ)
                end
                
                @threads for si in 1:15
                    N0, xi0 = states_ent[si]
                    ΛE_vals[si] = sim_ent_state(si, N0, xi0, θ)
                end
            else
                for si in 1:15
                    N0, xi0 = states_inc[si]
                    Λ_vals[si] = sim_inc_state(si, N0, xi0, θ)
                end
                for si in 1:15
                    N0, xi0 = states_ent[si]
                    ΛE_vals[si] = sim_ent_state(si, N0, xi0, θ)
                end
            end
            
            # Convert to dicts (for compatibility)
            Λ = Dict((states_inc[si][1], eq.x_vals[states_inc[si][2]]) => Λ_vals[si] for si in 1:15)
            ΛE = Dict((states_ent[si][1], eq.x_vals[states_ent[si][2]]) => ΛE_vals[si] for si in 1:15)
            
            return Λ, ΛE, Λ_vals, ΛE_vals
        end
        
        function obj(θ)
            Λ, ΛE, Λ_vals, ΛE_vals = simulate_Lambda(θ)
            
            inc_err = 0.0
            @inbounds for si in 1:15
                N, j = states_inc[si]
                pred = cdf(Normal(), (Λ_vals[si] - θ[1])/θ[2])
                inc_err += (pred - ccps.d_arr[N+1,j])^2
            end
            
            ent_err = 0.0
            @inbounds for si in 1:15
                N, j = states_ent[si]
                pred = cdf(Normal(), (ΛE_vals[si] - θ[3])/θ[4])
                ent_err += (pred - ccps.e_arr[N+1,j])^2
            end
            
            return inc_err + ent_err
        end
        
        simulate_Lambda, obj
    end
    
    sim_full, obj_full = make_simulator(horizon, n_sim, seed)
    rng = MersenneTwister(parse(Int, get(ENV, "BBL_SEED", "2007")))
    starts = start_from !== nothing ? [start_from] : [[rand(rng)*8, rand(rng)*2.3+1.2, rand(rng)*8, rand(rng)*2.3+1.2] for _ in 1:n_starts]
    
    all_results = [(i, x0, optimize(obj_full, x0, BFGS())) for (i,x0) in enumerate(starts)]
    best_res = argmin(r -> Optim.minimum(r[3]), all_results)[3]
    
    # Compute objective breakdown
    Λ, ΛE, Λ_vals, ΛE_vals = sim_full(Optim.minimizer(best_res))
    θ_best = Optim.minimizer(best_res)
    
    inc_sq = [(cdf(Normal(), (Λ_vals[si] - θ_best[1])/θ_best[2]) - ccps.d_arr[states_inc[si][1]+1,states_inc[si][2]])^2 for si in 1:15]
    ent_sq = [(cdf(Normal(), (ΛE_vals[si] - θ_best[3])/θ_best[4]) - ccps.e_arr[states_ent[si][1]+1,states_ent[si][2]])^2 for si in 1:15]
    
    Dict(:x => θ_best, :fun => Optim.minimum(best_res), :nit => Optim.iterations(best_res), :all_starts => all_results,
         :obj_breakdown => Dict(:inc_sum => sum(inc_sq), :ent_sum => sum(ent_sq), :total => sum(inc_sq)+sum(ent_sq),
                               :inc_mse => mean(inc_sq), :ent_mse => mean(ent_sq), :n_sim => n_sim, :horizon => horizon))
end

function bootstrap_bbl(eq::DynamicGame, θ_hat; n_boot=100, n_sim=1000, horizon=1000, T_ccp=10000, seed=42)
    println("\n  Running $n_boot bootstrap replications with $(nthreads()) threads...")
    
    boot_est = zeros(n_boot, 4)
    
    # Use threading for bootstrap (much faster than distributed)
    @threads for b in 1:n_boot
        try
            d_boot, e_boot = estimate_ccps(eq, T=T_ccp, seed=seed+10000*b)
            boot_est[b,:] = bbl_estimate(d_boot, e_boot, eq, n_sim=n_sim, horizon=horizon, 
                                        seed=seed+10000*b, n_starts=1, start_from=θ_hat)[:x]
        catch
            boot_est[b,:] .= NaN
        end
        
        # Progress reporting (thread-safe)
        if (b % 10 == 0)
            println("    Completed $b/$n_boot replications")
        end
    end
    
    valid = .!any(isnan.(boot_est), dims=2)[:]
    n_valid = sum(valid)
    println("    Completed $n_valid/$n_boot valid bootstrap replications")
    n_valid < 5 && return nothing
    
    boot_v = boot_est[valid,:]
    (se=vec(std(boot_v, dims=1, corrected=true)), ci_lo=[quantile(boot_v[:,i], 0.025) for i in 1:4], 
     ci_hi=[quantile(boot_v[:,i], 0.975) for i in 1:4], n_valid=n_valid)
end

function main()
    println("Dynamic Entry/Exit Game: Equilibrium and Estimation (OPTIMIZED)")
    println("Threads: $(nthreads())")
    println("="^70)
    
    println("\nQ4-5: Equilibrium Computation")
    inits = [("π", nothing), ("0", zeros(6,3)), ("1", ones(6,3)), ("U[0,10]", rand(MersenneTwister(123),6,3).*10), ("U[-5,15]", rand(MersenneTwister(456),6,3).*20 .- 5)]
    sols = map(inits) do (name, init_val)
        game = DynamicGame()
        init_val !== nothing && (game.V_bar = game.μ_cutoff = game.γ_cutoff = copy(init_val))
        solve_equilibrium!(game)
        @printf("  Init %-8s: V̄(3,0)=%.4f, μ̄(3,0)=%.4f\n", name, game.V_bar[4,2], game.μ_cutoff[4,2])
        game
    end
    println("  Result: ", all(isapprox(sols[1].V_bar, s.V_bar, atol=1e-6) for s in sols[2:end]) ? "UNIQUE" : "MULTIPLE")
    
    eq = sols[1]
    println("\nQ6: Equilibrium at (N,x)=(3,0)")
    @printf("  μ̄(3,0) = %.6f\n  γ̄(3,0) = %.6f\n  V̄(3,0) = %.6f\n", eq.μ_cutoff[4,2], eq.γ_cutoff[4,2], eq.V_bar[4,2])
    @printf("  V(3,0,-2) = %.6f (firm %s)\n", max(-2.0, eq.μ_cutoff[4,2]), -2 <= eq.μ_cutoff[4,2] ? "stays" : "exits")
    
    println("\nQ7: Market Simulation (10,000 periods)")
    N_hist = simulate(eq)
    @printf("  Average firms: %.4f\n", mean(N_hist))
    
    println("\nQ8: Entry Tax Counterfactual")
    tax = DynamicGame(entry_tax=5.0); solve_equilibrium!(tax); N_tax = simulate(tax)
    change = mean(N_tax) - mean(N_hist)
    @printf("  With 5-unit tax: %.4f firms\n  Change: %.4f firms (%.2f%%)\n", mean(N_tax), change, 100*change/mean(N_hist))
    
    println("\nQ9: BBL-like Estimation (forward simulation of Λ, Λ^E)")
    
    # Estimate CCPs
    println("  Step 1: Nonparametric CCP estimation from simulated data")
    T_ccp = parse(Int, get(ENV, "BBL_T_CCP", "10000"))
    println("    Simulating $T_ccp periods from equilibrium...")
    d_hat, e_hat = estimate_ccps(eq, T=T_ccp, seed=42)
    @printf("    Estimated %d incumbent CCPs and %d entry CCPs from data\n", length(d_hat), length(e_hat))
    
    println("  Step 2: Forward simulation (Λ, Λ^E) and minimum distance search")
    
    for (lbl, ns) in [("50", 50), ("1000", 1000)]
        println("  Using $lbl simulations, horizon=1000")
        res = bbl_estimate(d_hat, e_hat, eq, n_sim=ns, horizon=1000, seed=42, n_starts=3)
        for (i, x0, r) in res[:all_starts]
            @printf("    Start rng-%-3d: obj=%.6f, μ=%6.3f, σ_μ=%5.3f, γ=%6.3f, σ_γ=%5.3f, iters=%2d\n",
                    i, Optim.minimum(r), Optim.minimizer(r)[1], Optim.minimizer(r)[2], Optim.minimizer(r)[3], Optim.minimizer(r)[4], Optim.iterations(r))
        end
        ob = res[:obj_breakdown]
        @printf("    Settings: n_sim=%d, horizon=%d\n    Objective breakdown: total=%.6f, inc_sum=%.6f, ent_sum=%.6f\n", ob[:n_sim], ob[:horizon], ob[:total], ob[:inc_sum], ob[:ent_sum])
        ns == 1000 && (global res_1000 = res)
    end
    
    n_boot = parse(Int, get(ENV, "BBL_N_BOOTSTRAP", "1000"))
    if parse(Int, get(ENV, "BBL_BOOTSTRAP", "1")) > 0 && n_boot > 0
        boot = bootstrap_bbl(eq, res_1000[:x], n_boot=n_boot, n_sim=1000, horizon=1000, T_ccp=10000, seed=42)
        boot !== nothing && @printf("  Bootstrap SE:         μ=%6.3f, σ_μ=%5.3f, γ=%6.3f, σ_γ=%5.3f\n  Percentile 95%% CI:    μ=[%6.3f, %6.3f], σ_μ=[%5.3f, %5.3f], γ=[%6.3f, %6.3f], σ_γ=[%5.3f, %5.3f]\n",
            boot.se[1], boot.se[2], boot.se[3], boot.se[4], boot.ci_lo[1], boot.ci_hi[1], boot.ci_lo[2], boot.ci_hi[2], boot.ci_lo[3], boot.ci_hi[3], boot.ci_lo[4], boot.ci_hi[4])
    end
    
    println("\n" * "="^70)
end

# Run main if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end