import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

class DynamicEntryExitGame:
    def __init__(self, gamma=5, sigma_gamma=np.sqrt(5), mu=5, sigma_mu=np.sqrt(5), beta=0.9, N_max=5):
        self.gamma = gamma
        self.sigma_gamma = sigma_gamma
        self.mu = mu
        self.sigma_mu = sigma_mu
        self.beta = beta
        self.N_max = N_max
        self.x_values = np.array([-5, 0, 5])
        self.P_x = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6]
        ])
        self.gamma_cutoff = {}
        self.mu_cutoff = {}
        self.V_bar = {}
        self.solve_equilibrium()

    def compute_pi(self, N, x):
        if N == 0:
            return 0.0
        profit = ((10 + x) / (N + 1))**2 - 5
        return profit

    def solve_equilibrium(self, max_iter=2000, tol=1e-8, verbose=False):
        # Simplified equilibrium solving - assume cutoffs are known or solve iteratively
        # For this script, use fixed cutoffs based on true parameters
        for N in range(1, self.N_max + 1):
            for x in self.x_values:
                # Approximate cutoffs
                self.mu_cutoff[(N, x)] = self.mu + 0.5 * N
                self.gamma_cutoff[(N, x)] = self.gamma - 0.5 * N
                self.V_bar[(N, x)] = self.compute_pi(N, x) / (1 - self.beta)
        return True

    def simulate_market(self, N_init=0, x_init=0, T=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        data = []
        N_t = N_init
        x_t = x_init
        
        for t in range(T):
            N_initial = N_t  # Record N before exits for CCP estimation
            
            # Exit decisions for incumbents
            n_exits = 0
            if N_t > 0:
                mu_cut = self.mu_cutoff.get((N_t, x_t), self.mu)
                mu_draws = np.random.normal(self.mu, self.sigma_mu, N_t)
                stay = mu_draws <= mu_cut
                n_exits = N_t - np.sum(stay)
                N_t = int(np.sum(stay))
            
            # Entry decision
            entry = 0
            if N_t < self.N_max:
                gamma_cut = self.gamma_cutoff.get((N_t, x_t), self.gamma)
                gamma_draw = np.random.normal(self.gamma, self.sigma_gamma)
                if gamma_draw <= gamma_cut:
                    N_t += 1
                    entry = 1
            
            # Record state
            data.append({
                'period': t, 
                'N_initial': N_initial,
                'N': N_t, 
                'x': x_t,
                'entry': entry,
                'exits': n_exits
            })
            
            # Demand shifter transition
            x_idx = np.where(self.x_values == x_t)[0][0]
            x_idx_next = np.random.choice(len(self.x_values), 
                                         p=self.P_x[x_idx, :])
            x_t = self.x_values[x_idx_next]
        
        return pd.DataFrame(data)

def estimate_policy_functions(df_sim):
    d_hat = {}
    e_hat = {}
    
    for (N, x) in [(n, x_val) for n in range(6) for x_val in [-5, 0, 5]]:
        if N > 0:
            df_state = df_sim[(df_sim['N_initial'] == N) & (df_sim['x'] == x)]
        else:
            df_state = df_sim[(df_sim['N'] == N) & (df_sim['x'] == x)]
        
        if len(df_state) > 0:
            if N > 0:
                avg_exits = df_state['exits'].mean()
                d_hat[(N, x)] = max(0, min(1, 1 - avg_exits / N))
            
            if N < 5:
                e_hat[(N, x)] = df_state['entry'].mean()
    
    return d_hat, e_hat

def forward_simulate_value(N_start, x_start, is_incumbent, theta, d_hat, e_hat, 
                           game_base, n_sims=1000, T_sim=500, seed_base=2024):
    gamma_sim, sigma_gamma_sim, mu_sim, sigma_mu_sim = theta
    
    pdvs = []
    
    for sim in range(n_sims):
        np.random.seed(seed_base + sim)
        
        N_t = N_start
        x_t = x_start
        pdv = 0.0
        
        if not is_incumbent:
            pdv -= gamma_sim  # Pay entry cost
            N_t += 1
            pdv += game_base.compute_pi(N_t, x_t)
        
        start_period = 1 if not is_incumbent else 0
        
        for t in range(start_period, start_period + T_sim):
            p_stay_focal = d_hat.get((N_t, x_t), 0.5)
            stays_focal = np.random.uniform() <= p_stay_focal
            
            if not stays_focal:
                pdv += (game_base.beta ** t) * mu_sim
                break
            
            pi_t = game_base.compute_pi(N_t, x_t)
            pdv += (game_base.beta ** t) * pi_t
            
            if N_t > 1:
                p_stay = d_hat.get((N_t, x_t), 0.5)
                stays_others = np.random.uniform(N_t - 1) <= p_stay
                N_t = 1 + int(np.sum(stays_others))
            
            if N_t < 5:
                p_enter = e_hat.get((N_t, x_t), 0.5)
                enters = np.random.uniform() <= p_enter
                if enters:
                    N_t += 1
            
            x_idx = np.where(game_base.x_values == x_t)[0][0]
            x_idx_next = np.random.choice(len(game_base.x_values), 
                                         p=game_base.P_x[x_idx, :])
            x_t = game_base.x_values[x_idx_next]
        
        pdvs.append(pdv)
    
    return np.mean(pdvs)

def bbl_objective(theta_vec, d_hat, e_hat, game_base, states_subset):
    gamma_est, log_sigma_gamma, mu_est, log_sigma_mu = theta_vec
    sigma_gamma_est = np.exp(log_sigma_gamma)
    sigma_mu_est = np.exp(log_sigma_mu)
    theta = (gamma_est, sigma_gamma_est, mu_est, sigma_mu_est)
    
    moments = []
    for state in states_subset:
        (N, x) = state
        moment = 0.0
        
        if N > 0 and (N, x) in d_hat:
            Lambda_inc = forward_simulate_value(
                N, x, True, theta, d_hat, e_hat, game_base, 
                n_sims=2000, T_sim=1000)
            
            pred_stay = norm.cdf((Lambda_inc - mu_est) / sigma_mu_est)
            obs_stay = d_hat[(N, x)]
            moment += (pred_stay - obs_stay) ** 2
        
        if N < 5 and (N, x) in e_hat:
            Lambda_ent = forward_simulate_value(
                N, x, False, theta, d_hat, e_hat, game_base, 
                n_sims=2000, T_sim=1000)
            
            pred_enter = norm.cdf((gamma_est - Lambda_ent) / sigma_gamma_est)
            obs_enter = e_hat[(N, x)]
            moment += (pred_enter - obs_enter) ** 2
            
        moments.append(moment)
    
    obj = sum(moments)
    
    n_moments = sum(1 for state in states_subset if 
                   (state[0] > 0 and state in d_hat) or 
                   (state[0] < 5 and state in e_hat))
    if n_moments > 0:
        obj = obj / n_moments
    
    return obj

def run_bbl_estimation(game, df_sim, states_full, 
                       initial_guess=None, bounds=None,                        method='SLSQP', 
                       options=None, n_random_starts=10):
    if initial_guess is None:
        # Use multiple random starts for robustness
        initial_guesses = []
        for _ in range(n_random_starts):
            gamma_init = np.random.uniform(3, 7)
            sigma_gamma_init = np.random.uniform(1, 3)
            mu_init = np.random.uniform(3, 7)
            sigma_mu_init = np.random.uniform(1, 3)
            initial_guesses.append([gamma_init, np.log(sigma_gamma_init), mu_init, np.log(sigma_mu_init)])
    else:
        initial_guesses = [initial_guess]
    
    if bounds is None:
        bounds = [(1, 10), (-2, 2), (1, 10), (-2, 2)]
    
    if options is None:
        options = {'maxiter': 500, 'ftol': 1e-9}
    
    print("Estimating policy functions...")
    d_hat, e_hat = estimate_policy_functions(df_sim)
    
    print(f"Running BBL estimation with {len(initial_guesses)} starting points...")
    
    best_result = None
    best_fun = float('inf')
    
    for i, init_guess in enumerate(initial_guesses):
        print(f"  Starting optimization {i+1}/{len(initial_guesses)} with initial guess: {[f'{x:.2f}' for x in init_guess]}")
        result = minimize(bbl_objective, init_guess, args=(d_hat, e_hat, game, states_full), 
                          method=method, bounds=bounds, options=options)
        
        if result.success and result.fun < best_fun:
            best_result = result
            best_fun = result.fun
            print(f"    New best objective: {best_fun:.6f}")
    
    if best_result is not None:
        gamma_est, log_sigma_gamma_est, mu_est, log_sigma_mu_est = best_result.x
        sigma_gamma_est = np.exp(log_sigma_gamma_est)
        sigma_mu_est = np.exp(log_sigma_mu_est)
        
        print("BBL Estimation Results (best from multiple starts):")
        print(f"  γ (true: {game.gamma:.2f}) → {gamma_est:.4f}")
        print(f"  σ_γ (true: {game.sigma_gamma:.2f}) → {sigma_gamma_est:.4f}")
        print(f"  μ (true: {game.mu:.2f}) → {mu_est:.4f}")
        print(f"  σ_μ (true: {game.sigma_mu:.2f}) → {sigma_mu_est:.4f}")
        print(f"  Objective value: {best_result.fun:.6f}")
        print(f"  Iterations: {best_result.nit}")
    else:
        print("All optimizations failed")
        best_result = result  # Return the last one even if failed
    
    return best_result

if __name__ == "__main__":
    # Create game and simulate data
    game = DynamicEntryExitGame()
    df_sim = game.simulate_market(T=10000, seed=1995)
    
    # States for estimation
    states_full = [(n, x_val) for n in range(6) for x_val in [-5, 0, 5]]
    
    # Test with true initial guess first
    print("Testing with true initial guess:")
    true_guess = [game.gamma, np.log(game.sigma_gamma), game.mu, np.log(game.sigma_mu)]
    result_true = run_bbl_estimation(game, df_sim, states_full, initial_guess=true_guess, n_random_starts=1)
    
    print("\nTesting with multiple random starts:")
    # Run estimation with random starts
    result = run_bbl_estimation(game, df_sim, states_full)