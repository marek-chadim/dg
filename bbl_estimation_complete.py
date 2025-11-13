"""
Complete BBL Estimation for Dynamic Entry/Exit Game
This implements the full Bajari-Benkard-Levin estimation procedure
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple
import pickle

@dataclass
class BBLEstimator:
    """Bajari-Benkard-Levin Estimator for Dynamic Games"""
    
    def __init__(self, game, simulation_data):
        self.game = game
        self.sim_data = simulation_data
        self.d_hat = None
        self.e_hat = None
        
    def step_a_estimate_policies(self):
        """
        Step A: Estimate reduced form policy functions
        
        Returns d_hat(N,x) and e_hat(N,x) - the empirical probabilities
        of staying and entering at each state
        """
        print("\n=== BBL Step A: Estimating Policy Functions ===")
        
        # Extract from simulation with decisions tracked
        N_history = self.sim_data['N_history']
        x_history = self.sim_data['x_history']
        stay_decisions = self.sim_data['stay_decisions']  # Shape: (T, max_N)
        entry_decisions = self.sim_data['entry_decisions']  # Shape: (T,)
        
        d_hat = np.zeros((6, 3))  # Stay probabilities
        e_hat = np.zeros((6, 3))  # Entry probabilities
        
        stay_counts = np.zeros((6, 3))
        stay_totals = np.zeros((6, 3))
        entry_counts = np.zeros((6, 3))
        entry_totals = np.zeros((6, 3))
        
        for t in range(len(N_history)):
            N = N_history[t]
            x = x_history[t]
            x_idx = np.where(self.game.x_states == x)[0][0]
            
            # Incumbent decisions
            if N > 0:
                stays = stay_decisions[t, :N]  # Decisions of N incumbents
                stay_counts[N, x_idx] += np.sum(stays)
                stay_totals[N, x_idx] += N
            
            # Entry decision
            if N < 5:
                entry_counts[N, x_idx] += entry_decisions[t]
                entry_totals[N, x_idx] += 1
        
        # Compute frequencies
        for N in range(6):
            for x_idx in range(3):
                if stay_totals[N, x_idx] > 0:
                    d_hat[N, x_idx] = stay_counts[N, x_idx] / stay_totals[N, x_idx]
                
                if entry_totals[N, x_idx] > 0:
                    e_hat[N, x_idx] = entry_counts[N, x_idx] / entry_totals[N, x_idx]
        
        # Handle unobserved states (use nearby states or smoothing)
        for N in range(6):
            for x_idx in range(3):
                if stay_totals[N, x_idx] == 0 and N > 0:
                    # Use average of observed states
                    d_hat[N, x_idx] = np.mean(d_hat[d_hat > 0]) if np.any(d_hat > 0) else 0.5
                
                if entry_totals[N, x_idx] == 0 and N < 5:
                    e_hat[N, x_idx] = np.mean(e_hat[e_hat > 0]) if np.any(e_hat > 0) else 0.5
        
        self.d_hat = d_hat
        self.e_hat = e_hat
        
        print(f"Estimated stay probabilities (sample):")
        print(f"  d̂(1,0) = {d_hat[1,1]:.4f}")
        print(f"  d̂(3,0) = {d_hat[3,1]:.4f}")
        print(f"  d̂(5,0) = {d_hat[5,1]:.4f}")
        print(f"Estimated entry probabilities (sample):")
        print(f"  ê(0,0) = {e_hat[0,1]:.4f}")
        print(f"  ê(2,0) = {e_hat[2,1]:.4f}")
        print(f"  ê(4,0) = {e_hat[4,1]:.4f}")
        
        return d_hat, e_hat
    
    def step_bc_forward_simulate_incumbent(self, N_start, x_idx_start, 
                                          params, n_sims=50, max_T=100,
                                          base_random_draws=None):
        """
        Steps B-C: Forward simulate PDV for incumbent
        
        Critical: Use same random draws across parameter values (importance sampling)
        """
        mu_mean, mu_std, gamma_mean, gamma_std = params
        
        # Generate or use base random draws
        if base_random_draws is None:
            base_random_draws = {
                'mu': np.random.randn(n_sims, max_T, 10),  # 10 = max possible firms
                'gamma': np.random.randn(n_sims, max_T),
                'x': np.random.rand(n_sims, max_T)
            }
        
        pdvs = []
        
        for sim in range(n_sims):
            N = N_start
            x_idx = x_idx_start
            total_pdv = 0
            discount = 1.0
            firm_1_active = True
            
            for t in range(max_T):
                if not firm_1_active:
                    break
                
                # Current profit for Firm 1
                profit = self.game.compute_profit(N, self.game.x_states[x_idx])
                total_pdv += discount * profit
                
                # Firm 1's exit decision (we condition on staying in period 0)
                if t > 0:
                    mu_draw_1 = mu_mean + mu_std * base_random_draws['mu'][sim, t, 0]
                    cdf_val = norm.cdf((mu_draw_1 - mu_mean) / mu_std)
                    
                    if cdf_val > self.d_hat[N, x_idx]:
                        # Firm 1 exits
                        total_pdv += discount * mu_draw_1
                        firm_1_active = False
                        break
                
                # Other firms' exit decisions
                n_stay_others = 0
                for i in range(1, N):  # Firms 2 through N
                    mu_draw_i = mu_mean + mu_std * base_random_draws['mu'][sim, t, i]
                    cdf_val_i = norm.cdf((mu_draw_i - mu_mean) / mu_std)
                    
                    if cdf_val_i <= self.d_hat[N, x_idx]:
                        n_stay_others += 1
                
                # Entry decision
                n_enter = 0
                if N < 5:
                    gamma_draw = gamma_mean + gamma_std * base_random_draws['gamma'][sim, t]
                    cdf_val_entry = norm.cdf((gamma_draw - gamma_mean) / gamma_std)
                    
                    if cdf_val_entry <= self.e_hat[N, x_idx]:
                        n_enter = 1
                
                # Update number of firms (Firm 1 + others who stay + entrants)
                N = 1 + n_stay_others + n_enter
                N = min(N, 5)  # Cap at 5
                
                # Update demand state
                x_probs = self.game.x_trans[x_idx]
                x_rand = base_random_draws['x'][sim, t]
                x_idx = np.searchsorted(np.cumsum(x_probs), x_rand)
                x_idx = min(x_idx, 2)
                
                discount *= self.game.beta
            
            pdvs.append(total_pdv)
        
        return np.mean(pdvs), base_random_draws
    
    def step_e_forward_simulate_entrant(self, N_start, x_idx_start,
                                       params, n_sims=50, max_T=100,
                                       base_random_draws=None):
        """
        Step E: Forward simulate PDV for potential entrant
        
        Returns PDV net of entry cost (don't include -gamma_it in period 0)
        """
        mu_mean, mu_std, gamma_mean, gamma_std = params
        
        if base_random_draws is None:
            base_random_draws = {
                'mu': np.random.randn(n_sims, max_T, 10),
                'gamma': np.random.randn(n_sims, max_T),
                'x': np.random.rand(n_sims, max_T)
            }
        
        pdvs = []
        
        for sim in range(n_sims):
            N = N_start + 1  # Entrant becomes incumbent
            x_idx = x_idx_start
            total_pdv = 0
            discount = self.game.beta  # Start from next period
            firm_active = True
            
            for t in range(max_T):
                if not firm_active:
                    break
                
                # Current profit
                profit = self.game.compute_profit(N, self.game.x_states[x_idx])
                total_pdv += discount * profit
                
                # Exit decision for our firm
                mu_draw = mu_mean + mu_std * base_random_draws['mu'][sim, t, 0]
                cdf_val = norm.cdf((mu_draw - mu_mean) / mu_std)
                
                if cdf_val > self.d_hat[N, x_idx]:
                    # Firm exits
                    total_pdv += discount * mu_draw
                    firm_active = False
                    break
                
                # Other firms' decisions (similar to incumbent simulation)
                n_stay_others = 0
                for i in range(1, N):
                    mu_draw_i = mu_mean + mu_std * base_random_draws['mu'][sim, t, i]
                    cdf_val_i = norm.cdf((mu_draw_i - mu_mean) / mu_std)
                    if cdf_val_i <= self.d_hat[N, x_idx]:
                        n_stay_others += 1
                
                # Entry
                n_enter = 0
                if N < 5:
                    gamma_draw = gamma_mean + gamma_std * base_random_draws['gamma'][sim, t]
                    cdf_val_entry = norm.cdf((gamma_draw - gamma_mean) / gamma_std)
                    if cdf_val_entry <= self.e_hat[N, x_idx]:
                        n_enter = 1
                
                N = 1 + n_stay_others + n_enter
                N = min(N, 5)
                
                # Update x
                x_probs = self.game.x_trans[x_idx]
                x_rand = base_random_draws['x'][sim, t]
                x_idx = np.searchsorted(np.cumsum(x_probs), x_rand)
                x_idx = min(x_idx, 2)
                
                discount *= self.game.beta
            
            pdvs.append(total_pdv)
        
        return np.mean(pdvs), base_random_draws
    
    def step_f_objective_function(self, params_raw):
        """
        Step F: Minimum distance objective function
        
        params_raw = [mu_mean, log(mu_std), gamma_mean, log(gamma_std)]
        (use log for std parameters to ensure positivity)
        """
        mu_mean = params_raw[0]
        mu_std = np.exp(params_raw[1])
        gamma_mean = params_raw[2]
        gamma_std = np.exp(params_raw[3])
        
        params = (mu_mean, mu_std, gamma_mean, gamma_std)
        
        print(f"\nEvaluating at: μ={mu_mean:.3f}, σ_μ={mu_std:.3f}, "
              f"γ={gamma_mean:.3f}, σ_γ={gamma_std:.3f}")
        
        # Generate base random draws (fixed across all states)
        np.random.seed(12345)  # Fixed seed for reproducibility
        base_draws_inc = {
            'mu': np.random.randn(50, 100, 10),
            'gamma': np.random.randn(50, 100),
            'x': np.random.rand(50, 100)
        }
        base_draws_ent = {
            'mu': np.random.randn(50, 100, 10),
            'gamma': np.random.randn(50, 100),
            'x': np.random.rand(50, 100)
        }
        
        objective = 0
        n_moments = 0
        
        # Incumbent moments
        for N in range(1, 6):
            for x_idx in range(3):
                # Skip if never observed
                if self.d_hat[N, x_idx] == 0 or self.d_hat[N, x_idx] == 1:
                    continue
                
                # Simulate Lambda
                Lambda, _ = self.step_bc_forward_simulate_incumbent(
                    N, x_idx, params, n_sims=50, max_T=100,
                    base_random_draws=base_draws_inc
                )
                
                # Implied probability
                prob_stay = norm.cdf((Lambda - mu_mean) / mu_std)
                
                # Moment condition
                moment = prob_stay - self.d_hat[N, x_idx]
                objective += moment**2
                n_moments += 1
        
        # Entry moments
        for N in range(5):
            for x_idx in range(3):
                if self.e_hat[N, x_idx] == 0 or self.e_hat[N, x_idx] == 1:
                    continue
                
                # Simulate Lambda^E
                Lambda_E, _ = self.step_e_forward_simulate_entrant(
                    N, x_idx, params, n_sims=50, max_T=100,
                    base_random_draws=base_draws_ent
                )
                
                # Implied probability
                prob_enter = norm.cdf((Lambda_E - gamma_mean) / gamma_std)
                
                # Moment condition
                moment = prob_enter - self.e_hat[N, x_idx]
                objective += moment**2
                n_moments += 1
        
        print(f"  Objective = {objective:.6f} ({n_moments} moments)")
        
        return objective
    
    def estimate(self, initial_params=None):
        """
        Main estimation routine
        
        Returns: Dictionary with estimated parameters and diagnostics
        """
        print("\n" + "="*70)
        print("STARTING BBL ESTIMATION")
        print("="*70)
        
        # Step A: Estimate policy functions
        self.step_a_estimate_policies()
        
        # Step F: Minimize objective function
        print("\n=== Steps B-F: Estimating Structural Parameters ===")
        
        if initial_params is None:
            # Start near truth but not exactly at it
            initial_params = np.array([4.5, np.log(2.0), 4.5, np.log(2.0)])
        
        print(f"\nInitial parameters: μ={initial_params[0]:.3f}, "
              f"σ_μ={np.exp(initial_params[1]):.3f}, "
              f"γ={initial_params[2]:.3f}, σ_γ={np.exp(initial_params[3]):.3f}")
        
        # Optimization
        result = minimize(
            self.step_f_objective_function,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': True, 'xatol': 1e-3, 'fatol': 1e-4}
        )
        
        # Extract estimates
        mu_mean_hat = result.x[0]
        mu_std_hat = np.exp(result.x[1])
        gamma_mean_hat = result.x[2]
        gamma_std_hat = np.exp(result.x[3])
        
        print("\n" + "="*70)
        print("ESTIMATION RESULTS")
        print("="*70)
        print(f"\nTrue parameters:")
        print(f"  μ = 5.000,  σ²_μ = 5.000")
        print(f"  γ = 5.000,  σ²_γ = 5.000")
        print(f"\nEstimated parameters:")
        print(f"  μ̂ = {mu_mean_hat:.4f},  σ̂²_μ = {mu_std_hat**2:.4f}")
        print(f"  γ̂ = {gamma_mean_hat:.4f},  σ̂²_γ = {gamma_std_hat**2:.4f}")
        print(f"\nEstimation errors:")
        print(f"  Δμ = {abs(mu_mean_hat - 5):.4f}")
        print(f"  Δσ²_μ = {abs(mu_std_hat**2 - 5):.4f}")
        print(f"  Δγ = {abs(gamma_mean_hat - 5):.4f}")
        print(f"  Δσ²_γ = {abs(gamma_std_hat**2 - 5):.4f}")
        print(f"\nObjective value: {result.fun:.6f}")
        print(f"Converged: {result.success}")
        print(f"Iterations: {result.nit}")
        
        return {
            'mu_mean': mu_mean_hat,
            'mu_var': mu_std_hat**2,
            'gamma_mean': gamma_mean_hat,
            'gamma_var': gamma_std_hat**2,
            'objective': result.fun,
            'success': result.success,
            'result': result
        }


def run_complete_analysis():
    """
    Run complete homework solution with BBL estimation
    """
    from dynamic_entry_exit_game import DynamicEntryExitGame
    
    # Initialize
    game = DynamicEntryExitGame()
    
    # Solve equilibrium
    print("Solving for equilibrium...")
    equilibrium = game.solve_equilibrium(verbose=True)
    
    # Simulate with decision tracking
    print("\nSimulating market (this will take a moment)...")
    
    # Need to modify simulation to track decisions
    simulation_data = simulate_with_decisions(game, equilibrium, periods=10000)
    
    # Run BBL estimation
    estimator = BBLEstimator(game, simulation_data)
    estimates = estimator.estimate()
    
    # Save results
    with open('bbl_results.pkl', 'wb') as f:
        pickle.dump({
            'equilibrium': equilibrium,
            'simulation': simulation_data,
            'estimates': estimates
        }, f)
    
    print("\nResults saved to 'bbl_results.pkl'")
    
    return estimates


def simulate_with_decisions(game, equilibrium, periods=10000, seed=42):
    """
    Modified simulation that tracks individual decisions
    """
    np.random.seed(seed)
    
    N_history = []
    x_history = []
    stay_decisions = np.zeros((periods, 5))  # Max 5 firms
    entry_decisions = np.zeros(periods)
    
    N = 0
    x_idx = 1
    
    for t in range(periods):
        N_history.append(N)
        x_history.append(game.x_states[x_idx])
        
        # Track exit decisions
        if N > 0:
            for i in range(N):
                mu_draw = np.random.normal(game.mu_mean, game.mu_std)
                stay = mu_draw <= equilibrium['mu'][N, x_idx]
                stay_decisions[t, i] = 1 if stay else 0
            
            N = int(np.sum(stay_decisions[t, :N]))
        
        # Track entry decision
        if N < 5:
            gamma_draw = np.random.normal(game.gamma_mean, game.gamma_std)
            enters = gamma_draw <= equilibrium['gamma'][N, x_idx]
            entry_decisions[t] = 1 if enters else 0
            if enters:
                N += 1
        
        # Update x
        x_idx = np.random.choice(3, p=game.x_trans[x_idx])
    
    return {
        'N_history': np.array(N_history),
        'x_history': np.array(x_history),
        'stay_decisions': stay_decisions,
        'entry_decisions': entry_decisions
    }


if __name__ == "__main__":
    run_complete_analysis()
