import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
import time

@dataclass
class Parameters:
    """Model parameters"""
    gamma: float = 5.0      # Mean entry cost
    sigma_gamma: float = np.sqrt(5.0)  # Std dev of entry cost
    mu: float = 5.0         # Mean exit value
    sigma_mu: float = np.sqrt(5.0)     # Std dev of exit value
    beta: float = 0.9       # Discount factor
    fixed_cost: float = 5.0  # Per-period fixed cost
    max_firms: int = 5      # Maximum number of firms

class DynamicEntryExitGame:
    def __init__(self, params: Parameters):
        self.params = params
        self.x_values = np.array([-5, 0, 5])
        self.n_values = np.arange(0, params.max_firms + 1)
        
        # Transition matrix for demand shifter
        self.P_x = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6]
        ])
        
        # State space: (N, x) pairs
        self.states = [(n, x) for n in self.n_values for x in self.x_values]
        self.n_states = len(self.states)
        
        # Initialize cutoffs and value functions
        self.mu_cutoff = {}  # Not defined for N=0
        self.gamma_cutoff = {}  # Not defined for N=5
        self.V_bar = {}  # Not defined for N=0
        
    def profit(self, N: int, x: float) -> float:
        """Compute per-firm Nash equilibrium profit"""
        if N == 0:
            return 0.0
        numerator = (10 + x) / (N + 1)
        return numerator**2 - self.params.fixed_cost
    
    def stay_probability(self, N: int, x: float) -> float:
        """Probability an incumbent stays given cutoff"""
        if N == 0:
            return 0.0
        cutoff = self.mu_cutoff.get((N, x), 0.0)
        return norm.cdf((cutoff - self.params.mu) / self.params.sigma_mu)
    
    def entry_probability(self, N: int, x: float) -> float:
        """Probability potential entrant enters given cutoff"""
        if N >= self.params.max_firms:
            return 0.0
        cutoff = self.gamma_cutoff.get((N, x), 0.0)
        return norm.cdf((cutoff - self.params.gamma) / self.params.sigma_gamma)
    
    def transition_matrix_stay(self, N: int, x: float) -> np.ndarray:
        """
        Transition matrix for N_{t+1} | N_t, x_t, d_it=1
        Given that one incumbent stays, compute probability distribution over N_{t+1}
        """
        P = np.zeros(self.params.max_firms + 1)
        
        if N == 0:
            return P
        
        p_stay = self.stay_probability(N, x)
        p_entry = self.entry_probability(N, x)
        
        # Other N-1 incumbents each stay with probability p_stay
        # Potential entrant enters with probability p_entry
        # This incumbent stays (conditioning event)
        
        for n_other_stay in range(N):  # 0 to N-1 others can stay
            prob_other_stay = (np.math.comb(N-1, n_other_stay) * 
                             p_stay**n_other_stay * 
                             (1-p_stay)**(N-1-n_other_stay))
            
            # No entry
            N_next = 1 + n_other_stay  # This firm + others who stay
            if N_next <= self.params.max_firms:
                P[N_next] += prob_other_stay * (1 - p_entry)
            
            # Entry occurs
            N_next = 1 + n_other_stay + 1
            if N_next <= self.params.max_firms:
                P[N_next] += prob_other_stay * p_entry
        
        return P
    
    def transition_matrix_entry(self, N: int, x: float) -> np.ndarray:
        """
        Transition matrix for N_{t+1} | N_t, x_t, e_it=1
        Given that potential entrant enters, compute probability distribution over N_{t+1}
        """
        P = np.zeros(self.params.max_firms + 1)
        
        if N >= self.params.max_firms:
            return P
        
        p_stay = self.stay_probability(N, x)
        
        # N incumbents each stay with probability p_stay
        # This entrant enters (conditioning event) and becomes an incumbent next period
        
        for n_stay in range(N + 1):  # 0 to N incumbents can stay
            prob_stay = (np.math.comb(N, n_stay) * 
                        p_stay**n_stay * 
                        (1-p_stay)**(N-n_stay))
            
            N_next = n_stay + 1  # Staying incumbents + entrant
            if N_next <= self.params.max_firms:
                P[N_next] += prob_stay
        
        return P
    
    def compute_continuation_value(self, N: int, x: float, for_entry: bool = False) -> float:
        """
        Compute Psi_1 or Psi_2: discounted expected continuation value
        """
        value = 0.0
        x_idx = np.where(self.x_values == x)[0][0]
        
        if for_entry:
            trans_matrix = self.transition_matrix_entry(N, x)
        else:
            trans_matrix = self.transition_matrix_stay(N, x)
        
        for N_next in range(self.params.max_firms + 1):
            for x_next_idx, x_next in enumerate(self.x_values):
                if N_next > 0:  # V_bar only defined for N > 0
                    V_next = self.V_bar.get((N_next, x_next), 0.0)
                    value += (self.params.beta * V_next * 
                             self.P_x[x_idx, x_next_idx] * 
                             trans_matrix[N_next])
        
        return value
    
    def compute_V_bar(self, N: int, x: float) -> float:
        """
        Compute ex-ante value function V_bar(N, x)
        """
        if N == 0:
            return 0.0
        
        pi = self.profit(N, x)
        psi1 = self.compute_continuation_value(N, x, for_entry=False)
        cutoff = pi + psi1
        
        # Standardize
        z = (cutoff - self.params.mu) / self.params.sigma_mu
        
        # Probability of staying
        prob_stay = norm.cdf(z)
        
        # Expected value if staying
        value_stay = cutoff * prob_stay
        
        # Expected value if exiting (truncated mean)
        prob_exit = 1 - prob_stay
        if prob_exit > 1e-10:
            inverse_mills = norm.pdf(z) / prob_exit
            expected_exit = self.params.mu + self.params.sigma_mu * inverse_mills
            value_exit = prob_exit * expected_exit
        else:
            value_exit = 0.0
        
        return value_stay + value_exit
    
    def initialize_values(self, init_type: str = 'zero'):
        """Initialize cutoffs and value functions"""
        if init_type == 'zero':
            for N in range(1, self.params.max_firms + 1):
                for x in self.x_values:
                    self.mu_cutoff[(N, x)] = 0.0
                    self.V_bar[(N, x)] = 0.0
            
            for N in range(0, self.params.max_firms):
                for x in self.x_values:
                    self.gamma_cutoff[(N, x)] = 0.0
        
        elif init_type == 'random':
            np.random.seed(int(time.time() * 1000) % 2**32)
            for N in range(1, self.params.max_firms + 1):
                for x in self.x_values:
                    self.mu_cutoff[(N, x)] = np.random.uniform(-5, 15)
                    self.V_bar[(N, x)] = np.random.uniform(0, 50)
            
            for N in range(0, self.params.max_firms):
                for x in self.x_values:
                    self.gamma_cutoff[(N, x)] = np.random.uniform(-5, 15)
        
        elif init_type == 'high':
            for N in range(1, self.params.max_firms + 1):
                for x in self.x_values:
                    self.mu_cutoff[(N, x)] = 20.0
                    self.V_bar[(N, x)] = 100.0
            
            for N in range(0, self.params.max_firms):
                for x in self.x_values:
                    self.gamma_cutoff[(N, x)] = 20.0
    
    def solve_equilibrium(self, tol: float = 1e-6, max_iter: int = 1000, verbose: bool = True):
        """
        Solve for Markov Perfect Equilibrium using iteration
        """
        for iteration in range(max_iter):
            # Store old values
            old_mu = self.mu_cutoff.copy()
            old_gamma = self.gamma_cutoff.copy()
            old_V = self.V_bar.copy()
            
            # Update cutoffs
            for N in range(1, self.params.max_firms + 1):
                for x in self.x_values:
                    pi = self.profit(N, x)
                    psi1 = self.compute_continuation_value(N, x, for_entry=False)
                    self.mu_cutoff[(N, x)] = pi + psi1
            
            for N in range(0, self.params.max_firms):
                for x in self.x_values:
                    psi2 = self.compute_continuation_value(N, x, for_entry=True)
                    self.gamma_cutoff[(N, x)] = psi2
            
            # Update V_bar
            for N in range(1, self.params.max_firms + 1):
                for x in self.x_values:
                    self.V_bar[(N, x)] = self.compute_V_bar(N, x)
            
            # Check convergence
            max_diff = 0.0
            for key in old_mu:
                max_diff = max(max_diff, abs(self.mu_cutoff[key] - old_mu[key]))
            for key in old_gamma:
                max_diff = max(max_diff, abs(self.gamma_cutoff[key] - old_gamma[key]))
            for key in old_V:
                max_diff = max(max_diff, abs(self.V_bar[key] - old_V[key]))
            
            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: max difference = {max_diff:.2e}")
            
            if max_diff < tol:
                if verbose:
                    print(f"Converged in {iteration} iterations!")
                return True
        
        print(f"Warning: Did not converge after {max_iter} iterations")
        return False
    
    def compute_V_with_shock(self, N: int, x: float, mu_shock: float) -> float:
        """
        Compute V(N, x, mu) for a specific shock realization
        """
        if N == 0:
            return 0.0
        
        pi = self.profit(N, x)
        psi1 = self.compute_continuation_value(N, x, for_entry=False)
        stay_value = pi + psi1
        exit_value = mu_shock
        
        return max(exit_value, stay_value)
    
    def simulate_market(self, T: int = 10000, N_init: int = 0, x_init: float = 0.0, 
                       seed: int = 42) -> Dict:
        """
        Simulate market evolution for T periods
        """
        np.random.seed(seed)
        
        # Storage
        N_path = np.zeros(T, dtype=int)
        x_path = np.zeros(T)
        entry_path = np.zeros(T, dtype=bool)
        exit_path = np.zeros(T, dtype=int)
        
        # Initial state
        N_t = N_init
        x_t = x_init
        
        for t in range(T):
            N_path[t] = N_t
            x_path[t] = x_t
            
            # Exit decisions for incumbents
            n_exits = 0
            if N_t > 0:
                for i in range(N_t):
                    mu_shock = np.random.normal(self.params.mu, self.params.sigma_mu)
                    if mu_shock > self.mu_cutoff[(N_t, x_t)]:
                        n_exits += 1
            
            exit_path[t] = n_exits
            
            # Entry decision
            entered = False
            if N_t < self.params.max_firms:
                gamma_shock = np.random.normal(self.params.gamma, self.params.sigma_gamma)
                if gamma_shock <= self.gamma_cutoff[(N_t, x_t)]:
                    entered = True
            
            entry_path[t] = entered
            
            # Update number of firms
            N_t = N_t - n_exits + int(entered)
            N_t = max(0, min(N_t, self.params.max_firms))
            
            # Update demand shifter
            x_idx = np.where(self.x_values == x_t)[0][0]
            x_t = np.random.choice(self.x_values, p=self.P_x[x_idx])
        
        return {
            'N': N_path,
            'x': x_path,
            'entry': entry_path,
            'exit': exit_path
        }

def problem_4_solve():
    """Problem 4: Solve for equilibrium"""
    print("="*80)
    print("PROBLEM 4: Solving for Equilibrium")
    print("="*80)
    
    params = Parameters()
    game = DynamicEntryExitGame(params)
    game.initialize_values('zero')
    game.solve_equilibrium(tol=1e-6, verbose=True)
    
    print("\nEquilibrium Cutoffs (selected states):")
    print(f"μ(3, 0) = {game.mu_cutoff[(3, 0)]:.4f}")
    print(f"γ(3, 0) = {game.gamma_cutoff[(3, 0)]:.4f}")
    print(f"V̄(3, 0) = {game.V_bar[(3, 0)]:.4f}")
    print(f"V(3, 0, -2) = {game.compute_V_with_shock(3, 0, -2):.4f}")
    
    return game

def problem_5_multiple_equilibria():
    """Problem 5: Test for multiple equilibria"""
    print("\n" + "="*80)
    print("PROBLEM 5: Testing for Multiple Equilibria")
    print("="*80)
    
    params = Parameters()
    results = []
    init_types = ['zero', 'random', 'high', 'random', 'random']
    
    for i, init_type in enumerate(init_types):
        print(f"\n--- Attempt {i+1}: Initialization = {init_type} ---")
        game = DynamicEntryExitGame(params)
        game.initialize_values(init_type)
        game.solve_equilibrium(tol=1e-6, verbose=False)
        
        # Store key values
        result = {
            'mu_3_0': game.mu_cutoff[(3, 0)],
            'gamma_3_0': game.gamma_cutoff[(3, 0)],
            'V_bar_3_0': game.V_bar[(3, 0)]
        }
        results.append(result)
        print(f"μ(3, 0) = {result['mu_3_0']:.6f}")
        print(f"γ(3, 0) = {result['gamma_3_0']:.6f}")
        print(f"V̄(3, 0) = {result['V_bar_3_0']:.6f}")
    
    # Check if all converged to same equilibrium
    print("\n--- Analysis ---")
    mu_values = [r['mu_3_0'] for r in results]
    if max(mu_values) - min(mu_values) < 1e-4:
        print("✓ All initializations converged to the SAME equilibrium")
        print("  No evidence of multiple equilibria found")
    else:
        print("✗ Different initializations led to DIFFERENT equilibria")
        print("  Evidence of multiple equilibria!")
    
    return results

def problem_7_simulation(game):
    """Problem 7: Simulate market for 10000 periods"""
    print("\n" + "="*80)
    print("PROBLEM 7: Market Simulation (10,000 periods)")
    print("="*80)
    
    sim_data = game.simulate_market(T=10000, N_init=0, x_init=0.0, seed=42)
    
    avg_firms = np.mean(sim_data['N'])
    print(f"\nAverage number of firms: {avg_firms:.4f}")
    print(f"Standard deviation: {np.std(sim_data['N']):.4f}")
    print(f"Min firms: {np.min(sim_data['N'])}")
    print(f"Max firms: {np.max(sim_data['N'])}")
    
    # Distribution
    print("\nDistribution of firm counts:")
    for n in range(6):
        pct = 100 * np.mean(sim_data['N'] == n)
        print(f"  N = {n}: {pct:.2f}%")
    
    return sim_data

def problem_8_entry_tax():
    """Problem 8: Effect of entry tax"""
    print("\n" + "="*80)
    print("PROBLEM 8: Effect of 5-Unit Entry Tax")
    print("="*80)
    
    # Baseline
    params_base = Parameters()
    game_base = DynamicEntryExitGame(params_base)
    game_base.initialize_values('zero')
    game_base.solve_equilibrium(tol=1e-6, verbose=False)
    sim_base = game_base.simulate_market(T=10000, N_init=0, x_init=0.0, seed=42)
    avg_base = np.mean(sim_base['N'])
    
    # With tax (increase mean entry cost by 5)
    params_tax = Parameters(gamma=10.0)  # gamma = 5 + 5 tax
    game_tax = DynamicEntryExitGame(params_tax)
    game_tax.initialize_values('zero')
    game_tax.solve_equilibrium(tol=1e-6, verbose=False)
    sim_tax = game_tax.simulate_market(T=10000, N_init=0, x_init=0.0, seed=42)
    avg_tax = np.mean(sim_tax['N'])
    
    print(f"Average firms (baseline): {avg_base:.4f}")
    print(f"Average firms (with 5-unit tax): {avg_tax:.4f}")
    print(f"Change: {avg_tax - avg_base:.4f} ({100*(avg_tax/avg_base - 1):.2f}%)")
    
    print("\nWith a unique equilibrium, this comparative static is well-defined.")
    print("With multiple equilibria, the answer would depend on equilibrium selection.")

# Run all problems
if __name__ == "__main__":
    print("DYNAMIC ENTRY/EXIT GAME - COMPUTATIONAL SOLUTION")
    print("="*80)
    
    # Problem 4
    game = problem_4_solve()
    
    # Problem 5
    problem_5_multiple_equilibria()
    
    # Problem 7
    sim_data = problem_7_simulation(game)
    
    # Problem 8
    problem_8_entry_tax()
    
    print("\n" + "="*80)
    print("Note: Problems 9-10 (BBL Estimation) require additional implementation")
    print("See separate estimation module for complete BBL estimator")
    print("="*80)