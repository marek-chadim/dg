import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class DynamicEntryExitGame:
    """
    Solver for dynamic entry/exit game with Cournot competition.
    Based on Ericson & Pakes (1995) framework with BBL estimation.
    """
    
    def __init__(self, gamma=5, sigma_gamma=np.sqrt(5), mu=5, sigma_mu=np.sqrt(5)):
        """
        Initialize model parameters.
        
        Parameters:
        -----------
        gamma : float
            Mean entry cost
        sigma_gamma : float
            Std dev of entry cost
        mu : float
            Mean exit value
        sigma_mu : float
            Std dev of exit value
        """
        self.gamma = gamma
        self.sigma_gamma = sigma_gamma
        self.mu = mu
        self.sigma_mu = sigma_mu
        
        # Model primitives
        self.beta = 0.9  # Discount factor
        self.N_max = 5   # Maximum number of firms
        self.x_values = np.array([-5, 0, 5])  # Demand shifter values
        
        # Transition matrix for demand shifter
        self.P_x = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6]
        ])
        
        # State space: N_t in {0,1,2,3,4,5}, x_t in {-5,0,5}
        self.states = [(n, x) for n in range(self.N_max + 1) 
                       for x in self.x_values]
        
        # Initialize equilibrium objects
        self.mu_cutoff = {}
        self.gamma_cutoff = {}
        self.V_bar = {}
        
    def compute_pi(self, N: int, x: float) -> float:
        """
        Question 1: Compute Nash equilibrium single-period profit per firm.
        
        In Cournot competition with N firms, inverse demand p = 10 - Q + x,
        MC = 0, and fixed cost = 5:
        
        Each firm produces: q* = (10 + x)/(N + 1)
        Price: p* = (10 + x)/(N + 1)
        Profit per firm: π = p*·q* - 5 = [(10 + x)/(N + 1)]^2 - 5
        
        Parameters:
        -----------
        N : int
            Number of firms
        x : float
            Demand shifter
            
        Returns:
        --------
        float : Per-firm profit
        """
        if N == 0:
            return 0.0
        
        profit = ((10 + x) / (N + 1))**2 - 5
        return profit
    
    def compute_transition_prob(self, N: int, x: int, decision: str, 
                                cutoff_dict: dict) -> np.ndarray:
        """
        Compute transition probabilities for number of firms.
        
        Parameters:
        -----------
        N : int
            Current number of firms
        x : float
            Current demand shifter
        decision : str
            'd' for incumbent decision (d_it=1), 'e' for entrant decision (e_it=1)
        cutoff_dict : dict
            Dictionary of cutoff values
            
        Returns:
        --------
        np.ndarray : Probability distribution over N_{t+1}
        """
        prob = np.zeros(self.N_max + 1)
        
        if decision == 'd':  # Incumbent stays
            if N == 0:
                prob[0] = 1.0
                return prob
                
            # Probability each incumbent stays (μ_it ≤ μ(N,x))
            mu_cut = cutoff_dict.get((N, x), self.mu)
            p_stay = norm.cdf((mu_cut - self.mu) / self.sigma_mu)
            
            # Binomial distribution for number of staying firms
            for n_stay in range(N + 1):
                # Start with n_stay staying firms (includes the conditioned firm)
                # Actually need n_stay - 1 others to stay (conditioning on one staying)
                if n_stay == 0:
                    continue
                prob_n_stay = (binom_pmf(n_stay - 1, N - 1, p_stay) if N > 1 
                              else (1.0 if n_stay == 1 else 0.0))
                
                # Add potential entrant if N < N_max
                if N < self.N_max:
                    gamma_cut = cutoff_dict.get((n_stay, x), self.gamma)
                    p_enter = norm.cdf((gamma_cut - self.gamma) / self.sigma_gamma)
                    prob[n_stay] += prob_n_stay * (1 - p_enter)
                    if n_stay + 1 <= self.N_max:
                        prob[n_stay + 1] += prob_n_stay * p_enter
                else:
                    prob[n_stay] += prob_n_stay
                    
        else:  # Entrant enters (e_it = 1)
            if N >= self.N_max:
                prob[N] = 1.0
                return prob
                
            # Probability each incumbent stays
            mu_cut = cutoff_dict.get((N, x), self.mu) if N > 0 else self.mu
            p_stay = norm.cdf((mu_cut - self.mu) / self.sigma_mu) if N > 0 else 0
            
            # Distribution over staying incumbents
            for n_stay in range(N + 1):
                prob_n_stay = binom_pmf(n_stay, N, p_stay)
                # Add the entrant (enters next period)
                if n_stay + 1 <= self.N_max:
                    prob[n_stay + 1] += prob_n_stay
                else:
                    prob[self.N_max] += prob_n_stay
                    
        return prob
    
    def solve_equilibrium(self, max_iter=1000, tol=1e-6, verbose=True):
        """
        Question 4: Solve for equilibrium using iterative procedure.
        
        Algorithm:
        1. Guess μ(N,x), γ(N,x), V̄(N,x)
        2. Compute transition matrices
        3. Compute continuation values Ψ₁, Ψ₂
        4. Update cutoffs: μ' = π + Ψ₁, γ' = Ψ₂
        5. Update V̄' using truncated normal formula
        6. Iterate until convergence
        """
        if verbose:
            print("Solving for equilibrium...")
            print(f"Parameters: γ={self.gamma}, σ_γ={self.sigma_gamma}, "
                  f"μ={self.mu}, σ_μ={self.sigma_mu}")
        
        # Step 1: Initialize guesses
        for (N, x) in self.states:
            if N > 0:
                self.mu_cutoff[(N, x)] = self.compute_pi(N, x)
                self.V_bar[(N, x)] = self.compute_pi(N, x) / (1 - self.beta)
            if N < self.N_max:
                self.gamma_cutoff[(N, x)] = 0.0
        
        for iteration in range(max_iter):
            mu_old = self.mu_cutoff.copy()
            gamma_old = self.gamma_cutoff.copy()
            V_bar_old = self.V_bar.copy()
            
            # Step 2 & 3: Compute Ψ₁ and Ψ₂
            Psi_1 = {}
            Psi_2 = {}
            
            for (N, x) in self.states:
                # Ψ₁: continuation value for incumbent
                if N > 0:
                    value = 0.0
                    x_idx = np.where(self.x_values == x)[0][0]
                    
                    for x_idx_next, x_next in enumerate(self.x_values):
                        prob_x = self.P_x[x_idx, x_idx_next]
                        
                        # Transition over N given incumbent stays
                        trans_N = self.compute_transition_prob(
                            N, x, 'd', self.mu_cutoff)
                        
                        for N_next in range(self.N_max + 1):
                            if N_next > 0 and trans_N[N_next] > 0:
                                value += (self.beta * prob_x * trans_N[N_next] * 
                                         self.V_bar.get((N_next, x_next), 0))
                    
                    Psi_1[(N, x)] = value
                
                # Ψ₂: continuation value for entrant
                if N < self.N_max:
                    value = 0.0
                    x_idx = np.where(self.x_values == x)[0][0]
                    
                    for x_idx_next, x_next in enumerate(self.x_values):
                        prob_x = self.P_x[x_idx, x_idx_next]
                        
                        # Transition over N given entrant enters
                        trans_N = self.compute_transition_prob(
                            N, x, 'e', self.gamma_cutoff)
                        
                        for N_next in range(self.N_max + 1):
                            if N_next > 0 and trans_N[N_next] > 0:
                                value += (self.beta * prob_x * trans_N[N_next] * 
                                         self.V_bar.get((N_next, x_next), 0))
                    
                    Psi_2[(N, x)] = value
            
            # Step 4: Update cutoffs
            for (N, x) in self.states:
                if N > 0:
                    self.mu_cutoff[(N, x)] = (self.compute_pi(N, x) + 
                                             Psi_1.get((N, x), 0))
                if N < self.N_max:
                    self.gamma_cutoff[(N, x)] = Psi_2.get((N, x), 0)
            
            # Step 5: Update V̄ using truncated normal formula
            for (N, x) in self.states:
                if N > 0:
                    pi_val = self.compute_pi(N, x)
                    psi_val = Psi_1.get((N, x), 0)
                    threshold = pi_val + psi_val
                    
                    # Standardized threshold
                    z = (threshold - self.mu) / self.sigma_mu
                    
                    # Probability of staying
                    prob_stay = norm.cdf(z)
                    
                    # Expected value if staying (truncated normal)
                    if prob_stay > 1e-10:
                        mills = norm.pdf(z) / (1 - norm.cdf(z))
                        E_mu_stay = self.mu + self.sigma_mu * mills
                    else:
                        E_mu_stay = threshold
                    
                    # V̄ = P(exit) × E[μ|exit] + P(stay) × threshold
                    self.V_bar[(N, x)] = ((1 - prob_stay) * E_mu_stay + 
                                         prob_stay * threshold)
            
            # Check convergence
            max_diff = 0
            for key in mu_old:
                max_diff = max(max_diff, abs(self.mu_cutoff[key] - mu_old[key]))
            for key in gamma_old:
                max_diff = max(max_diff, abs(self.gamma_cutoff[key] - gamma_old[key]))
            for key in V_bar_old:
                max_diff = max(max_diff, abs(self.V_bar[key] - V_bar_old[key]))
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: max difference = {max_diff:.6e}")
            
            if max_diff < tol:
                if verbose:
                    print(f"Converged in {iteration + 1} iterations!")
                return True
        
        if verbose:
            print(f"Did not converge after {max_iter} iterations")
        return False
    
    def simulate_market(self, N_init=0, x_init=0, T=10000, seed=None):
        """
        Question 7: Simulate market dynamics.
        
        Parameters:
        -----------
        N_init : int
            Initial number of firms
        x_init : float
            Initial demand shifter
        T : int
            Number of periods to simulate
        seed : int
            Random seed
            
        Returns:
        --------
        pd.DataFrame : Simulated data
        """
        if seed is not None:
            np.random.seed(seed)
        
        data = []
        N_t = N_init
        x_t = x_init
        
        for t in range(T):
            # Record state
            data.append({'period': t, 'N': N_t, 'x': x_t})
            
            # Exit decisions for incumbents
            if N_t > 0:
                mu_cut = self.mu_cutoff.get((N_t, x_t), self.mu)
                # Draw private exit values
                mu_draws = np.random.normal(self.mu, self.sigma_mu, N_t)
                # Firms stay if μ_it ≤ μ(N_t, x_t)
                stay = mu_draws <= mu_cut
                N_t = np.sum(stay)
            
            # Entry decision
            if N_t < self.N_max:
                gamma_cut = self.gamma_cutoff.get((N_t, x_t), self.gamma)
                # Draw private entry cost
                gamma_draw = np.random.normal(self.gamma, self.sigma_gamma)
                # Entrant enters if γ_it ≤ γ(N_t, x_t)
                if gamma_draw <= gamma_cut:
                    N_t += 1
            
            # Demand shifter transition
            x_idx = np.where(self.x_values == x_t)[0][0]
            x_idx_next = np.random.choice(len(self.x_values), 
                                         p=self.P_x[x_idx, :])
            x_t = self.x_values[x_idx_next]
        
        return pd.DataFrame(data)


def binom_pmf(k, n, p):
    """Binomial PMF: P(X = k) where X ~ Binomial(n, p)"""
    if k < 0 or k > n:
        return 0.0
    from scipy.special import comb
    return comb(n, k) * (p**k) * ((1-p)**(n-k))


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Dynamic Entry/Exit Game - Homework Solution")
    print("="*70)
    
    # Initialize model with true parameters
    game = DynamicEntryExitGame(gamma=5, sigma_gamma=np.sqrt(5), 
                                 mu=5, sigma_mu=np.sqrt(5))
    
    # Question 1: Show profit formula
    print("\n" + "="*70)
    print("Question 1: Nash Equilibrium Profit Formula")
    print("="*70)
    print("\nπ(N, x) = [(10 + x)/(N + 1)]² - 5")
    print("\nExamples:")
    for N in [1, 2, 3, 4, 5]:
        for x in [-5, 0, 5]:
            pi = game.compute_pi(N, x)
            print(f"  N={N}, x={x:2d}: π = {pi:7.3f}")
    
    # Question 4: Solve for equilibrium
    print("\n" + "="*70)
    print("Question 4: Solving for Equilibrium")
    print("="*70)
    game.solve_equilibrium(max_iter=2000, tol=1e-8, verbose=True)
    
    # Question 6: Report equilibrium values at (N=3, x=0)
    print("\n" + "="*70)
    print("Question 6: Equilibrium Values at (N=3, x=0)")
    print("="*70)
    N_query, x_query = 3, 0
    print(f"μ(3, 0)  = {game.mu_cutoff.get((N_query, x_query), np.nan):.4f}")
    print(f"γ(3, 0)  = {game.gamma_cutoff.get((N_query, x_query), np.nan):.4f}")
    print(f"V̄(3, 0)  = {game.V_bar.get((N_query, x_query), np.nan):.4f}")
    
    # V(3, 0, -2): Value for specific μ = -2
    mu_specific = -2
    pi_30 = game.compute_pi(3, 0)
    mu_cut_30 = game.mu_cutoff.get((3, 0), game.mu)
    if mu_specific > mu_cut_30:
        V_specific = mu_specific  # Exit
    else:
        # Stay: get π + continuation
        V_specific = pi_30 + (game.V_bar.get((3, 0), 0) - pi_30) / (1 - game.beta)
    print(f"V(3, 0, -2) ≈ {mu_specific:.4f} (firm exits since μ={mu_specific} > μ̄={mu_cut_30:.4f})")
    
    # Question 7: Simulate market
    print("\n" + "="*70)
    print("Question 7: Market Simulation")
    print("="*70)
    df_sim = game.simulate_market(N_init=0, x_init=0, T=10000, seed=1995)
    avg_firms = df_sim['N'].mean()
    print(f"Average number of firms over 10,000 periods: {avg_firms:.3f}")
    print(f"Distribution of firms:")
    print(df_sim['N'].value_counts().sort_index())
    
    print("\n" + "="*70)
    print("Homework solution complete!")
    print("="*70)
