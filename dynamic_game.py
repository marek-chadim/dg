"""
Dynamic Entry/Exit Game - Complete Implementation
Economics 600a - Yale University, Fall 2025
Marek Chadim

Based on:
- Ericson & Pakes (1995) framework
- Bajari, Benkard, Levin (2007) two-step estimation
- Hotz & Miller (1993) CCP approach
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import comb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(2007)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_pi(N: int, x: float) -> float:
    """
    Compute Nash equilibrium single-period profit per firm.
    
    Formula: π(N, x) = [(10 + x)/(N + 1)]² - 5
    
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


def binom_pmf(k: int, n: int, p: float) -> float:
    """Binomial PMF: P(X = k) where X ~ Binomial(n, p)"""
    if k < 0 or k > n:
        return 0.0
    if n == 0:
        return 1.0 if k == 0 else 0.0
    return comb(n, k, exact=True) * (p**k) * ((1-p)**(n-k))


# ==============================================================================
# Dynamic Entry/Exit Game Class
# ==============================================================================

class DynamicEntryExitGame:
    """
    Solver for dynamic entry/exit game with Cournot competition.
    
    Based on:
    - Ericson & Pakes (1995) framework
    - BBL (2007) two-step estimation
    - Hotz & Miller (1993) CCP approach
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
        """Compute Nash equilibrium single-period profit per firm."""
        return compute_pi(N, x)
    
    def compute_transition_prob(self, N: int, x: float, decision: str) -> np.ndarray:
        """
        Compute transition probabilities for number of firms.
        
        Parameters:
        -----------
        N : int
            Current number of firms
        x : float
            Current demand shifter
        decision : str
            'd' for incumbent stays (d_it=1), 'e' for entrant enters (e_it=1)
            
        Returns:
        --------
        np.ndarray : Probability distribution over N_{t+1} in {0,1,2,3,4,5}
        """
        prob = np.zeros(self.N_max + 1)
        
        if decision == 'd':  # Incumbent stays (conditioning on d_it=1)
            if N == 0:
                prob[0] = 1.0
                return prob
                
            # Probability each OTHER incumbent stays
            mu_cut = self.mu_cutoff.get((N, x), self.mu)
            p_stay = norm.cdf((mu_cut - self.mu) / self.sigma_mu)
            
            # Distribution over number of OTHER firms that stay
            for n_others_stay in range(N):
                prob_n_others = binom_pmf(n_others_stay, N - 1, p_stay)
                n_stay_total = n_others_stay + 1  # Include the conditioned firm
                
                # Add potential entrant if room
                if n_stay_total < self.N_max:
                    gamma_cut = self.gamma_cutoff.get((n_stay_total, x), self.gamma)
                    p_enter = norm.cdf((gamma_cut - self.gamma) / self.sigma_gamma)
                    prob[n_stay_total] += prob_n_others * (1 - p_enter)
                    prob[n_stay_total + 1] += prob_n_others * p_enter
                else:
                    prob[n_stay_total] += prob_n_others
                    
        else:  # Entrant enters (e_it = 1)
            if N >= self.N_max:
                prob[N] = 1.0
                return prob
                
            # Probability each incumbent stays
            if N > 0:
                mu_cut = self.mu_cutoff.get((N, x), self.mu)
                p_stay = norm.cdf((mu_cut - self.mu) / self.sigma_mu)
            else:
                p_stay = 0
            
            # Distribution over staying incumbents
            for n_stay in range(N + 1):
                prob_n_stay = binom_pmf(n_stay, N, p_stay)
                # Add the entrant (enters next period)
                n_next = min(n_stay + 1, self.N_max)
                prob[n_next] += prob_n_stay
                    
        return prob

    def solve_equilibrium(self, max_iter=1000, tol=1e-8, verbose=False, damping=0.5):
        """Solve MPE using value function iteration with damping for stability"""
        # Initialize with reasonable values
        for (N, x) in self.states:
            if N > 0:
                pi = self.compute_pi(N, x)
                self.mu_cutoff[(N, x)] = max(pi, 0) + self.mu
                self.V_bar[(N, x)] = max(pi / (1 - self.beta), self.mu)
            if N < self.N_max:
                self.gamma_cutoff[(N, x)] = self.gamma
        
        for it in range(max_iter):
            mu_old = self.mu_cutoff.copy()
            gamma_old = self.gamma_cutoff.copy()
            V_old = self.V_bar.copy()
            Psi_1, Psi_2 = {}, {}
            
            # Compute continuation values
            for (N, x) in self.states:
                x_idx = np.where(self.x_values == x)[0][0]
                if N > 0:
                    # Compute transition ONCE (doesn't depend on x_next)
                    trans = self.compute_transition_prob(N, x, 'd')
                    val = 0.0
                    for x_idx_next, x_next in enumerate(self.x_values):
                        for N_next in range(self.N_max + 1):
                            if N_next > 0 and trans[N_next] > 1e-10:
                                val += self.beta * self.P_x[x_idx, x_idx_next] * trans[N_next] * V_old.get((N_next, x_next), 0)
                    Psi_1[(N, x)] = val
                
                if N < self.N_max:
                    trans = self.compute_transition_prob(N, x, 'e')
                    val = 0.0
                    for x_idx_next, x_next in enumerate(self.x_values):
                        for N_next in range(self.N_max + 1):
                            if N_next > 0 and trans[N_next] > 1e-10:
                                val += self.beta * self.P_x[x_idx, x_idx_next] * trans[N_next] * V_old.get((N_next, x_next), 0)
                    Psi_2[(N, x)] = val
            
            # Update cutoffs with damping
            for (N, x) in self.states:
                if N > 0:
                    mu_new = self.compute_pi(N, x) + Psi_1.get((N, x), 0)
                    self.mu_cutoff[(N, x)] = damping * mu_new + (1 - damping) * mu_old[(N, x)]
                if N < self.N_max:
                    gamma_new = Psi_2.get((N, x), 0)
                    self.gamma_cutoff[(N, x)] = damping * gamma_new + (1 - damping) * gamma_old[(N, x)]
            
            # Update V̄ using truncated normal
            for (N, x) in self.states:
                if N > 0:
                    thresh = self.mu_cutoff[(N, x)]
                    z = (thresh - self.mu) / self.sigma_mu
                    prob_stay = norm.cdf(z)
                    
                    if prob_stay < 0.9999:
                        mills = norm.pdf(z) / (1 - norm.cdf(z))
                        E_mu_exit = self.mu + self.sigma_mu * mills
                    else:
                        E_mu_exit = thresh
                    
                    thresh = self.mu_cutoff[(N, x)]
                    V_new = prob_stay * thresh + (1 - prob_stay) * E_mu_exit
                    self.V_bar[(N, x)] = damping * V_new + (1 - damping) * V_old[(N, x)]
            
            # Check convergence
            diff = max(
                max(abs(self.mu_cutoff[k] - mu_old[k]) for k in mu_old),
                max(abs(self.gamma_cutoff[k] - gamma_old[k]) for k in gamma_old),
                max(abs(self.V_bar[k] - V_old[k]) for k in V_old)
            )
            
            if verbose and it % 50 == 0:
                print(f"  Iter {it}: diff={diff:.6e}")
            if diff < tol:
                if verbose:
                    print(f"  Converged in {it+1} iterations, final diff: {diff:.6e}")
                return True
        
        if verbose:
            print(f"  Max iterations reached. Final diff: {diff:.6e}")
        return diff < tol * 100

    def simulate_market(self, N_init=0, x_init=0, T=10000, seed=None):
        """
        Simulate market dynamics.
        
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
        pd.DataFrame : Simulated data with columns ['period', 'N', 'x', 'entry', 'exits']
        """
        if seed is not None:
            np.random.seed(seed)
        
        data = []
        N_t = N_init
        x_t = x_init
        
        for t in range(T):
            N_initial = N_t
            
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


# ==============================================================================
# BBL Estimator
# ==============================================================================

def estimate_ccps(df_sim, smooth=True, epsilon=0.01):
    """
    Estimate conditional choice probabilities from simulated data.
    
    Parameters:
    -----------
    df_sim : pd.DataFrame
        Simulated market data
    smooth : bool
        If True, clip CCPs to [epsilon, 1-epsilon] to avoid corner solutions
    epsilon : float
        Smoothing parameter
    
    Returns:
    --------
    d_hat : dict
        Estimated probability that incumbent stays, by state (N, x)
    e_hat : dict
        Estimated probability that entrant enters, by state (N, x)
    """
    d_hat = {}
    e_hat = {}
    
    # Estimate d_hat(N, x): probability incumbent stays
    for N in range(1, 6):
        for x in [-5, 0, 5]:
            mask = (df_sim['N_initial'] == N) & (df_sim['x'] == x)
            
            if mask.sum() > 0:
                total_incumbent_periods = mask.sum() * N
                total_exits = df_sim.loc[mask, 'exits'].sum()
                total_stays = total_incumbent_periods - total_exits
                
                prob = total_stays / total_incumbent_periods
                
                if smooth:
                    prob = np.clip(prob, epsilon, 1 - epsilon)
                
                d_hat[(N, x)] = prob
            else:
                d_hat[(N, x)] = 0.5
    
    # Estimate e_hat(N, x): probability entrant enters
    for N in range(0, 5):
        for x in [-5, 0, 5]:
            mask = (df_sim['N'] == N) & (df_sim['x'] == x)
            
            if mask.sum() > 0:
                prob = df_sim.loc[mask, 'entry'].mean()
                
                if smooth:
                    prob = np.clip(prob, epsilon, 1 - epsilon)
                
                e_hat[(N, x)] = prob
            else:
                e_hat[(N, x)] = 0.5
    
    return d_hat, e_hat


class BBLEstimator:
    """BBL estimation for dynamic entry/exit game."""
    
    def __init__(self, d_hat, e_hat, x_values, P_x, N_max=5, beta=0.9):
        """
        Initialize BBL estimator.
        
        Parameters:
        -----------
        d_hat : dict
            Estimated stay probabilities
        e_hat : dict
            Estimated entry probabilities
        x_values : np.ndarray
            Possible demand shifter values
        P_x : np.ndarray
            Transition matrix for demand shifter
        N_max : int
            Maximum number of firms
        beta : float
            Discount factor
        """
        self.d_hat = d_hat
        self.e_hat = e_hat
        self.x_values = x_values
        self.P_x = P_x
        self.N_max = N_max
        self.beta = beta
        
        # Pre-generate fixed random draws for simulation
        self.setup_fixed_draws(n_sims=50, max_periods=100)
    
    def setup_fixed_draws(self, n_sims=50, max_periods=100):
        """Pre-generate fixed random draws (critical for estimation stability)."""
        np.random.seed(2007)
        
        # Fixed N(0,1) draws for mu shocks
        self.mu_draws_std = np.random.randn(15, n_sims, max_periods, self.N_max)
        
        # Fixed N(0,1) draws for gamma shocks
        self.gamma_draws_std = np.random.randn(15, n_sims, max_periods)
        
        # Fixed draws for x_t transitions
        self.x_transition_draws = np.random.rand(15, n_sims, max_periods)
    
    def compute_pi(self, N, x):
        """Compute per-firm profit."""
        return compute_pi(N, x)
    
    def forward_simulate_incumbent(self, N_init, x_init, mu, sigma_mu, 
                                   gamma, sigma_gamma, state_idx, sim_idx):
        """Forward simulate PDV for one incumbent conditional on staying."""
        max_periods = 100
        pdv = 0.0
        
        N_t = N_init
        x_t = x_init
        x_idx = np.where(self.x_values == x_t)[0][0]
        
        for t in range(max_periods):
            pi_t = self.compute_pi(N_t, x_t)
            pdv += (self.beta ** t) * pi_t
            
            # Simulate exit decisions for OTHER incumbents
            n_others = N_t - 1
            n_stay_others = 0
            
            if n_others > 0:
                for i in range(n_others):
                    mu_it = mu + sigma_mu * self.mu_draws_std[state_idx, sim_idx, t, i]
                    prob_mu = norm.cdf((mu_it - mu) / sigma_mu)
                    
                    if prob_mu <= self.d_hat.get((N_t, x_t), 0.5):
                        n_stay_others += 1
            
            N_t = n_stay_others + 1
            
            # Simulate Firm 1's exit decision for NEXT period
            if N_t > 0:
                mu_firm1 = mu + sigma_mu * self.mu_draws_std[state_idx, sim_idx, t, n_others]
                prob_mu_firm1 = norm.cdf((mu_firm1 - mu) / sigma_mu)
                
                if prob_mu_firm1 > self.d_hat.get((N_t, x_t), 0.5):
                    pdv += (self.beta ** (t + 1)) * mu_firm1
                    break
            
            # Simulate entry decision if room
            if N_t < self.N_max:
                gamma_it = gamma + sigma_gamma * self.gamma_draws_std[state_idx, sim_idx, t]
                prob_gamma = norm.cdf((gamma_it - gamma) / sigma_gamma)
                
                if prob_gamma <= self.e_hat.get((N_t, x_t), 0.5):
                    N_t += 1
            
            # Transition demand shifter
            u = self.x_transition_draws[state_idx, sim_idx, t]
            cumprob = np.cumsum(self.P_x[x_idx, :])
            x_idx = np.searchsorted(cumprob, u)
            x_t = self.x_values[x_idx]
            
            if N_t == 0:
                break
        
        return pdv
    
    def forward_simulate_entrant(self, N_init, x_init, mu, sigma_mu,
                                gamma, sigma_gamma, state_idx, sim_idx):
        """Forward simulate PDV for potential entrant entering (net of entry cost)."""
        max_periods = 100
        pdv = 0.0
        
        N_t = N_init
        x_t = x_init
        x_idx = np.where(self.x_values == x_t)[0][0]
        
        # Simulate exit decisions for current incumbents
        n_stay = 0
        if N_t > 0:
            for i in range(N_t):
                mu_it = mu + sigma_mu * self.mu_draws_std[state_idx, sim_idx, 0, i]
                prob_mu = norm.cdf((mu_it - mu) / sigma_mu)
                
                if prob_mu <= self.d_hat.get((N_t, x_t), 0.5):
                    n_stay += 1
        
        N_t = n_stay + 1
        
        # Transition demand shifter
        u = self.x_transition_draws[state_idx, sim_idx, 0]
        cumprob = np.cumsum(self.P_x[x_idx, :])
        x_idx = np.searchsorted(cumprob, u)
        x_t = self.x_values[x_idx]
        
        # Now simulate as incumbent
        for t in range(1, max_periods):
            pi_t = self.compute_pi(N_t, x_t)
            pdv += (self.beta ** t) * pi_t
            
            # Exit decisions for others
            n_others = N_t - 1
            n_stay_others = 0
            
            if n_others > 0:
                for i in range(n_others):
                    mu_it = mu + sigma_mu * self.mu_draws_std[state_idx, sim_idx, t, i]
                    prob_mu = norm.cdf((mu_it - mu) / sigma_mu)
                    
                    if prob_mu <= self.d_hat.get((N_t, x_t), 0.5):
                        n_stay_others += 1
            
            N_t = n_stay_others + 1
            
            # Our entrant's exit decision for next period
            if N_t > 0:
                mu_entrant = mu + sigma_mu * self.mu_draws_std[state_idx, sim_idx, t, n_others]
                prob_mu_entrant = norm.cdf((mu_entrant - mu) / sigma_mu)
                
                if prob_mu_entrant > self.d_hat.get((N_t, x_t), 0.5):
                    pdv += (self.beta ** (t + 1)) * mu_entrant
                    break
            
            # Entry decision
            if N_t < self.N_max:
                gamma_it = gamma + sigma_gamma * self.gamma_draws_std[state_idx, sim_idx, t]
                prob_gamma = norm.cdf((gamma_it - gamma) / sigma_gamma)
                
                if prob_gamma <= self.e_hat.get((N_t, x_t), 0.5):
                    N_t += 1
            
            # Transition x
            u = self.x_transition_draws[state_idx, sim_idx, t]
            cumprob = np.cumsum(self.P_x[x_idx, :])
            x_idx = np.searchsorted(cumprob, u)
            x_t = self.x_values[x_idx]
            
            if N_t == 0:
                break
        
        return pdv
    
    def compute_lambda_values(self, mu, sigma_mu, gamma, sigma_gamma, n_sims=50):
        """Compute Lambda values by averaging forward simulations."""
        Lambda = {}
        Lambda_E = {}
        
        state_list = [(N, x) for N in range(1, 6) for x in self.x_values]
        
        # Compute Lambda for incumbents
        for state_idx, (N, x) in enumerate(state_list):
            pdvs = []
            for sim_idx in range(n_sims):
                pdv = self.forward_simulate_incumbent(
                    N, x, mu, sigma_mu, gamma, sigma_gamma, state_idx, sim_idx
                )
                pdvs.append(pdv)
            Lambda[(N, x)] = np.mean(pdvs)
        
        # Compute Lambda^E for entrants
        state_list_entry = [(N, x) for N in range(0, 5) for x in self.x_values]
        
        for state_idx, (N, x) in enumerate(state_list_entry):
            pdvs = []
            for sim_idx in range(n_sims):
                pdv = self.forward_simulate_entrant(
                    N, x, mu, sigma_mu, gamma, sigma_gamma, state_idx, sim_idx
                )
                pdvs.append(pdv)
            Lambda_E[(N, x)] = np.mean(pdvs)
        
        return Lambda, Lambda_E


def objective_function(params, estimator, d_hat, e_hat):
    """Minimum distance objective function."""
    mu, sigma_mu, gamma, sigma_gamma = params
    
    if sigma_mu <= 0 or sigma_gamma <= 0:
        return 1e10
    
    Lambda, Lambda_E = estimator.compute_lambda_values(
        mu, sigma_mu, gamma, sigma_gamma, n_sims=50
    )
    
    obj = 0.0
    
    # Incumbent stay probabilities
    for N in range(1, 6):
        for x in estimator.x_values:
            Lambda_val = Lambda[(N, x)]
            pred_prob = norm.cdf((Lambda_val - mu) / sigma_mu)
            actual_prob = d_hat[(N, x)]
            obj += (pred_prob - actual_prob) ** 2
    
    # Entry probabilities
    for N in range(0, 5):
        for x in estimator.x_values:
            Lambda_E_val = Lambda_E[(N, x)]
            pred_prob = norm.cdf((Lambda_E_val - gamma) / sigma_gamma)
            actual_prob = e_hat[(N, x)]
            obj += (pred_prob - actual_prob) ** 2
    
    return obj


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Dynamic Entry/Exit Game - Complete Analysis")
    print("="*70)
    
    # Initialize game
    print("\nInitializing game with true parameters...")
    game = DynamicEntryExitGame()
    
    # Solve equilibrium
    print("\nSolving equilibrium...")
    game.solve_equilibrium(verbose=True)
    
    print(f"\nEquilibrium values at (N=3, x=0):")
    print(f"  μ(3, 0)  = {game.mu_cutoff.get((3, 0), np.nan):.6f}")
    print(f"  γ(3, 0)  = {game.gamma_cutoff.get((3, 0), np.nan):.6f}")
    print(f"  V̄(3, 0)  = {game.V_bar.get((3, 0), np.nan):.6f}")
    
    # Simulate market
    print("\nSimulating market (T=10,000 periods)...")
    df_sim = game.simulate_market(N_init=0, x_init=0, T=10000, seed=1995)
    
    avg_firms = df_sim['N'].mean()
    print(f"  Average number of firms: {avg_firms:.3f}")
    
    # BBL Estimation
    print("\n" + "="*70)
    print("BBL Estimation")
    print("="*70)
    
    print("\nEstimating CCPs from data...")
    d_hat, e_hat = estimate_ccps(df_sim, smooth=True, epsilon=0.01)
    
    print("\nInitializing BBL estimator...")
    estimator = BBLEstimator(d_hat, e_hat, game.x_values, game.P_x)
    
    print("\nRunning optimization (this may take several minutes)...")
    params_init = np.array([5.0, np.sqrt(5), 5.0, np.sqrt(5)])
    bounds = [(0, 20), (0.5, 5), (0, 20), (0.5, 5)]
    
    result = minimize(
        objective_function,
        params_init,
        args=(estimator, d_hat, e_hat),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50, 'disp': True, 'ftol': 1e-6}
    )
    
    mu_hat, sigma_mu_hat, gamma_hat, sigma_gamma_hat = result.x
    
    print("\n" + "="*70)
    print("Estimation Results")
    print("="*70)
    print("\nTrue Parameters:")
    print(f"  μ        = 5.0000")
    print(f"  σ_μ      = {np.sqrt(5):.4f}")
    print(f"  γ        = 5.0000")
    print(f"  σ_γ      = {np.sqrt(5):.4f}")
    
    print("\nEstimated Parameters:")
    print(f"  μ̂        = {mu_hat:.4f}")
    print(f"  σ̂_μ      = {sigma_mu_hat:.4f}")
    print(f"  γ̂        = {gamma_hat:.4f}")
    print(f"  σ̂_γ      = {sigma_gamma_hat:.4f}")
    
    print("\nEstimation Errors:")
    print(f"  Δμ       = {mu_hat - 5:.4f}  ({100*(mu_hat - 5)/5:.1f}%)")
    print(f"  Δσ_μ     = {sigma_mu_hat - np.sqrt(5):.4f}  ({100*(sigma_mu_hat - np.sqrt(5))/np.sqrt(5):.1f}%)")
    print(f"  Δγ       = {gamma_hat - 5:.4f}  ({100*(gamma_hat - 5)/5:.1f}%)")
    print(f"  Δσ_γ     = {sigma_gamma_hat - np.sqrt(5):.4f}  ({100*(sigma_gamma_hat - np.sqrt(5))/np.sqrt(5):.1f}%)")
    
    print(f"\nObjective function value: {result.fun:.6f}")
    print(f"Optimization success: {result.success}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
