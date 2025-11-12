"""
BBL Estimation - CORRECTED Implementation
Based on actual BBL authors' code from Bajari, Benkard, Levin (2007)

KEY INSIGHT: 
- Homework has you use ONE policy (estimated optimal) for both baseline and alternative
- This makes W ≈ 0, removing identification
- BBL authors use ALTERNATIVE POLICIES (perturbations) to create variation

The correct approach:
1. Estimate CCPs: d̂(N,x), ê(N,x) from data
2. For each state & alternative policy perturbation:
   - Simulate forward with OPTIMAL policy → PDV_optimal
   - Simulate forward with PERTURBED policy → PDV_alternative  
   - W = PDV_optimal - PDV_alternative
3. Average W over many alternatives
4. Match model-implied stay probability to data
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time

# True parameters
TRUE_PARAMS = {'mu': 5.0, 'sigma_mu': np.sqrt(5.0), 
               'gamma': 5.0, 'sigma_gamma': np.sqrt(5.0)}

# Model parameters
BETA = 0.9
STATES_N = [0, 1, 2, 3, 4, 5]
STATES_X = [-5, 0, 5]
X_TRANSITION = np.array([[0.6, 0.2, 0.2],
                         [0.2, 0.6, 0.2],
                         [0.2, 0.2, 0.6]])

def profit(N, x):
    """Single period profit per firm."""
    if N == 0:
        return 0.0
    return ((10 + x) / (N + 1))**2 - 5

def load_data():
    """Load simulated data."""
    data = np.load('/Users/marek/Desktop/IO/entry/game_data.npz', allow_pickle=True)
    df = data['df_sim']
    # Columns: period, N, x, entry_action
    states = df[:, 3]  # x values
    N_firms = df[:, 2]  # N values
    actions = df[:, 4]  # entry_action
    return states, N_firms, actions

class BBLEstimatorCorrected:
    """BBL Estimator using ALTERNATIVE POLICIES like authors."""
    
    def __init__(self, states, N_firms, actions, n_alternatives=100, 
                 n_sims=50, horizon=80, sd_exit=0.75):
        """
        Args:
            n_alternatives: Number of alternative policy perturbations
            n_sims: Forward simulations per alternative
            horizon: Forward simulation length
            sd_exit: Standard deviation of exit policy perturbations
        """
        self.states = states
        self.N_firms = N_firms
        self.actions = actions
        self.T = len(states)
        self.n_alternatives = n_alternatives
        self.n_sims = n_sims
        self.horizon = horizon
        self.sd_exit = sd_exit
        
        # Estimate CCPs from data (Step A)
        self.d_hat, self.e_hat = self._estimate_ccps()
        
        # Pre-generate random draws for alternatives
        self.alt_exit_draws = np.random.randn(n_alternatives) * sd_exit
        
        print(f"Initialized with {n_alternatives} alternative policies")
        print(f"Each alternative uses {n_sims} forward simulations of {horizon} periods")
        
    def _estimate_ccps(self):
        """Estimate conditional choice probabilities from data."""
        stay_counts = {}
        total_counts = {}
        entry_counts = {}
        entry_total = {}
        
        for t in range(self.T - 1):
            N, x = self.N_firms[t], self.states[t]
            state = (N, x)
            
            # Count incumbent stays
            if N > 0:
                n_incumbents = N
                n_stayed = self.N_firms[t+1]
                if self.actions[t] == 1:  # Entry occurred
                    n_stayed -= 1
                    
                if state not in stay_counts:
                    stay_counts[state] = 0
                    total_counts[state] = 0
                stay_counts[state] += n_stayed
                total_counts[state] += n_incumbents
            
            # Count entries
            if N < 5:
                if state not in entry_counts:
                    entry_counts[state] = 0
                    entry_total[state] = 0
                entry_counts[state] += self.actions[t]
                entry_total[state] += 1
        
        # Compute probabilities
        d_hat = {}
        for state in stay_counts:
            d_hat[state] = stay_counts[state] / total_counts[state] if total_counts[state] > 0 else 0.5
            
        e_hat = {}
        for state in entry_counts:
            e_hat[state] = entry_counts[state] / entry_total[state] if entry_total[state] > 0 else 0.5
            
        # Fill unobserved states with nearby values
        for N in STATES_N:
            for x in STATES_X:
                if (N, x) not in d_hat and N > 0:
                    d_hat[(N, x)] = 0.5
                if (N, x) not in e_hat and N < 5:
                    e_hat[(N, x)] = 0.5
                    
        return d_hat, e_hat
    
    def _simulate_forward_with_policy(self, N_start, x_start, exit_perturbation, 
                                     mu, sigma_mu, gamma, sigma_gamma, base_draws):
        """
        Simulate forward with perturbed exit policy.
        
        Args:
            exit_perturbation: Amount to add to exit probability for focal firm
            base_draws: Fixed random draws (N(0,1)) for consistency
        
        Returns:
            PDV of focal firm's payoffs
        """
        N, x = N_start, x_start
        pdv = 0.0
        focal_firm_alive = True
        
        for t in range(self.horizon):
            if not focal_firm_alive:
                break
                
            # Current period profit
            pi = profit(N, x)
            
            # Exit decisions
            if N > 0:
                # Focal firm exit decision (with perturbation)
                exit_draw = base_draws['exit'][t, 0]
                mu_it = mu + sigma_mu * exit_draw
                exit_prob_focal = self.d_hat.get((N, x), 0.5) + exit_perturbation
                exit_prob_focal = np.clip(exit_prob_focal, 0.01, 0.99)
                
                # Focal firm exits if draw is above threshold
                exit_threshold = norm.ppf(1 - exit_prob_focal)
                if exit_draw >= exit_threshold:
                    # Firm exits, receives exit value
                    pdv += (BETA ** t) * mu_it
                    focal_firm_alive = False
                    break
                else:
                    # Firm stays, earns profit
                    pdv += (BETA ** t) * pi
                
                # Other firms exit (using optimal policy, no perturbation)
                n_other_firms = N - 1
                if n_other_firms > 0:
                    exit_prob_others = self.d_hat.get((N, x), 0.5)
                    for i in range(1, n_other_firms + 1):
                        exit_draw_other = base_draws['exit'][t, i]
                        exit_threshold_other = norm.ppf(1 - exit_prob_others)
                        if exit_draw_other >= exit_threshold_other:
                            N -= 1
            
            # Entry decision (no perturbation)
            if N < 5:
                entry_draw = base_draws['entry'][t]
                gamma_it = gamma + sigma_gamma * entry_draw
                entry_prob = self.e_hat.get((N, x), 0.5)
                entry_threshold = norm.ppf(entry_prob)
                if entry_draw <= entry_threshold:
                    N += 1
            
            # State transition
            x_draw = np.random.uniform(0, 1)
            x_idx = np.searchsorted(np.cumsum(X_TRANSITION[STATES_X.index(x), :]), x_draw)
            x = STATES_X[x_idx]
            
        return pdv
    
    def compute_w_values(self, mu, sigma_mu, gamma, sigma_gamma):
        """
        Compute W values using alternative policy approach.
        
        W(N,x, alt) = PDV(optimal policy) - PDV(alternative policy)
        
        Returns:
            Dictionary mapping (N, x) to average W across alternatives
        """
        W_values = {}
        
        for N in STATES_N:
            if N == 0:
                continue
                
            for x in STATES_X:
                W_list = []
                
                # Loop over alternative policies
                for alt_idx in range(self.n_alternatives):
                    exit_pert = self.alt_exit_draws[alt_idx]
                    
                    # Average over multiple simulations
                    w_sims = []
                    for sim in range(self.n_sims):
                        # Generate fixed draws for this simulation
                        base_draws = {
                            'exit': np.random.randn(self.horizon, 10),
                            'entry': np.random.randn(self.horizon)
                        }
                        
                        # Simulate with optimal policy (no perturbation)
                        pdv_optimal = self._simulate_forward_with_policy(
                            N, x, 0.0, mu, sigma_mu, gamma, sigma_gamma, base_draws
                        )
                        
                        # Simulate with alternative policy (with perturbation)
                        pdv_alternative = self._simulate_forward_with_policy(
                            N, x, exit_pert, mu, sigma_mu, gamma, sigma_gamma, base_draws
                        )
                        
                        w = pdv_optimal - pdv_alternative
                        w_sims.append(w)
                    
                    W_list.append(np.mean(w_sims))
                
                W_values[(N, x)] = np.mean(W_list)
        
        return W_values
    
    def objective(self, params):
        """
        Minimum distance objective function.
        
        Match: Φ((W - μ) / σ_μ) ≈ d̂(N, x)
        
        This says: probability of staying = probability that μ_it < W
        """
        mu, log_sigma_mu, gamma, log_sigma_gamma = params
        sigma_mu = np.exp(log_sigma_mu)
        sigma_gamma = np.exp(log_sigma_gamma)
        
        # Compute W values
        W_values = self.compute_w_values(mu, sigma_mu, gamma, sigma_gamma)
        
        # Compute objective
        obj = 0.0
        for (N, x), W in W_values.items():
            # Probability of staying under model
            prob_stay_model = norm.cdf((W - mu) / sigma_mu)
            
            # Probability of staying in data
            prob_stay_data = self.d_hat[(N, x)]
            
            # Add squared error
            obj += (prob_stay_model - prob_stay_data) ** 2
        
        return obj
    
    def estimate(self, initial_guess=None):
        """Estimate parameters."""
        if initial_guess is None:
            initial_guess = [5.0, np.log(2.0), 5.0, np.log(2.0)]
        
        print("\nStarting BBL estimation (CORRECTED with alternative policies)...")
        print(f"Initial guess: μ={initial_guess[0]:.2f}, σ_μ={np.exp(initial_guess[1]):.2f}, "
              f"γ={initial_guess[2]:.2f}, σ_γ={np.exp(initial_guess[3]):.2f}")
        
        t0 = time.time()
        
        result = minimize(
            self.objective,
            initial_guess,
            method='Nelder-Mead',
            options={'maxiter': 50, 'disp': True, 'xatol': 0.1, 'fatol': 0.01}
        )
        
        elapsed = time.time() - t0
        
        mu_hat, log_sigma_mu, gamma_hat, log_sigma_gamma = result.x
        sigma_mu_hat = np.exp(log_sigma_mu)
        sigma_gamma_hat = np.exp(log_sigma_gamma)
        
        print(f"\n{'='*70}")
        print("ESTIMATION RESULTS (CORRECTED METHOD)")
        print(f"{'='*70}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"\nParameter Estimates:")
        print(f"  μ̂ = {mu_hat:.4f}  (true: {TRUE_PARAMS['mu']:.4f}, error: {100*(mu_hat-TRUE_PARAMS['mu'])/TRUE_PARAMS['mu']:.1f}%)")
        print(f"  σ̂_μ = {sigma_mu_hat:.4f}  (true: {TRUE_PARAMS['sigma_mu']:.4f}, error: {100*(sigma_mu_hat-TRUE_PARAMS['sigma_mu'])/TRUE_PARAMS['sigma_mu']:.1f}%)")
        print(f"  γ̂ = {gamma_hat:.4f}  (true: {TRUE_PARAMS['gamma']:.4f}, error: {100*(gamma_hat-TRUE_PARAMS['gamma'])/TRUE_PARAMS['gamma']:.1f}%)")
        print(f"  σ̂_γ = {sigma_gamma_hat:.4f}  (true: {TRUE_PARAMS['sigma_gamma']:.4f}, error: {100*(sigma_gamma_hat-TRUE_PARAMS['sigma_gamma'])/TRUE_PARAMS['sigma_gamma']:.1f}%)")
        print(f"\nObjective value: {result.fun:.6f}")
        print(f"Optimization success: {result.success}")
        
        return result

if __name__ == '__main__':
    print("="*70)
    print("BBL ESTIMATION - CORRECTED IMPLEMENTATION")
    print("="*70)
    print("\nThis implements the ACTUAL BBL method using alternative policies")
    print("Key difference from homework: perturb policy to create W variation\n")
    
    # Load data
    states, N_firms, actions = load_data()
    print(f"Loaded {len(states)} periods of simulated data")
    
    # Create estimator with BBL authors' settings (scaled down)
    # Authors use: 500 alternatives, 2000 sims, 80 horizon
    # We use: 100 alternatives, 50 sims, 80 horizon (for speed)
    estimator = BBLEstimatorCorrected(
        states, N_firms, actions,
        n_alternatives=100,  # Authors: 500
        n_sims=50,           # Authors: 2000  
        horizon=80,          # Authors: 80
        sd_exit=0.75         # Authors: 0.75
    )
    
    # Run estimation
    result = estimator.estimate()
