import numpy as np
from scipy.stats import norm
from scipy.special import ndtr  # Vectorized normal CDF
import matplotlib.pyplot as plt

# ============================================================================
# PARAMETERS
# ============================================================================
# Model parameters
gamma_mean = 5.0
gamma_var = 5.0
mu_mean = 5.0
mu_var = 5.0

gamma_sd = np.sqrt(gamma_var)
mu_sd = np.sqrt(mu_var)

beta = 0.9
max_N = 5

# State space
N_states = np.arange(0, max_N + 1)  # 0, 1, 2, 3, 4, 5
x_states = np.array([-5, 0, 5])

# Transition matrix for x (demand shifter)
P_x = np.array([
    [0.6, 0.2, 0.2],  # from x=-5
    [0.2, 0.6, 0.2],  # from x=0
    [0.2, 0.2, 0.6]   # from x=5
])

# ============================================================================
# PROFIT FUNCTION
# ============================================================================
def profit(N, x):
    """
    Nash equilibrium single-period profit per firm.
    π(N,x) = [(10+x)/(N+1)]^2 - 5
    """
    if N == 0:
        return 0.0  # No firms, no profit
    return ((10 + x) / (N + 1))**2 - 5

# ============================================================================
# STATE INDEXING
# ============================================================================
def state_to_index(N, x):
    """Convert (N, x) to flat index for arrays."""
    N_idx = N
    x_idx = np.where(x_states == x)[0][0]
    return N_idx * len(x_states) + x_idx

def index_to_state(idx):
    """Convert flat index back to (N, x)."""
    N_idx = idx // len(x_states)
    x_idx = idx % len(x_states)
    return N_states[N_idx], x_states[x_idx]

# Total number of states
n_states = len(N_states) * len(x_states)

# ============================================================================
# TRANSITION MATRICES
# ============================================================================
def compute_transition_incumbent(mu_cutoffs, gamma_cutoffs):
    """
    Compute Pr(N_{t+1} | N_t, x_t, d_it=1) for all states.
    
    When an incumbent stays (d_it=1):
    - The focal incumbent is staying (deterministic)
    - Other N_t - 1 incumbents stay with probability p(stay|N_t,x_t)
    - One potential entrant enters with probability p(enter|N_t,x_t)
    
    Returns: Dictionary indexed by (N_t, x_t) -> array of length max_N+1
    """
    transitions = {}
    
    for N in N_states:
        for x in x_states:
            if N == 0:
                # No incumbents, this transition is not defined
                transitions[(N, x)] = np.zeros(max_N + 1)
                continue
            
            # Probability that an incumbent stays
            p_stay = ndtr((mu_cutoffs[(N, x)] - mu_mean) / mu_sd)
            
            # Probability that entrant enters
            p_enter = ndtr((gamma_cutoffs[(N, x)] - gamma_mean) / gamma_sd) if N < max_N else 0.0
            
            # Distribution over N_{t+1}
            prob_N_next = np.zeros(max_N + 1)
            
            # Focal incumbent is staying, so we start with 1 firm
            # Other N-1 incumbents can stay or exit
            # Entrant can enter or not
            
            for n_other_stay in range(N):  # 0 to N-1 other incumbents stay
                # Binomial probability for other incumbents
                if N == 1:
                    # Only focal firm, no other incumbents
                    prob_other = 1.0 if n_other_stay == 0 else 0.0
                else:
                    prob_other = (
                        np.math.comb(N - 1, n_other_stay) *
                        (p_stay ** n_other_stay) *
                        ((1 - p_stay) ** (N - 1 - n_other_stay))
                    )
                
                # Total firms if no entry: 1 (focal) + n_other_stay
                N_no_entry = 1 + n_other_stay
                N_with_entry = N_no_entry + 1
                
                # Add probability of no entry
                if N_no_entry <= max_N:
                    prob_N_next[N_no_entry] += prob_other * (1 - p_enter)
                
                # Add probability of entry
                if N_with_entry <= max_N:
                    prob_N_next[N_with_entry] += prob_other * p_enter
            
            transitions[(N, x)] = prob_N_next
    
    return transitions

def compute_transition_entrant(mu_cutoffs, gamma_cutoffs):
    """
    Compute Pr(N_{t+1} | N_t, x_t, e_it=1) for all states.
    
    When an entrant enters (e_it=1):
    - All N_t current incumbents stay with probability p(stay|N_t,x_t)
    - The entrant becomes an incumbent next period (deterministic)
    
    Returns: Dictionary indexed by (N_t, x_t) -> array of length max_N+1
    """
    transitions = {}
    
    for N in N_states:
        for x in x_states:
            # Probability that an incumbent stays
            if N > 0:
                p_stay = ndtr((mu_cutoffs[(N, x)] - mu_mean) / mu_sd)
            else:
                p_stay = 0.0  # No incumbents
            
            # Distribution over N_{t+1}
            prob_N_next = np.zeros(max_N + 1)
            
            # Entrant is entering, so becomes incumbent next period
            # Current N incumbents can stay or exit
            
            for n_stay in range(N + 1):  # 0 to N incumbents stay
                # Binomial probability
                if N == 0:
                    prob_stay = 1.0 if n_stay == 0 else 0.0
                else:
                    prob_stay = (
                        np.math.comb(N, n_stay) *
                        (p_stay ** n_stay) *
                        ((1 - p_stay) ** (N - n_stay))
                    )
                
                # Total firms next period: n_stay + 1 (entrant)
                N_next = n_stay + 1
                
                if N_next <= max_N:
                    prob_N_next[N_next] += prob_stay
            
            transitions[(N, x)] = prob_N_next
    
    return transitions

# ============================================================================
# CONTINUATION VALUES
# ============================================================================
def compute_continuation_values(V_bar, transitions_incumbent, transitions_entrant):
    """
    Compute Ψ_1(N_t, x_t) and Ψ_2(N_t, x_t).
    
    Ψ_1(N_t, x_t) = β * Σ_{N_{t+1}, x_{t+1}} V̄(N_{t+1}, x_{t+1}) * Pr(x_{t+1}|x_t) * Pr(N_{t+1}|N_t, x_t, d_it=1)
    Ψ_2(N_t, x_t) = β * Σ_{N_{t+1}, x_{t+1}} V̄(N_{t+1}, x_{t+1}) * Pr(x_{t+1}|x_t) * Pr(N_{t+1}|N_t, x_t, e_it=1)
    """
    Psi_1 = {}
    Psi_2 = {}
    
    for N in N_states:
        for x_idx, x in enumerate(x_states):
            
            if N == 0:
                # No incumbents, Ψ_1 not defined
                Psi_1[(N, x)] = 0.0
            else:
                value = 0.0
                trans_incumbent = transitions_incumbent[(N, x)]
                
                for x_next_idx, x_next in enumerate(x_states):
                    for N_next in N_states:
                        if N_next > 0:  # V̄ only defined for N > 0
                            value += (
                                V_bar[(N_next, x_next)] *
                                P_x[x_idx, x_next_idx] *
                                trans_incumbent[N_next]
                            )
                
                Psi_1[(N, x)] = beta * value
            
            if N == max_N:
                # No entry possible
                Psi_2[(N, x)] = 0.0
            else:
                value = 0.0
                trans_entrant = transitions_entrant[(N, x)]
                
                for x_next_idx, x_next in enumerate(x_states):
                    for N_next in N_states:
                        if N_next > 0:  # V̄ only defined for N > 0
                            value += (
                                V_bar[(N_next, x_next)] *
                                P_x[x_idx, x_next_idx] *
                                trans_entrant[N_next]
                            )
                
                Psi_2[(N, x)] = beta * value
    
    return Psi_1, Psi_2

# ============================================================================
# CUTOFF UPDATES
# ============================================================================
def update_cutoffs(Psi_1, Psi_2):
    """
    Update cutoffs using:
    μ'(N_t, x_t) = π(N_t, x_t) + Ψ_1(N_t, x_t)
    γ'(N_t, x_t) = Ψ_2(N_t, x_t)
    """
    mu_new = {}
    gamma_new = {}
    
    for N in N_states:
        for x in x_states:
            if N > 0:
                mu_new[(N, x)] = profit(N, x) + Psi_1[(N, x)]
            
            if N < max_N:
                gamma_new[(N, x)] = Psi_2[(N, x)]
    
    return mu_new, gamma_new

# ============================================================================
# VALUE FUNCTION UPDATE
# ============================================================================
def update_value_function(Psi_1):
    """
    Update V̄(N_t, x_t) using the truncated normal formula.
    
    V̄(N_t, x_t) = Φ(z) * [π(N_t,x_t) + Ψ_1(N_t,x_t)] + [1 - Φ(z)] * [μ + σ_μ * φ(z)/(1-Φ(z))]
    
    where z = (π(N_t,x_t) + Ψ_1(N_t,x_t) - μ) / σ_μ
    """
    V_bar_new = {}
    
    for N in N_states:
        for x in x_states:
            if N == 0:
                # V̄ not defined for N=0
                continue
            
            cutoff = profit(N, x) + Psi_1[(N, x)]
            z = (cutoff - mu_mean) / mu_sd
            
            Phi_z = ndtr(z)
            phi_z = norm.pdf(z)
            
            # Probability of staying times stay value
            stay_term = Phi_z * cutoff
            
            # Probability of exiting times expected exit value (truncated mean)
            if Phi_z < 1.0:
                inverse_mills = phi_z / (1 - Phi_z)
                exit_term = (1 - Phi_z) * (mu_mean + mu_sd * inverse_mills)
            else:
                exit_term = 0.0
            
            V_bar_new[(N, x)] = stay_term + exit_term
    
    return V_bar_new

# ============================================================================
# MAIN ITERATION
# ============================================================================
def solve_equilibrium(max_iter=1000, tol=1e-6, verbose=True):
    """
    Solve for the equilibrium using the iterative algorithm.
    """
    # Initial guess: use static profits for cutoffs, zero for value functions
    mu_cutoffs = {}
    gamma_cutoffs = {}
    V_bar = {}
    
    for N in N_states:
        for x in x_states:
            if N > 0:
                mu_cutoffs[(N, x)] = profit(N, x)
                V_bar[(N, x)] = profit(N, x) / (1 - beta)  # Myopic value
            if N < max_N:
                gamma_cutoffs[(N, x)] = 0.0
    
    if verbose:
        print("Starting equilibrium computation...")
        print(f"Parameters: γ={gamma_mean}, σ²_γ={gamma_var}, μ={mu_mean}, σ²_μ={mu_var}, β={beta}")
        print()
    
    for iteration in range(max_iter):
        # Step 2: Compute transition matrices
        trans_incumbent = compute_transition_incumbent(mu_cutoffs, gamma_cutoffs)
        trans_entrant = compute_transition_entrant(mu_cutoffs, gamma_cutoffs)
        
        # Step 3: Compute continuation values
        Psi_1, Psi_2 = compute_continuation_values(V_bar, trans_incumbent, trans_entrant)
        
        # Step 4: Update cutoffs
        mu_new, gamma_new = update_cutoffs(Psi_1, Psi_2)
        
        # Step 5: Update value function
        V_bar_new = update_value_function(Psi_1)
        
        # Check convergence
        max_diff_mu = max(abs(mu_new[(N, x)] - mu_cutoffs[(N, x)]) 
                          for N in N_states for x in x_states if N > 0)
        max_diff_gamma = max(abs(gamma_new[(N, x)] - gamma_cutoffs[(N, x)]) 
                             for N in N_states for x in x_states if N < max_N)
        max_diff_V = max(abs(V_bar_new[(N, x)] - V_bar[(N, x)]) 
                         for N in N_states for x in x_states if N > 0)
        
        max_diff = max(max_diff_mu, max_diff_gamma, max_diff_V)
        
        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iteration {iteration:4d}: max_diff = {max_diff:.8f}")
        
        if max_diff < tol:
            if verbose:
                print(f"\nConverged after {iteration} iterations!")
                print(f"Final max difference: {max_diff:.2e}")
            break
        
        # Update for next iteration
        mu_cutoffs = mu_new
        gamma_cutoffs = gamma_new
        V_bar = V_bar_new
    else:
        print(f"Warning: Did not converge after {max_iter} iterations. Max diff: {max_diff:.2e}")
    
    return mu_cutoffs, gamma_cutoffs, V_bar

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
def display_results(mu_cutoffs, gamma_cutoffs, V_bar):
    """Display equilibrium values in a nice format."""
    print("\n" + "="*70)
    print("EQUILIBRIUM CUTOFFS AND VALUE FUNCTIONS")
    print("="*70)
    
    print("\nμ(N_t, x_t) - Exit cutoffs for incumbents:")
    print("-" * 50)
    print(f"{'State (N, x)':<15} {'μ(N,x)':<12} {'π(N,x)':<12} {'Stay Prob':<12}")
    print("-" * 50)
    for N in N_states:
        if N > 0:
            for x in x_states:
                stay_prob = ndtr((mu_cutoffs[(N, x)] - mu_mean) / mu_sd)
                print(f"({N:2d}, {x:3d}){'':<7} {mu_cutoffs[(N, x)]:>10.4f}  "
                      f"{profit(N, x):>10.4f}  {stay_prob:>10.4f}")
    
    print("\nγ(N_t, x_t) - Entry cutoffs for potential entrants:")
    print("-" * 50)
    print(f"{'State (N, x)':<15} {'γ(N,x)':<12} {'Entry Prob':<12}")
    print("-" * 50)
    for N in N_states:
        if N < max_N:
            for x in x_states:
                entry_prob = ndtr((gamma_cutoffs[(N, x)] - gamma_mean) / gamma_sd)
                print(f"({N:2d}, {x:3d}){'':<7} {gamma_cutoffs[(N, x)]:>10.4f}  {entry_prob:>10.4f}")
    
    print("\nV̄(N_t, x_t) - Ex-ante value functions:")
    print("-" * 50)
    print(f"{'State (N, x)':<15} {'V̄(N,x)':<12}")
    print("-" * 50)
    for N in N_states:
        if N > 0:
            for x in x_states:
                print(f"({N:2d}, {x:3d}){'':<7} {V_bar[(N, x)]:>10.4f}")
    
    # Answer Problem 6 specifically
    print("\n" + "="*70)
    print("PROBLEM 6: Values at state (3, 0)")
    print("="*70)
    N, x = 3, 0
    print(f"μ(3, 0)  = {mu_cutoffs[(N, x)]:.6f}")
    print(f"γ(3, 0)  = {gamma_cutoffs[(N, x)]:.6f}")
    print(f"V̄(3, 0)  = {V_bar[(N, x)]:.6f}")
    
    # Compute V(3, 0, -2)
    mu_val = -2
    if mu_val > mu_cutoffs[(N, x)]:
        # Firm exits
        V_value = mu_val
        decision = "EXIT"
    else:
        # Firm stays
        V_value = profit(N, x) + beta * sum(
            V_bar.get((N_next, x_next), 0) * P_x[np.where(x_states == x)[0][0], x_next_idx]
            * compute_transition_incumbent(mu_cutoffs, gamma_cutoffs)[(N, x)][N_next]
            for x_next_idx, x_next in enumerate(x_states)
            for N_next in N_states if N_next > 0
        )
        decision = "STAY"
    
    print(f"V(3, 0, -2) = {V_value:.6f}  (Firm will {decision})")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    mu_cutoffs, gamma_cutoffs, V_bar = solve_equilibrium()
    display_results(mu_cutoffs, gamma_cutoffs, V_bar)
