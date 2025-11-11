import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parameters
gamma_mean = 5
gamma_var = 5
mu_mean = 5
mu_var = 5
beta = 0.9

# State space
N_states = np.arange(0, 6)  # 0 to 5 firms
x_states = np.array([-5, 0, 5])  # demand shifters

# Transition matrix for x
P_x = np.array([
    [0.6, 0.2, 0.2],
    [0.2, 0.6, 0.2],
    [0.2, 0.2, 0.6]
])

def profit(N, x):
    """Compute per-firm Cournot profit: π(N,x) = [(10+x)/(N+1)]^2 - 5"""
    if N == 0:
        return 0
    return ((10 + x) / (N + 1))**2 - 5

def compute_transition_matrices(mu_cutoffs, gamma_cutoffs, mu_mean, mu_var, gamma_mean, gamma_var):
    """
    Compute Pr(N_{t+1}|N_t, x_t, d_it=1) and Pr(N_{t+1}|N_t, x_t, e_it=1)
    
    Returns:
        P_stay: dict with keys (N_t, x_idx) -> vector of length 6 for N_{t+1} probs
        P_enter: dict with keys (N_t, x_idx) -> vector of length 6 for N_{t+1} probs
    """
    P_stay = {}
    P_enter = {}
    
    for N_t in range(1, 6):  # Incumbents exist for N_t >= 1
        for x_idx, x_t in enumerate(x_states):
            # Probability each incumbent stays (exits)
            p_stay = norm.cdf((mu_cutoffs[N_t, x_idx] - mu_mean) / np.sqrt(mu_var))
            p_exit = 1 - p_stay
            
            # Probability entrant enters
            p_entry = norm.cdf((gamma_cutoffs[N_t, x_idx] - gamma_mean) / np.sqrt(gamma_var))
            
            # For d_it=1 (one incumbent staying): N_t-1 others face exit, 1 potential entrant
            # N_{t+1} can range from 1 (all others exit, no entry) to N_t+1 (all stay, entry)
            prob_stay = np.zeros(6)
            for n_exit in range(N_t):  # 0 to N_t-1 others can exit
                n_stay_others = (N_t - 1) - n_exit
                for enter in [0, 1]:
                    N_next = 1 + n_stay_others + enter  # 1 (conditioning firm) + others staying + potential entrant
                    if N_next <= 5:
                        prob_exit_others = (p_exit ** n_exit) * (p_stay ** n_stay_others) * \
                                          scipy.special.comb(N_t - 1, n_exit)
                        prob_entry_event = p_entry if enter == 1 else (1 - p_entry)
                        prob_stay[N_next] += prob_exit_others * prob_entry_event
            
            P_stay[(N_t, x_idx)] = prob_stay
            
            # For e_it=1 (entrant entering): all N_t incumbents face exit
            # N_{t+1} ranges from 1 (all exit, just new entrant) to N_t+1 (all stay + new entrant)
            prob_enter = np.zeros(6)
            for n_stay_incumbents in range(N_t + 1):
                N_next = n_stay_incumbents + 1  # staying incumbents + new entrant
                if N_next <= 5:
                    prob_stay_event = (p_stay ** n_stay_incumbents) * (p_exit ** (N_t - n_stay_incumbents)) * \
                                     scipy.special.comb(N_t, n_stay_incumbents)
                    prob_enter[N_next] += prob_stay_event
            
            P_enter[(N_t, x_idx)] = prob_enter
    
    # Special case: N_t = 0 (no incumbents, only potential entrant)
    for x_idx, x_t in enumerate(x_states):
        p_entry = norm.cdf((gamma_cutoffs[0, x_idx] - gamma_mean) / np.sqrt(gamma_var))
        prob_enter = np.zeros(6)
        prob_enter[1] = p_entry  # If enters, N_{t+1} = 1
        prob_enter[0] = 1 - p_entry  # If doesn't enter, N_{t+1} = 0
        P_enter[(0, x_idx)] = prob_enter
    
    return P_stay, P_enter

def compute_psi(V_bar, P_trans, x_idx):
    """
    Compute Ψ = β * Σ_{N',x'} V(N',x') Pr(x'|x) Pr(N'|N,x,action)
    
    Args:
        V_bar: array of shape (6, 3) with V(N, x)
        P_trans: transition probability vector for N_{t+1}
        x_idx: current x index
    """
    psi = 0
    for x_next_idx in range(3):
        for N_next in range(6):
            if N_next > 0:  # V_bar only defined for N >= 1
                psi += beta * V_bar[N_next, x_next_idx] * P_x[x_idx, x_next_idx] * P_trans[N_next]
    return psi

def solve_equilibrium(max_iter=1000, tol=1e-6):
    """Solve for equilibrium using the iterative procedure"""
    
    # Initialize guesses
    mu_cutoffs = np.ones((6, 3)) * 5  # mu(N_t, x_t)
    gamma_cutoffs = np.ones((6, 3)) * 5  # gamma(N_t, x_t)
    V_bar = np.ones((6, 3)) * 10  # V_bar(N_t, x_t)
    
    # Not defined for certain states
    V_bar[0, :] = 0  # No value function when N=0
    
    for iteration in range(max_iter):
        mu_old = mu_cutoffs.copy()
        gamma_old = gamma_cutoffs.copy()
        V_old = V_bar.copy()
        
        # Step 2: Compute transition matrices
        P_stay, P_enter = compute_transition_matrices(mu_cutoffs, gamma_cutoffs, 
                                                       mu_mean, mu_var, gamma_mean, gamma_var)
        
        # Step 3: Compute Ψ1 and Ψ2
        Psi1 = np.zeros((6, 3))
        Psi2 = np.zeros((6, 3))
        
        for N_t in range(1, 6):
            for x_idx in range(3):
                Psi1[N_t, x_idx] = compute_psi(V_bar, P_stay[(N_t, x_idx)], x_idx)
        
        for N_t in range(0, 5):  # Entry possible for N_t < 5
            for x_idx in range(3):
                Psi2[N_t, x_idx] = compute_psi(V_bar, P_enter[(N_t, x_idx)], x_idx)
        
        # Step 4: Update cutoffs
        for N_t in range(1, 6):
            for x_idx in range(3):
                x_t = x_states[x_idx]
                mu_cutoffs[N_t, x_idx] = profit(N_t, x_t) + Psi1[N_t, x_idx]
        
        for N_t in range(0, 5):
            for x_idx in range(3):
                gamma_cutoffs[N_t, x_idx] = Psi2[N_t, x_idx]
        
        # Step 5: Update V_bar using the formula
        for N_t in range(1, 6):
            for x_idx in range(3):
                x_t = x_states[x_idx]
                pi = profit(N_t, x_t)
                cutoff = mu_cutoffs[N_t, x_idx]
                
                # Standardized cutoff
                z = (cutoff - mu_mean) / np.sqrt(mu_var)
                
                # V_bar = [1-Φ(z)] * [μ + σ * φ(z)/(1-Φ(z))] + Φ(z) * [π + Ψ1]
                prob_exit = 1 - norm.cdf(z)
                prob_stay = norm.cdf(z)
                
                if prob_exit > 1e-10:
                    expected_mu_exit = mu_mean + np.sqrt(mu_var) * norm.pdf(z) / prob_exit
                    V_bar[N_t, x_idx] = prob_exit * expected_mu_exit + prob_stay * (pi + Psi1[N_t, x_idx])
                else:
                    V_bar[N_t, x_idx] = prob_stay * (pi + Psi1[N_t, x_idx])
        
        # Check convergence
        mu_diff = np.max(np.abs(mu_cutoffs - mu_old))
        gamma_diff = np.max(np.abs(gamma_cutoffs - gamma_old))
        V_diff = np.max(np.abs(V_bar[1:, :] - V_old[1:, :]))
        
        max_diff = max(mu_diff, gamma_diff, V_diff)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: max diff = {max_diff:.6e}")
        
        if max_diff < tol:
            print(f"Converged in {iteration} iterations")
            return mu_cutoffs, gamma_cutoffs, V_bar
    
    print("Warning: Did not converge")
    return mu_cutoffs, gamma_cutoffs, V_bar

# Solve the equilibrium
mu_cutoffs, gamma_cutoffs, V_bar = solve_equilibrium()

# Question 6: Report values at state (3, 0)
N_report = 3
x_report_idx = 1  # x=0 corresponds to index 1

print("\n" + "="*50)
print("Question 6 Results:")
print("="*50)
print(f"μ(3, 0) = {mu_cutoffs[N_report, x_report_idx]:.4f}")
print(f"γ(3, 0) = {gamma_cutoffs[N_report, x_report_idx]:.4f}")
print(f"V̄(3, 0) = {V_bar[N_report, x_report_idx]:.4f}")

# Compute V(3, 0, -2)
mu_it = -2
x_t = 0
pi = profit(N_report, x_t)
P_stay_temp, _ = compute_transition_matrices(mu_cutoffs, gamma_cutoffs, mu_mean, mu_var, gamma_mean, gamma_var)
Psi1_temp = compute_psi(V_bar, P_stay_temp[(N_report, x_report_idx)], x_report_idx)
V_3_0_minus2 = max(mu_it, pi + Psi1_temp)
print(f"V(3, 0, -2) = {V_3_0_minus2:.4f}")

print("\n" + "="*50)
print("Full Results:")
print("="*50)
print("\nμ(N_t, x_t) cutoffs:")
print("      x=-5    x=0     x=5")
for N in range(1, 6):
    print(f"N={N}: {mu_cutoffs[N, 0]:6.2f}  {mu_cutoffs[N, 1]:6.2f}  {mu_cutoffs[N, 2]:6.2f}")

print("\nγ(N_t, x_t) cutoffs:")
print("      x=-5    x=0     x=5")
for N in range(0, 5):
    print(f"N={N}: {gamma_cutoffs[N, 0]:6.2f}  {gamma_cutoffs[N, 1]:6.2f}  {gamma_cutoffs[N, 2]:6.2f}")

print("\nV̄(N_t, x_t):")
print("      x=-5    x=0     x=5")
for N in range(1, 6):
    print(f"N={N}: {V_bar[N, 0]:6.2f}  {V_bar[N, 1]:6.2f}  {V_bar[N, 2]:6.2f}")
