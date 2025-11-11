"""Test modified policy iteration convergence"""
import numpy as np
from scipy.stats import norm
from scipy.special import comb

# Parameters
gamma_bar = 5.0
sigma_gamma = np.sqrt(5)
mu_bar = 5.0
sigma_mu = np.sqrt(5)
beta = 0.9
N_max = 5
x_values = [-5, 0, 5]

# Transition matrix for x
P_x = np.array([[0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6]])

def profit(N, x):
    """Cournot profit"""
    if N == 0:
        return 0.0
    return (10 + x)**2 / (N + 1)**2 - 5.0

def binom_pmf(k, n, p):
    """Binomial PMF"""
    if k < 0 or k > n or n == 0:
        return 1.0 if k == 0 else 0.0
    return comb(n, k, exact=True) * (p**k) * ((1-p)**(n-k))

# Initialize
mu_cutoff = {(N, x): profit(N, x) for N in range(1, N_max+1) for x in x_values}
gamma_cutoff = {(N, x): 0.0 for N in range(N_max+1) for x in x_values}
V_bar = {(N, x): profit(N, x) / (1 - beta) for N in range(1, N_max+1) for x in x_values}

print("Modified Policy Iteration Test")
print("="*60)
print("Initial values at (3,0):")
print(f"  μ̄(3,0) = {mu_cutoff[(3,0)]:.4f}")
print(f"  γ̄(3,0) = {gamma_cutoff[(3,0)]:.4f}")
print(f"  V̄(3,0) = {V_bar[(3,0)]:.4f}")
print()

# Modified policy iteration
for it in range(500):
    mu_old = mu_cutoff.copy()
    gamma_old = gamma_cutoff.copy()
    V_old = V_bar.copy()
    
    # Compute Psi using OLD cutoffs and OLD V̄
    Psi1 = {}
    Psi2 = {}
    
    for N in range(1, N_max+1):
        for x_idx, x in enumerate(x_values):
            # Psi1: continuation value if incumbent stays
            p_stay = norm.cdf((mu_old[(N, x)] - mu_bar) / sigma_mu)
            
            val = 0.0
            for n_other_stay in range(N):
                prob = binom_pmf(n_other_stay, N - 1, p_stay)
                N_next = n_other_stay + 1
                
                if N_next < N_max:
                    p_enter = norm.cdf((gamma_old[(N_next, x)] - gamma_bar) / sigma_gamma)
                    # Only ONE potential entrant per period
                    for x_next_idx, x_next in enumerate(x_values):
                        val += beta * P_x[x_idx, x_next_idx] * prob * (1 - p_enter) * V_old.get((N_next, x_next), 0)
                        val += beta * P_x[x_idx, x_next_idx] * prob * p_enter * V_old.get((N_next + 1, x_next), 0)
                else:
                    for x_next_idx, x_next in enumerate(x_values):
                        val += beta * P_x[x_idx, x_next_idx] * prob * V_old.get((N_next, x_next), 0)
            
            Psi1[(N, x)] = val
            
            # Psi2: continuation value if entrant enters
            if N < N_max:
                val = 0.0
                for n_stay in range(N + 1):
                    prob = binom_pmf(n_stay, N, p_stay)
                    N_next = n_stay + 1  # Entrant already entered, NO second entrant
                    
                    for x_next_idx, x_next in enumerate(x_values):
                        val += beta * P_x[x_idx, x_next_idx] * prob * V_old.get((N_next, x_next), 0)
                
                Psi2[(N, x)] = val
    
    # Update cutoffs (NO damping)
    for N in range(1, N_max+1):
        for x in x_values:
            mu_cutoff[(N, x)] = profit(N, x) + Psi1[(N, x)]
            if N < N_max:
                gamma_cutoff[(N, x)] = Psi2.get((N, x), 0.0)
    
    # Update V̄ using NEW cutoffs (modified policy iteration)
    for N in range(1, N_max+1):
        for x in x_values:
            thresh = mu_cutoff[(N, x)]  # NEW cutoff
            z = (thresh - mu_bar) / sigma_mu
            
            if z > 10:
                V_bar[(N, x)] = mu_bar + sigma_mu * norm.pdf(z) / 1e-12
            elif z < -10:
                V_bar[(N, x)] = thresh
            else:
                Phi_z = norm.cdf(z)
                phi_z = norm.pdf(z)
                stay_term = Phi_z * thresh
                if Phi_z < 0.9999:
                    mills = phi_z / max(1 - Phi_z, 1e-12)
                    exit_term = (1 - Phi_z) * (mu_bar + sigma_mu * mills)
                else:
                    exit_term = 0.0
                V_bar[(N, x)] = stay_term + exit_term
    
    # Check convergence
    max_diff = max(
        max(abs(mu_cutoff[(N, x)] - mu_old[(N, x)]) for N in range(1, N_max+1) for x in x_values),
        max(abs(gamma_cutoff[(N, x)] - gamma_old[(N, x)]) for N in range(N_max+1) for x in x_values),
        max(abs(V_bar[(N, x)] - V_old[(N, x)]) for N in range(1, N_max+1) for x in x_values)
    )
    
    if it % 20 == 0:
        print(f"Iteration {it}: max_diff = {max_diff:.6e}")
    
    if max_diff < 1e-10:
        print(f"\n✓ Converged in {it+1} iterations")
        break
else:
    print(f"\n✗ Did not converge after 500 iterations")

print()
print("Final values at (3,0):")
print(f"  μ̄(3,0) = {mu_cutoff[(3,0)]:.4f}")
print(f"  γ̄(3,0) = {gamma_cutoff[(3,0)]:.4f}")
print(f"  V̄(3,0) = {V_bar[(3,0)]:.4f}")
