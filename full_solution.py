"""Full Solution Maximum Likelihood Estimation for Dynamic Entry/Exit Game
Implements exact finite state space full solution using nonlinear root finding
for value functions, inspired by Hotz-Miller inversion.
"""
import pickle
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from dg import DynamicGame

def load_estimation_data(filename='estimation_data.pkl'):
    """Load estimation data from pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['d_hat'], data['e_hat'], data['theta_hat'], data['eq']

def compute_ccps_from_equilibrium(game):
    """Compute CCPs from solved equilibrium."""
    d_computed = {}
    for N in range(1, 6):
        for j, xv in enumerate(game.x_vals):
            d_computed[(N, xv)] = norm.cdf((game.mu_cutoff[N, j] - game.mu_mean) / game.mu_std)
    e_computed = {}
    for N in range(5):
        for j, xv in enumerate(game.x_vals):
            e_computed[(N, xv)] = norm.cdf((game.gamma_cutoff[N, j] - game.gamma_mean) / game.gamma_std)
    return d_computed, e_computed

def full_solution_objective(theta, d_hat, e_hat):
    """Objective function for full solution MLE.
    
    For given theta = [mu_mean, mu_var, gamma_mean, gamma_var],
    solve equilibrium exactly, compute CCPs, return sum of squared errors.
    """
    mu_mean, mu_var, gamma_mean, gamma_var = theta
    
    # Ensure variances are positive
    if mu_var <= 0 or gamma_var <= 0:
        return 1e10
    
    # Create game with these parameters
    game = DynamicGame(mu_mean=mu_mean, mu_var=mu_var, gamma_mean=gamma_mean, gamma_var=gamma_var)
    
    # Solve equilibrium
    converged = game.solve_equilibrium()
    if not converged:
        return 1e10  # Penalize non-convergence
    
    # Compute CCPs from equilibrium
    d_comp, e_comp = compute_ccps_from_equilibrium(game)
    
    # Compute sum of squared errors
    error = 0.0
    for key in d_hat:
        error += (d_comp[key] - d_hat[key]) ** 2
    for key in e_hat:
        error += (e_comp[key] - e_hat[key]) ** 2
    
    return error

def estimate_full_solution(d_hat, e_hat, theta_init, bounds=None):
    """Perform full solution MLE."""
    if bounds is None:
        # Tighter bounds around initial guess
        bounds = [(theta_init[0]-2, theta_init[0]+2), (theta_init[1]-1, theta_init[1]+1), 
                  (theta_init[2]-2, theta_init[2]+2), (theta_init[3]-1, theta_init[3]+1)]
    
    result = minimize(
        full_solution_objective,
        theta_init,
        args=(d_hat, e_hat),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50, 'ftol': 1e-4}  # Fewer iterations, looser tolerance
    )
    
    return result

def main():
    print("Full Solution MLE for Dynamic Entry/Exit Game")
    print("=" * 50)
    
    # Load data
    d_hat, e_hat, theta_init, eq_true = load_estimation_data()
    print(f"Loaded {len(d_hat)} incumbent CCPs and {len(e_hat)} entry CCPs")
    print(f"Initial guess from BBL: μ={theta_init[0]:.3f}, σ_μ={theta_init[1]:.3f}, γ={theta_init[2]:.3f}, σ_γ={theta_init[3]:.3f}")
    
    # Perform estimation
    print("\nEstimating parameters using full solution...")
    result = estimate_full_solution(d_hat, e_hat, theta_init)
    
    if result.success:
        theta_hat = result.x
        print("Estimation successful!")
        print(f"Objective value: {result.fun:.6f}")
        print(f"Estimated parameters:")
        print(f"  μ (flow profit mean): {theta_hat[0]:.6f}")
        print(f"  σ_μ (flow profit std): {theta_hat[1]:.6f}")
        print(f"  γ (entry cost mean): {theta_hat[2]:.6f}")
        print(f"  σ_γ (entry cost std): {theta_hat[3]:.6f}")
        
        # Compute CCP errors
        game_hat = DynamicGame(mu_mean=theta_hat[0], mu_var=theta_hat[1]**2, 
                              gamma_mean=theta_hat[2], gamma_var=theta_hat[3]**2)
        game_hat.solve_equilibrium()
        d_comp, e_comp = compute_ccps_from_equilibrium(game_hat)
        
        mse_d = np.mean([(d_comp[k] - d_hat[k])**2 for k in d_hat])
        mse_e = np.mean([(e_comp[k] - e_hat[k])**2 for k in e_hat])
        print(f"CCPs MSE - Incumbents: {mse_d:.6f}, Entry: {mse_e:.6f}")
        
        # Compare to true parameters
        true_theta = [eq_true.mu_mean, eq_true.mu_std**2, eq_true.gamma_mean, eq_true.gamma_std**2]
        print(f"\nTrue parameters: μ={true_theta[0]:.3f}, σ_μ={true_theta[1]:.3f}, γ={true_theta[2]:.3f}, σ_γ={true_theta[3]:.3f}")
        print(f"Estimation error: μ={theta_hat[0]-true_theta[0]:.3f}, σ_μ={theta_hat[1]-true_theta[1]:.3f}, γ={theta_hat[2]-true_theta[2]:.3f}, σ_γ={theta_hat[3]-true_theta[3]:.3f}")
    else:
        print("Estimation failed:", result.message)

if __name__ == "__main__":
    main()