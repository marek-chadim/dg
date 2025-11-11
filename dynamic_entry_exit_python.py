"""
Dynamic Entry/Exit Game - Python Implementation
Homework Solution for Empirical IO (Ackerberg-BBL Framework)

This implements the Pakes-McGuire (1994) style algorithm for computing 
Markov Perfect Equilibrium in a dynamic entry/exit game with:
- Cournot competition with homogeneous products
- Stochastic exit values and entry costs
- Industry-wide demand shocks

Based on: Doraszelski & Pakes (2007), Berry & Compiani (2020)
"""

import numpy as np
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List
import time

@dataclass
class ModelParameters:
    """Model primitives"""
    gamma_bar: float = 5.0      # Mean entry cost
    sigma_gamma: float = np.sqrt(5)  # SD entry cost
    mu_bar: float = 5.0         # Mean exit value
    sigma_mu: float = np.sqrt(5)    # SD exit value
    beta: float = 0.9           # Discount factor
    max_N: int = 5              # Maximum number of firms
    x_values: List[int] = None  # Demand states
    
    def __post_init__(self):
        if self.x_values is None:
            self.x_values = [-5, 0, 5]


class DynamicEntryExitGame:
    """
    Solver for dynamic entry/exit game with Cournot competition.
    
    Implements Gauss-Seidel value iteration to compute Markov Perfect Equilibrium
    cutoff policies and value functions.
    """
    
    def __init__(self, params: ModelParameters):
        self.p = params
        self.convergence_history = []
        
        # Initialize policy and value functions
        self.mu_cuts = {}  # Exit cutoffs
        self.gamma_cuts = {}  # Entry cutoffs
        self.V_bar = {}  # Integrated value functions
        
        # Transition matrix for x (demand states)
        self.x_transition = {
            -5: {-5: 0.6, 0: 0.2, 5: 0.2},
            0: {-5: 0.2, 0: 0.6, 5: 0.2},
            5: {-5: 0.2, 0: 0.2, 5: 0.6}
        }
        
        self._initialize_values()
    
    def _initialize_values(self):
        """Initialize guesses per homework Step 1.

        Homework specifies we simply 'guess' μ(N,x), γ(N,x), V̄(N,x).
        Use:
          μ_guess(N,x)   = π(N,x)            (only for N>0)
          γ_guess(N,x)   = 0                 (only for N<max_N)
          V̄_guess(N,x)  = π(N,x)/(1-β)      (only for N>0)
        These are natural myopic-based starting values without shifting by μ̄ or γ̄.
        """
        for N in range(self.p.max_N + 1):
            self.mu_cuts[N] = {}
            self.gamma_cuts[N] = {}
            self.V_bar[N] = {}
            for x in self.p.x_values:
                pi = self.profit(N, x)
                if N > 0:
                    self.mu_cuts[N][x] = pi  # exit cutoff initial guess
                    self.V_bar[N][x] = pi / (1 - self.p.beta)
                if N < self.p.max_N:
                    self.gamma_cuts[N][x] = 0.0  # entry cutoff initial guess
    
    def profit(self, N: int, x: int) -> float:
        """
        Single-period Nash equilibrium profit per firm.
        
        With inverse demand p = 10 - Q + x and Cournot competition:
        π(N,x) = [(10+x)/(N+1)]² - 5
        """
        if N == 0:
            return 0.0
        return (10 + x)**2 / (N + 1)**2 - 5.0
    
    def compute_transition_probs(self) -> Tuple[Dict, Dict]:
        """
        Compute state transition probabilities consistent with homework setup:
        Exactly one potential entrant per period; in the entrant case there is no
        "secondary" entrant after the focal potential entrant. In incumbent case,
        potential entrant may enter (at most one entrant total in the period).
        
        Returns:
            P_stay: Pr(N_{t+1} | N_t, x_t, incumbent stays)
            P_enter: Pr(N_{t+1} | N_t, x_t, entrant enters)
        """
        P_stay = {x: {} for x in self.p.x_values}
        P_enter = {x: {} for x in self.p.x_values}
        
        for x in self.p.x_values:
            for N in range(self.p.max_N + 1):
                P_stay[x][N] = np.zeros(self.p.max_N + 1)
                P_enter[x][N] = np.zeros(self.p.max_N + 1)
                
                if N == 0:
                    continue
                
                # Probability each incumbent stays
                z_stay = (self.mu_cuts[N][x] - self.p.mu_bar) / self.p.sigma_mu
                p_stay = norm.cdf(z_stay)
                
                # Incumbent case: focal firm stays, other N-1 decide; THEN at most one entrant
                for n_other_stay in range(N):
                    binom_prob = comb(N - 1, n_other_stay, exact=True) * \
                                p_stay**n_other_stay * (1 - p_stay)**(N - 1 - n_other_stay)
                    N_after_incumbents = n_other_stay + 1  # include focal
                    if N_after_incumbents < self.p.max_N:
                        # Single potential entrant decides
                        z_enter = (self.gamma_cuts[N_after_incumbents][x] - self.p.gamma_bar) / self.p.sigma_gamma
                        p_enter = norm.cdf(z_enter)
                        # Either no entry (stay at N_after_incumbents) or one entry (N_after_incumbents+1)
                        P_stay[x][N][N_after_incumbents] += binom_prob * (1 - p_enter)
                        P_stay[x][N][N_after_incumbents + 1] += binom_prob * p_enter
                    else:
                        # Capacity reached; no entry possible
                        P_stay[x][N][N_after_incumbents] += binom_prob

                # Entrant case: focal entrant enters first (N increases by 1), incumbents then decide.
                # No secondary entrant this period.
                if N < self.p.max_N:
                    for n_stay in range(N + 1):
                        binom_prob = comb(N, n_stay, exact=True) * \
                                    p_stay**n_stay * (1 - p_stay)**(N - n_stay)
                        N_next = n_stay + 1  # entrant count adds exactly one
                        P_enter[x][N][min(N_next, self.p.max_N)] += binom_prob
        
        return P_stay, P_enter
    
    def update_continuation_values(self, P_stay: Dict, P_enter: Dict, V_old: Dict) -> Tuple[Dict, Dict]:
        """
        Compute Ψ₁ and Ψ₂: expected discounted continuation values.
        Uses V_old from previous iteration for proper value function iteration.
        """
        Psi1 = {x: {} for x in self.p.x_values}
        Psi2 = {x: {} for x in self.p.x_values}
        
        for x in self.p.x_values:
            for N in range(1, self.p.max_N + 1):
                psi1 = 0.0
                psi2 = 0.0
                
                for x_next in self.p.x_values:
                    p_x = self.x_transition[x][x_next]
                    
                    for N_next in range(1, self.p.max_N + 1):
                        # Use V_old, not self.V_bar (from previous iteration)
                        psi1 += self.p.beta * V_old[N_next][x_next] * \
                               p_x * P_stay[x][N][N_next]
                        
                        if N < self.p.max_N:
                            psi2 += self.p.beta * V_old[N_next][x_next] * \
                                   p_x * P_enter[x][N][N_next]
                
                Psi1[x][N] = psi1
                if N < self.p.max_N:
                    Psi2[x][N] = psi2
        
        return Psi1, Psi2
    
    def update_integrated_value(self, N: int, x: int, delta: float, pi: float, psi1: float) -> float:
        """
        Update V̄(N,x) using closed-form truncated normal formula.
        
        CORRECTED FORMULA:
        V̄(N,x) = Pr(stay) × [π + Ψ₁] + Pr(exit) × E[μ|exit]
               = Φ(z) × δ + [1-Φ(z)] × [μ̄ + σ_μ·λ(z)]
        
        where:
        - δ = π + Ψ₁ is the stay value (exit cutoff)
        - z = (δ - μ̄)/σ_μ
        - λ(z) is inverse Mills ratio
        """
        z = (delta - self.p.mu_bar) / self.p.sigma_mu
        
        # Handle numerical extremes
        if z > 10:  # Everyone exits
            # Use approximation for large z: λ(z) ≈ 1/z
            mills_approx = 1.0 / z if z > 0 else 0
            return self.p.mu_bar + self.p.sigma_mu * mills_approx
        elif z < -10:  # Nobody exits, all stay
            return pi + psi1  # Stay value = π + Ψ₁
        
        # Standard calculation
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        # Inverse Mills ratio with numerical protection
        mills_ratio = phi_z / max(1 - Phi_z, 1e-10)
        
        # V̄ = Pr(stay) × [π + Ψ₁] + Pr(exit) × E[μ|exit]
        stay_term = Phi_z * delta  # Note: delta = π + Ψ₁
        exit_term = (1 - Phi_z) * (self.p.mu_bar + self.p.sigma_mu * mills_ratio)
        
        return stay_term + exit_term
    
    def solve_equilibrium(self, max_iter: int = 200, tol: float = 1e-6, 
                         damping: float = 0.0, verbose: bool = True) -> Dict:
        """
        Solve for Markov Perfect Equilibrium using Gauss-Seidel iteration.
        
        Returns:
            Dictionary with converged policies and values
        """
        start_time = time.time()
        
        for iteration in range(max_iter):
            # Store old values for convergence check
            mu_old = {N: {x: self.mu_cuts[N][x] for x in self.p.x_values} 
                     for N in range(1, self.p.max_N + 1)}
            gamma_old = {N: {x: self.gamma_cuts[N][x] for x in self.p.x_values} 
                        for N in range(self.p.max_N)}
            V_old = {N: {x: self.V_bar[N][x] for x in self.p.x_values} 
                    for N in range(1, self.p.max_N + 1)}
            
            # Step 1: Compute transition matrices
            P_stay, P_enter = self.compute_transition_probs()
            
            # Step 2: Compute continuation values (using V_old)
            Psi1, Psi2 = self.update_continuation_values(P_stay, P_enter, V_old)
            
            # Step 3: Update cutoffs and values (with optional damping)
            for N in range(1, self.p.max_N + 1):
                for x in self.p.x_values:
                    pi = self.profit(N, x)
                    
                    # Update exit cutoff (indifference condition)
                    mu_new = pi + Psi1[x][N]
                    if damping > 0:
                        self.mu_cuts[N][x] = damping * mu_new + (1 - damping) * mu_old[N][x]
                    else:
                        self.mu_cuts[N][x] = mu_new
                    
                    # Update entry cutoff
                    if N < self.p.max_N:
                        gamma_new = Psi2[x][N]
                        if damping > 0:
                            self.gamma_cuts[N][x] = damping * gamma_new + (1 - damping) * gamma_old[N][x]
                        else:
                            self.gamma_cuts[N][x] = gamma_new
                    
                    # Update integrated value function
                    V_new = self.update_integrated_value(
                        N, x, self.mu_cuts[N][x], pi, Psi1[x][N]
                    )
                    if damping > 0:
                        self.V_bar[N][x] = damping * V_new + (1 - damping) * V_old[N][x]
                    else:
                        self.V_bar[N][x] = V_new
            
            # Check convergence
            max_diff = 0.0
            for N in range(1, self.p.max_N + 1):
                for x in self.p.x_values:
                    max_diff = max(
                        max_diff,
                        abs(self.mu_cuts[N][x] - mu_old[N][x]),
                        abs(self.V_bar[N][x] - V_old[N][x])
                    )
            
            # Check gamma cutoffs too
            for N in range(self.p.max_N):
                for x in self.p.x_values:
                    max_diff = max(max_diff, abs(self.gamma_cuts[N][x] - gamma_old[N][x]))
            
            self.convergence_history.append({
                'iteration': iteration,
                'error': max_diff
            })
            
            if verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: max error = {max_diff:.2e}")
            
            if max_diff < tol:
                elapsed = time.time() - start_time
                if verbose:
                    print(f"\n✓ Converged in {iteration} iterations ({elapsed:.2f}s)")
                break
        else:
            print(f"Warning: Did not converge after {max_iter} iterations")
        
        return {
            'mu_cuts': self.mu_cuts,
            'gamma_cuts': self.gamma_cuts,
            'V_bar': self.V_bar,
            'converged': max_diff < tol,
            'iterations': iteration
        }
    
    def compute_value_at_point(self, N: int, x: int, mu: float) -> float:
        """
        Compute V(N, x, μ) for specific exit value.
        
        Used for Question 6: V(3, 0, -2)
        
        CORRECTED:
        If μ ≤ μ̄(N,x): firm stays, gets V̄(N,x)
        If μ > μ̄(N,x): firm exits, gets μ
        """
        delta = self.mu_cuts[N][x]  # Exit cutoff μ̄(N,x)
        
        if mu <= delta:
            # Firm stays: gets integrated value
            return self.V_bar[N][x]
        else:
            # Firm exits: gets exit value
            return mu
    
    def print_results(self):
        """Print key equilibrium results"""
        print("\n" + "="*60)
        print("QUESTION 6: Equilibrium Values at (N=3, x=0)")
        print("="*60)
        
        N, x = 3, 0
        print(f"Exit cutoff μ̄(3,0)  = {self.mu_cuts[N][x]:.4f}")
        print(f"Entry cutoff γ̄(3,0) = {self.gamma_cuts[N][x]:.4f}")
        print(f"Integrated value V̄(3,0) = {self.V_bar[N][x]:.4f}")
        print(f"Point value V(3,0,-2) = {self.compute_value_at_point(N, x, -2):.4f}")
        
        print("\n" + "="*60)
        print("Exit Cutoff Policy μ̄(N,x)")
        print("="*60)
        print("N\\x  ", end="")
        for x in self.p.x_values:
            print(f"{x:>8}", end="")
        print()
        print("-" * 30)
        for N in range(1, self.p.max_N + 1):
            print(f"{N:3}  ", end="")
            for x in self.p.x_values:
                print(f"{self.mu_cuts[N][x]:8.3f}", end="")
            print()
        
        print("\n" + "="*60)
        print("Entry Cutoff Policy γ̄(N,x)")
        print("="*60)
        print("N\\x  ", end="")
        for x in self.p.x_values:
            print(f"{x:>8}", end="")
        print()
        print("-" * 30)
        for N in range(self.p.max_N):
            print(f"{N:3}  ", end="")
            for x in self.p.x_values:
                print(f"{self.gamma_cuts[N][x]:8.3f}", end="")
            print()
    
    def plot_convergence(self):
        """Plot convergence path"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = [h['iteration'] for h in self.convergence_history]
        errors = [np.log10(h['error'] + 1e-12) for h in self.convergence_history]
        
        ax.plot(iterations, errors, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('log₁₀(Max Error)', fontsize=12)
        ax.set_title('Convergence of Value Iteration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def simulate_market(game: DynamicEntryExitGame, T: int = 10000, 
                   initial_N: int = 0, initial_x: int = 0) -> Dict:
    """
    Simulate market evolution for T periods.
    
    Question 7: Forward simulation from empty market.
    """
    history = []
    N, x = initial_N, initial_x
    
    for t in range(T):
        history.append({'period': t, 'N': N, 'x': x})
        
        # Incumbent exit decisions
        survivors = 0
        for _ in range(N):
            mu_draw = game.p.mu_bar + game.p.sigma_mu * np.random.randn()
            if mu_draw <= game.mu_cuts[N][x]:
                survivors += 1
        
        # Entry decision
        entry = 0
        if survivors < game.p.max_N:
            gamma_draw = game.p.gamma_bar + game.p.sigma_gamma * np.random.randn()
            if gamma_draw <= game.gamma_cuts[survivors][x]:
                entry = 1
        
        # Update state
        N = min(survivors + entry, game.p.max_N)
        
        # Draw next x
        probs = game.x_transition[x]
        x = np.random.choice(game.p.x_values, p=[probs[xi] for xi in game.p.x_values])
    
    # Compute statistics
    N_values = [h['N'] for h in history]
    avg_N = np.mean(N_values)
    
    # Distribution
    N_dist = {n: np.sum(np.array(N_values) == n) / T * 100 
             for n in range(game.p.max_N + 1)}
    
    print("\n" + "="*60)
    print(f"QUESTION 7: Market Simulation (T={T} periods)")
    print("="*60)
    print(f"Average number of firms: {avg_N:.3f}")
    print("\nDistribution of N:")
    for n in range(game.p.max_N + 1):
        print(f"  N={n}: {N_dist[n]:5.2f}%")
    
    return {
        'history': history,
        'avg_N': avg_N,
        'N_dist': N_dist
    }


if __name__ == "__main__":
    # Initialize model
    print("Dynamic Entry/Exit Game - Python Implementation")
    print("="*60)
    
    params = ModelParameters()
    game = DynamicEntryExitGame(params)
    
    # Solve for equilibrium
    print("\nSolving for Markov Perfect Equilibrium...")
    results = game.solve_equilibrium(tol=1e-8, max_iter=1000, damping=0.0, verbose=True)
    
    # Display results
    game.print_results()
    
    # Plot convergence
    fig = game.plot_convergence()
    plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
    print("\n✓ Convergence plot saved to 'convergence.png'")
    
    # Simulate market
    print("\n" + "="*60)
    print("Running forward simulation...")
    sim_results = simulate_market(game, T=10000)
    
    print("\n" + "="*60)
    print("✓ All computations complete!")
    print("="*60)
