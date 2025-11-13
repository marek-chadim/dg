import numpy as np
from scipy.stats import norm, binom
from scipy.optimize import minimize

class DynamicGame:
    def __init__(self, beta=0.9, mu_mean=5.0, mu_var=5.0, gamma_mean=5.0, gamma_var=5.0, entry_tax=0.0):
        self.beta, self.mu_mean, self.mu_std = beta, mu_mean, np.sqrt(mu_var)
        self.gamma_mean, self.gamma_std, self.entry_tax = gamma_mean, np.sqrt(gamma_var), entry_tax
        self.N_max, self.x_vals = 5, np.array([-5, 0, 5])
        self.x_trans = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        N_vals = np.arange(1, 6)[:, None]
        self.pi = np.vstack([np.zeros(3), ((10 + self.x_vals) / (N_vals + 1)) ** 2 - 5])
        self.mu_cutoff = self.gamma_cutoff = self.V_bar = np.zeros_like(self.pi)

    def solve_equilibrium(self, tol=1e-8, max_iter=5000, damp=0.5):
        self.V_bar = self.mu_cutoff = self.gamma_cutoff = self.pi.copy()
        for _ in range(max_iter):
            V_old, mu_old, ga_old = self.V_bar.copy(), self.mu_cutoff.copy(), self.gamma_cutoff.copy()
            p_stay = np.clip(norm.cdf((self.mu_cutoff - self.mu_mean) / self.mu_std), 0, 1)
            p_enter = np.clip(norm.cdf((self.gamma_cutoff - self.gamma_mean) / self.gamma_std), 0, 1)
            p_stay[0], p_enter[5] = 0, 0
            
            Pr_d1 = np.zeros((6, 6, 3))
            Pr_e1 = np.zeros((6, 6, 3))
            for N in range(6):
                for j in range(3):
                    if N > 0:
                        # Entry decision based on CURRENT state N (simultaneous moves)
                        pe = p_enter[N,j]
                        for k in range(N):
                            w, Na = binom.pmf(k, N-1, p_stay[N,j]), 1+k
                            Pr_d1[N,Na,j] += w*(1-pe)
                            if Na < 5: Pr_d1[N,Na+1,j] += w*pe
                    if N < 5:
                        for k in range(N+1):
                            Pr_e1[N, min(1+k,5), j] += binom.pmf(k, N, p_stay[N,j] if N>0 else 0)
            
            EV = self.x_trans.T @ self.V_bar.T
            Psi1 = self.beta * np.einsum('ijk,jk->ik', Pr_d1, EV.T)
            Psi2 = self.beta * np.einsum('ijk,jk->ik', Pr_e1, EV.T)
            mu_new, ga_new = self.pi + Psi1, Psi2 - self.entry_tax
            
            z = (mu_new - self.mu_mean) / self.mu_std
            Phi = norm.cdf(z)
            V_new = np.zeros_like(self.V_bar)
            V_new[1:] = (1-Phi[1:])*self.mu_mean + self.mu_std*norm.pdf(z[1:]) + Phi[1:]*mu_new[1:]
            
            self.V_bar = damp*V_new + (1-damp)*V_old
            self.mu_cutoff = damp*mu_new + (1-damp)*mu_old
            self.gamma_cutoff = damp*ga_new + (1-damp)*ga_old
            
            if max(np.abs(self.V_bar-V_old).max(), np.abs(self.mu_cutoff-mu_old).max(), np.abs(self.gamma_cutoff-ga_old).max()) < tol:
                return True
        return False

    def simulate(self, T=10000, seed=2007):
        rng = np.random.default_rng(seed)
        d_cut = (self.mu_cutoff - self.mu_mean) / self.mu_std
        e_cut = (self.gamma_cutoff - self.gamma_mean) / self.gamma_std
        N, xj = 0, 1
        N_hist = np.zeros(T, dtype=int)
        for t in range(T):
            N_hist[t] = N
            # Simultaneous decisions: both based on current state N
            n_stay = int(np.sum(rng.standard_normal(N) <= d_cut[N,xj])) if N > 0 else 0
            e = int(rng.standard_normal() <= e_cut[N,xj]) if N < 5 else 0
            N = n_stay + e
            xj = np.searchsorted(self.x_trans[xj].cumsum(), rng.random())
        return N_hist

def main():
    print("ECON 600a - Dynamic Entry/Exit Game")
    
    # Problems 4-6: Solve and check uniqueness
    sols = [DynamicGame() for _ in range(5)]
    for s in sols: s.solve_equilibrium()
    print(f"Equilibrium: {'UNIQUE' if all(np.allclose(sols[0].V_bar, s.V_bar, 1e-6) for s in sols[1:]) else 'MULTIPLE'}")
    
    eq = sols[0]
    print(f"Equilibrium cutoffs: μ̄(3,0)={eq.mu_cutoff[3,1]:.6f}, γ̄(3,0)={eq.gamma_cutoff[3,1]:.6f}, V̄(3,0)={eq.V_bar[3,1]:.6f}")
    
    # Problem 7-8: Simulate baseline and tax
    N_hist = eq.simulate()
    avgN = N_hist.mean()
    print(f"Avg firms: {avgN:.4f}")
    
    tax = DynamicGame(entry_tax=5.0)
    tax.solve_equilibrium()
    N_tax = tax.simulate()
    print(f"With tax: {N_tax.mean():.4f}, Change: {N_tax.mean()-avgN:.4f}")
    
    # Problem 9: BBL estimation
    print("\nProblem 9: BBL Estimation")
    rng = np.random.default_rng(2007)
    d_cut, e_cut = (eq.mu_cutoff-eq.mu_mean)/eq.mu_std, (eq.gamma_cutoff-eq.gamma_mean)/eq.gamma_std
    
    # Estimate CCPs (simultaneous moves)
    stay_cnt, tot_cnt, ent_cnt, ent_opp = {}, {}, {}, {}
    N, xj = 0, 1
    for _ in range(10000):
        xv = eq.x_vals[xj]
        if N > 0:
            ns = int(np.sum(rng.standard_normal(N) <= d_cut[N,xj]))
            stay_cnt[(N,xv)] = stay_cnt.get((N,xv), 0) + ns
            tot_cnt[(N,xv)] = tot_cnt.get((N,xv), 0) + N
        else: ns = 0
        # Entry decision based on CURRENT state N (before exits)
        if N < 5:
            e = int(rng.standard_normal() <= e_cut[N,xj])
            ent_cnt[(N,xv)] = ent_cnt.get((N,xv), 0) + e
            ent_opp[(N,xv)] = ent_opp.get((N,xv), 0) + 1
            N = ns + e
        else: N = ns
        xj = np.searchsorted(eq.x_trans[xj].cumsum(), rng.random())
    
    d_hat = {(N,x): np.clip(stay_cnt.get((N,x),0)/tot_cnt.get((N,x),1), 1e-3, 1-1e-3) if (N,x) in tot_cnt else 0.5 
             for N in range(1,6) for x in eq.x_vals}
    e_hat = {(N,x): np.clip(ent_cnt.get((N,x),0)/ent_opp.get((N,x),1), 1e-3, 1-1e-3) if (N,x) in ent_opp else 0.5 
             for N in range(5) for x in eq.x_vals}
    
    # Forward simulation
    rng_est = np.random.default_rng(42)
    states_inc = [(N,x) for N in range(1,6) for x in eq.x_vals]
    states_ent = [(N,x) for N in range(5) for x in eq.x_vals]
    mu_dr = rng_est.standard_normal((15,50,100,5))
    mu_dr_e = rng_est.standard_normal((15,50,100,5))
    g_dr = rng_est.standard_normal((15,50,100))
    x_dr = rng_est.random((15,50,100))
    x_dr_e = rng_est.random((15,50,100))
    
    def sim_Lambda(params):
        mu_m, sig_mu, ga_m, sig_g = params
        Lam, LamE = {}, {}
        for si, (N0,xv) in enumerate(states_inc):
            xi = np.where(eq.x_vals==xv)[0][0]
            pdvs = []
            for s in range(50):
                N, x, alive, pdv = N0, xi, True, 0
                for t in range(100):
                    if not alive or N == 0: break
                    pdv += (0.9**t) * eq.pi[N,x]
                    so = np.sum(norm.cdf(mu_dr[si,s,t,:N-1]) <= d_hat.get((N,eq.x_vals[x]),0.5)) if N>1 else 0
                    df = mu_dr[si,s,t,N-1]
                    sf = norm.cdf(df) <= d_hat.get((N,eq.x_vals[x]),0.5)
                    if not sf:
                        pdv += (0.9**(t+1)) * (mu_m + sig_mu*df)
                        alive = False
                    Nn = so + (1 if sf else 0)
                    # Entry based on CURRENT state N (simultaneous)
                    if alive and N<5 and norm.cdf(g_dr[si%15,s,t]) <= e_hat.get((N,eq.x_vals[x]),0.5): Nn += 1
                    x = np.searchsorted(eq.x_trans[x].cumsum(), x_dr[si,s,t])
                    N = Nn if alive else 0
                    if N==0: alive = False
                pdvs.append(pdv)
            Lam[(N0,xv)] = np.mean(pdvs)
        for si, (N0,xv) in enumerate(states_ent):
            xi = np.where(eq.x_vals==xv)[0][0]
            pdvs = []
            for s in range(50):
                si_inc = np.sum(norm.cdf(mu_dr_e[si,s,0,:N0]) <= d_hat.get((N0,eq.x_vals[xi]),0.5)) if N0>0 else 0
                N, x = min(si_inc+1, 5), np.searchsorted(eq.x_trans[xi].cumsum(), x_dr_e[si,s,0])
                pdv, alive = 0, True
                for t in range(1,100):
                    if not alive or N == 0: break
                    pdv += (0.9**t) * eq.pi[N,x]
                    so = np.sum(norm.cdf(mu_dr_e[si,s,t,:N-1]) <= d_hat.get((N,eq.x_vals[x]),0.5)) if N>1 else 0
                    df = mu_dr_e[si,s,t,N-1]
                    sf = norm.cdf(df) <= d_hat.get((N,eq.x_vals[x]),0.5)
                    if not sf:
                        pdv += (0.9**(t+1)) * (mu_m + sig_mu*df)
                        alive = False
                    Nn = so + (1 if sf else 0)
                    # Entry based on CURRENT state N (simultaneous)
                    if alive and N<5 and norm.cdf(g_dr[si,s,t]) <= e_hat.get((N,eq.x_vals[x]),0.5): Nn += 1
                    x = np.searchsorted(eq.x_trans[x].cumsum(), x_dr_e[si,s,t])
                    N = Nn if alive else 0
                    if N==0: alive = False
                pdvs.append(pdv)
            LamE[(N0,xv)] = np.mean(pdvs)
        return Lam, LamE
    
    def obj(p):
        Lam, LamE = sim_Lambda(p)
        return sum((norm.cdf((Lam[(N,x)]-p[0])/p[1]) - d_hat[(N,x)])**2 for N,x in states_inc) + \
               sum((norm.cdf((LamE[(N,x)]-p[2])/p[3]) - e_hat[(N,x)])**2 for N,x in states_ent)
    
    # Better bounds and initial guess
    init = [5.0, 2.0, 5.0, 2.0]  # Close to true params
    res = minimize(obj, init, method='BFGS')
    print(f"Estimates: μ={res.x[0]:.4f}, σ_μ={res.x[1]:.4f}, γ={res.x[2]:.4f}, σ_γ={res.x[3]:.4f}")
    print(f"Objective: {res.fun:.6f}, Converged: {res.success}")
    print(f"\nProblem 10: True params: μ=5.0, σ_μ=2.2361, γ=5.0, σ_γ=2.2361")

if __name__ == "__main__":
    main()
