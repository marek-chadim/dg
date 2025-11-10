# Empirical IO Homework: Dynamic Entry/Exit Game
## Solutions in the Style of Berry & Compiani (2020)

---

## Question 1: Nash-Equilibrium Single Period Profits

We derive the equilibrium profit function π(Nₜ, xₜ) that enters the dynamic programming problem. Consider Nₜ symmetric firms engaged in simultaneous Cournot competition with inverse demand p = 10 - Q + xₜ, zero marginal cost, and fixed cost 5.

**Equilibrium characterization.** Firm i solves:
```
max_{qᵢ} (10 - qᵢ - Q₋ᵢ + xₜ)qᵢ - 5
```

The first-order condition yields symmetric equilibrium quantity:
```
q* = (10 + xₜ)/(Nₜ + 1)
```

Substituting into the profit function:

**Proposition 1.** The single-period Nash equilibrium profit per firm is:
```
π(Nₜ, xₜ) = [(10 + xₜ)²/(Nₜ + 1)²] - 5
```

This profit function exhibits natural properties: ∂π/∂Nₜ < 0 (business stealing) and ∂π/∂xₜ > 0 (demand expansion). These comparative statics will drive entry and exit dynamics.

---

## Question 2: Cutoff Equilibrium Structure

**Proposition 2.** Under standard regularity conditions (monotonicity of payoffs in the unobservables), the Markov Perfect Equilibrium takes a cutoff form.

The economic intuition is straightforward. Consider the value function for an active firm:
```
V(Nₜ, xₜ, μᵢₜ) = max{μᵢₜ, π(Nₜ, xₜ) + βE[V(Nₜ₊₁, xₜ₊₁, μᵢₜ₊₁)|Nₜ, xₜ, stay]}
```

The exit value μᵢₜ enters additively and monotonically. As μᵢₜ increases, exit becomes strictly more attractive. This single-crossing property implies there exists a unique threshold μ̄(Nₜ, xₜ) where the firm is indifferent between continuation and exit.

**Firms with μᵢₜ < μ̄(Nₜ, xₜ)** find continuation optimal.
**Firms with μᵢₜ > μ̄(Nₜ, xₜ)** find exit optimal.

The same logic applies to potential entrants. Higher entry costs γᵢₜ make entry less attractive, implying a cutoff γ̄(Nₜ, xₜ) above which entry is unprofitable. This cutoff structure reduces the infinite-dimensional policy function to a finite set of threshold parameters—a substantial simplification for both computation and estimation.

---

## Question 3: The Integrated Value Function V̄(Nₜ, xₜ)

**Definition.** The integrated value function is:
```
V̄(Nₜ, xₜ) = ∫ V(Nₜ, xₜ, μᵢₜ) p(dμᵢₜ)
```

This object has important economic and econometric interpretations.

**Economic interpretation.** V̄(Nₜ, xₜ) measures the ex-ante expected value of an incumbent firm at state (Nₜ, xₜ), before the idiosyncratic shock μᵢₜ is realized. It represents the firm's valuation based solely on observable market structure.

**Relation to alternative-specific value functions.** This connects directly to Rust (1987). In his framework, alternative-specific value functions integrate over choice-specific unobservables:
```
v̄(a, x) = ∫ v(a, x, ε) dF(ε)
```

Similarly, our V̄(Nₜ, xₜ) integrates over the continuation-specific unobservable μᵢₜ. However, there is a subtle distinction: V̄ is defined only for Nₜ ≥ 1, as there is no incumbent value function when the market is empty.

**Computational role.** V̄(Nₜ, xₜ) serves as a sufficient statistic for continuation values in the Bellman equation. Firms need only track (Nₜ, xₜ) and their own μᵢₜ, not the full distribution of rivals' shocks. This dramatically reduces the state space.

---

## Question 4: Iterative Equilibrium Solution

We now describe the computational algorithm and provide economic intuition for each step.

### Algorithm Structure

The procedure iterates on three objects: cutoffs {μ̄(Nₜ, xₜ), γ̄(Nₜ, xₜ)} and integrated values V̄(Nₜ, xₜ). Note the dimensional reduction: only 15 relevant states exist (excluding Nₜ=0 for exit cutoffs and Nₜ=5 for entry cutoffs).

**Step 1: Initialization**
Guess initial values for the 30 relevant parameters.

**Step 2: Construct Transition Matrices**
Given cutoff policies, compute state transition probabilities. Two matrices arise for each xₜ:
- Pr(Nₜ₊₁|Nₜ, xₜ, dᵢₜ=1): conditioning on a focal incumbent staying
- Pr(Nₜ₊₁|Nₜ, xₜ, eᵢₜ=1): conditioning on a potential entrant entering

**Why are these different?** The incumbent transition conditions on Nₜ-1 other firms making exit decisions plus one potential entrant. The entrant transition conditions on Nₜ incumbents making exit decisions. The conditioning sets differ, yielding distinct transition probabilities.

**Step 3: Compute Continuation Values**
```
Ψ₁(Nₜ, xₜ) = β Σ_{Nₜ₊₁} Σ_{xₜ₊₁} V̄(Nₜ₊₁, xₜ₊₁) Pr(xₜ₊₁|xₜ) Pr(Nₜ₊₁|Nₜ, xₜ, stay)
Ψ₂(Nₜ, xₜ) = β Σ_{Nₜ₊₁} Σ_{xₜ₊₁} V̄(Nₜ₊₁, xₜ₊₁) Pr(xₜ₊₁|xₜ) Pr(Nₜ₊₁|Nₜ, xₜ, enter)
```

These represent expected discounted continuation payoffs under the current policy guess.

**Step 4: Update Cutoffs**

This step uses the indifference conditions that define equilibrium cutoffs.

**For exit:** At μ̄(Nₜ, xₜ), the firm is indifferent:
```
μ̄(Nₜ, xₜ) = π(Nₜ, xₜ) + Ψ₁(Nₜ, xₜ)
```
Left side: exit payoff. Right side: continuation payoff (current profit plus expected future value).

**For entry:** At γ̄(Nₜ, xₜ), the entrant is indifferent:
```
γ̄(Nₜ, xₜ) = Ψ₂(Nₜ, xₜ)
```
The entrant pays γ but produces only from next period forward, so current profit doesn't appear.

**Why do these equations give optimal cutoffs?** They emerge directly from the Bellman equation. A firm's optimal policy satisfies:
```
stay iff π(Nₜ, xₜ) - μᵢₜ + Ψ₁(Nₜ, xₜ) ≥ μᵢₜ
```
Rearranging: stay iff μᵢₜ ≤ π(Nₜ, xₜ) + Ψ₁(Nₜ, xₜ). The right side is precisely μ̄(Nₜ, xₜ).

**Step 5: Update Integrated Value Functions**

We compute V̄'(Nₜ, xₜ) using the max operator:
```
V̄(Nₜ, xₜ) = ∫ max{μᵢₜ, π(Nₜ, xₜ) + Ψ₁(Nₜ, xₜ)} p(dμᵢₜ)
```

With μᵢₜ ~ N(μ̄, σ²_μ), this integral has a closed form involving the standard normal cdf Φ and pdf φ:

```
V̄(Nₜ, xₜ) = [1 - Φ(z)] [μ̄ + σ_μ λ(z)] + Φ(z) [π(Nₜ, xₜ) + Ψ₁(Nₜ, xₜ)]
```

where z = [π(Nₜ, xₜ) + Ψ₁(Nₜ, xₜ) - μ̄]/σ_μ and λ(z) = φ(z)/[1-Φ(z)] is the inverse Mills ratio.

**Economic interpretation:** The first term represents the expected payoff conditional on exit (weighted by exit probability). The second term represents the expected payoff conditional on continuation (weighted by continuation probability).

**Where does this formula come from?** It follows from integrating a piecewise function over a normal distribution. The max operator creates a kink at the cutoff, yielding two regions. Standard results for truncated normal distributions (see Stokey, Lucas & Prescott 1989) deliver the closed form.

**Step 6: Iterate**
With updated {μ̄', γ̄', V̄'}, return to Step 2 until convergence.

---

## Question 5: Multiple Equilibria

To investigate equilibrium multiplicity, we employ the standard diagnostic: convergence from diverse initial conditions.

**Test design:** Solve the model from 5 qualitatively different initializations:
1. Pessimistic: μ̄ = 0, γ̄ = 0, V̄ = 0
2. Optimistic: μ̄ = 20, γ̄ = 20, V̄ = 50  
3. Baseline: μ̄ = 5, γ̄ = 5, V̄ = 10
4. Random: uniform draws on [0, 20]
5. Asymmetric: vary cutoffs by state

**Expected outcome:** If all initializations converge to the same cutoffs (within numerical tolerance), this provides evidence of a unique equilibrium. Conversely, convergence to distinct fixed points would reveal multiplicity.

**Why might uniqueness obtain?** Several features suggest a unique equilibrium:
- Monotone profit functions in state variables
- Continuous action spaces in the underlying Cournot game
- Symmetric firm structure
- Relatively small state space

However, the oligopolistic interaction and forward-looking behavior could generate coordination issues. Empirically testing convergence is more reliable than theoretical arguments in this setting.

---

## Question 6: Equilibrium Values at (N=3, x=0)

At the equilibrium (assuming uniqueness from Question 5), we report:

**Exit cutoff:** μ̄(3, 0) = [value from computation]
**Entry cutoff:** γ̄(3, 0) = [value from computation]  
**Integrated value:** V̄(3, 0) = [value from computation]

For V(3, 0, -2), we evaluate the value function at the specific shock realization μᵢₜ = -2:
```
V(3, 0, -2) = max{-2, π(3, 0) + Ψ₁(3, 0)}
```

Since μᵢₜ = -2 < μ̄(3, 0), the firm optimally stays, yielding:
```
V(3, 0, -2) = π(3, 0) - (-2) + β·E[V̄(Nₜ₊₁, xₜ₊₁)|3, 0, stay]
            = π(3, 0) + 2 + Ψ₁(3, 0)
```

---

## Question 7: Forward Simulation from Empty Market

**Simulation design:** Initialize at (N₀=0, x₀=0) and simulate T=10,000 periods using equilibrium policies.

**Implementation:**
1. At each period t, given (Nₜ, xₜ):
   - Draw μᵢₜ ~ N(5, 5) for each incumbent i=1,...,Nₜ
   - Each incumbent stays iff μᵢₜ < μ̄(Nₜ, xₜ)
   - Draw γₜ ~ N(5, 5) for potential entrant
   - Entrant enters iff γₜ < γ̄(Nₜ, xₜ)
   - Draw xₜ₊₁ according to Pr(xₜ₊₁|xₜ)
   - Update Nₜ₊₁ = (# staying incumbents) + (1 if entry, 0 otherwise)

2. Compute average: N̄ = (1/T) Σₜ Nₜ

**Expected patterns:** Given the profit function, we anticipate:
- Markets converge to a stochastic steady state
- Average firm count balances entry incentives against exit pressure
- Demand shocks xₜ create persistent fluctuations in market structure

---

## Question 8: Entry Tax Counterfactual

**Policy change:** A 5-unit entry tax shifts the entry cost distribution to γᵢₜ + 5.

**Computational approach:**
1. Re-solve for equilibrium with modified entry condition:
   ```
   γ̄*(Nₜ, xₜ) = Ψ₂(Nₜ, xₜ) - 5
   ```
2. Re-simulate from (N₀=0, x₀=0) using new equilibrium policies
3. Compare N̄* to baseline N̄

**Expected effect:** The tax reduces entry cutoffs, lowering entry probability at all states. This should decrease the long-run average number of firms. The magnitude depends on:
- Elasticity of entry decisions to cost changes
- Dynamic feedback through altered exit incentives (fewer firms → higher profits → reduced exit)

**Multiple equilibria problem:** If multiple equilibria exist, counterfactual predictions become ambiguous. Different equilibria may respond differently to the policy intervention. Without equilibrium selection criteria, we cannot provide point predictions. This highlights the critical importance of uniqueness for credible policy analysis—a central theme in Berry & Compiani (2020).

---

## Question 9: BBL-Style Forward Simulation Estimator

We now estimate the structural parameters (γ̄, σ²_γ, μ̄, σ²_μ) from simulated data, assuming knowledge of all other primitives.

### Step A: Estimate Reduced-Form Policy Functions

From the simulated dataset, estimate choice probabilities nonparametrically:
```
d̂(Nₜ, xₜ) = (# incumbents staying at (Nₜ, xₜ)) / (# total incumbents at (Nₜ, xₜ))
ê(Nₜ, xₜ) = (# entries at (Nₜ, xₜ)) / (# periods at (Nₜ, xₜ))
```

**Implementation notes:** 
- Set d̂(0, xₜ) = ê(5, xₜ) = 0 (no incumbents at N=0, no entry at N=5)
- For unvisited states, use nearest-neighbor values or fit flexible parametric model (probit/logit with high-order polynomial in Nₜ, xₜ)

These probabilities estimate the true policy functions under the data-generating parameters.

### Step B: Forward Simulation of Incumbent Value

For each state (Nₜ, xₜ) with Nₜ ≥ 1, simulate the expected discounted value for one "focal" incumbent conditional on not exiting initially.

**Simulation procedure:**
1. Fix standard normal draws {εᵢₜ} for all firms and time periods
2. For candidate parameters θ = (γ̄, σ²_γ, μ̄, σ²_μ):
   - Initialize focal firm at (Nₜ, xₜ) with known survival in period t
   - For τ = t+1, t+2, ..., T (or until focal firm exits):
     * Simulate exit decisions for other incumbents: stay if Φ((μᵢτ - μ̄)/σ_μ) ≤ d̂(Nτ, xτ)
     * Simulate entry decision: enter if Φ((γτ - γ̄)/σ_γ) ≤ ê(Nτ, xτ)
     * Update Nτ₊₁ and draw xτ₊₁
     * Record focal firm's profit π(Nτ, xτ)
     * If focal firm exits at τ, add discounted exit value βᵗ⁻ᵗμᵢτ

3. Sum discounted profits: Λ̃(Nₜ, xₜ|θ) = Σ βᵗ⁻ᵗ[π(Nτ, xτ) or μᵢτ]

**Crucial detail:** Hold the underlying standard normal draws {εᵢₜ} fixed as θ varies—exactly as in BLP simulation. This ensures continuity of simulated moments in parameters.

**Why this simulation?** We're computing the continuation value that enters the firm's stay/exit decision. At true parameters, this simulated value should rationalize the observed policy function d̂(Nₜ, xₜ).

### Step C: Average Across Simulation Draws

Repeat Step B with S=50 independent draw sequences, averaging to reduce simulation error:
```
Λ(Nₜ, xₜ|θ) = (1/S) Σₛ Λ̃ₛ(Nₜ, xₜ|θ)
```

### Step D: Incumbent Moment Conditions

The model implies that an incumbent stays when:
```
μᵢₜ < Λ(Nₜ, xₜ|θ⁰)
```
at true parameters θ⁰. Given μᵢₜ ~ N(μ̄, σ²_μ), this occurs with probability:
```
Pr(stay|Nₜ, xₜ; θ⁰) = Φ([Λ(Nₜ, xₜ|θ⁰) - μ̄]/σ_μ)
```

At the true parameters, this probability should match the data:
```
Φ([Λ(Nₜ, xₜ|θ⁰) - μ̄]/σ_μ) ≈ d̂(Nₜ, xₜ)    for all (Nₜ, xₜ) with Nₜ ≥ 1
```

This provides 15 moment conditions for incumbent decisions.

### Step E: Entrant Moment Conditions

Similarly, simulate expected discounted profits for a potential entrant at each state (Nₜ, xₜ) with Nₜ < 5. The simulation is analogous to Step B but:
- Don't include entry cost γᵢₜ in the computed value
- Start simulation from period t+1 (entrant produces from next period)

Denote the averaged simulated value Λᴱ(Nₜ, xₜ|θ). The entrant enters when:
```
Λᴱ(Nₜ, xₜ|θ⁰) > γᵢₜ
```

This occurs with probability:
```
Pr(enter|Nₜ, xₜ; θ⁰) = Φ([Λᴱ(Nₜ, xₜ|θ⁰) - γ̄]/σ_γ)
```

Matching the data:
```
Φ([Λᴱ(Nₜ, xₜ|θ⁰) - γ̄]/σ_γ) ≈ ê(Nₜ, xₜ)    for all (Nₜ, xₜ) with Nₜ < 5
```

This provides 15 moment conditions for entry decisions.

### Step F: Minimum Distance Estimation

Estimate θ̂ by minimizing the distance between model predictions and data:

```
θ̂ = argmin_θ { Σ_{Nₜ=1}⁵ Σ_{xₜ} [Φ((Λ(Nₜ,xₜ|θ) - μ̄)/σ_μ) - d̂(Nₜ,xₜ)]²
              + Σ_{Nₜ=0}⁴ Σ_{xₜ} [Φ((Λᴱ(Nₜ,xₜ|θ) - γ̄)/σ_γ) - ê(Nₜ,xₜ)]² }
```

This is a simulated minimum distance estimator with 30 moment conditions and 4 parameters—substantially overidentified.

**Implementation:** Use derivative-free optimization (Nelder-Mead or similar) since gradients are unavailable due to simulation.

---

## Question 10: Estimator Performance

**Standard errors:** As noted, conventional minimum distance standard error formulas are invalid here. They ignore:
1. First-stage estimation error in {d̂, ê}
2. Simulation error in {Λ, Λᴱ}

Correct inference requires either bootstrap or explicit accounting for both error sources—see Bajari et al. (2007) for details.

**Assessment of estimator quality:** Compare θ̂ to true data-generating values (γ̄=5, σ²_γ=5, μ̄=5, σ²_μ=5). Key diagnostics:
- Bias: Is θ̂ close to θ⁰?
- Objective function value: Small residuals suggest good fit
- Overidentification: Do all 30 moments align, or are there systematic mispredictions?

**Expected performance:** The BBL estimator should perform well in this controlled environment because:
- Large sample (T=10,000 provides rich variation across states)
- Known model primitives (demand, costs, β, transition probabilities)
- Correct specification (we're fitting the true data-generating process)

However, finite-sample bias may arise from:
- Simulation error with S=50 draws
- Nonparametric estimation of {d̂, ê} in sparse states
- Local minima in the objective function

In practice, this exercise demonstrates the computational feasibility of forward simulation methods while highlighting their dependence on accurate policy function estimation—a central theme in the two-step literature following Hotz & Miller (1993).

---

## References
This solution draws on concepts from Berry & Compiani (2020), "An Instrumental Variable Approach to Dynamic Models," particularly their discussion of policy functions, forward simulation, and the econometric endogeneity of market structure in dynamic oligopoly settings.