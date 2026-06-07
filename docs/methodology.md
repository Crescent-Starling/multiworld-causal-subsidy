# Methodology

## Overview

This project combines **causal inference** with **AI-driven agent-based simulation** to evaluate coupon subsidy policies. The core innovation is a **multi-world evaluation framework** that runs parallel simulations under different subsidy strategies, enabling robust counterfactual comparison and risk quantification.

---

## 1. Causal Inference Pipeline

### 1.1 Meta-Learner Uplift Modeling (CausalML)

We use Uber's CausalML library to estimate heterogeneous treatment effects via four Meta-Learner approaches:

| Learner | Method | Strength |
|---------|--------|----------|
| **T-Learner** | Two separate models for treatment/control | Simple, captures non-linearity |
| **X-Learner** | T-Learner + propensity-corrected counterfactual | Better with imbalanced groups |
| **DR-Learner** | Doubly robust: combines outcome & propensity models | Robust to model misspecification |
| **S-Learner** | Single model with treatment as feature | Avoids regularization-induced bias |

**Reference**: Künzel, S. R., et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156–4165.

### 1.2 Causal Graph & Identification (DoWhy)

Microsoft's DoWhy framework provides:

1. **Causal DAG construction** — Explicitly encodes assumed causal relationships among variables
2. **Identification** — Determines if the causal effect is identifiable from observational data
3. **Estimation** — Uses backdoor adjustment, IPW, or regression-based methods
4. **Refutation** — Validates robustness via:
   - Random common cause (placebo test)
   - Data subset validation
   - Unobserved confounder simulation
   - Dummy outcome replacement

**Reference**: Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference.

### 1.3 Propensity Score Matching (PSM)

Three matching methods are implemented:

- **Nearest neighbor**: Match each treated unit to the closest control by propensity score
- **Caliper**: Only match if propensity score difference < threshold (prevents poor matches)
- **Optimal**: Hungarian algorithm minimizes total global distance (via `scipy.optimize.linear_sum_assignment`)

Match quality is assessed via **Standardized Mean Difference (SMD)** across all covariates (SMD < 0.1 indicates good balance).

**Reference**: Rubin, D. B. (1973). Matching to remove bias in observational studies. *Biometrics*, 29(1), 159–183.

---

## 2. AI-Driven Agent-Based Simulation

### 2.1 Behavioral Economics Foundation

Our agents are grounded in three behavioral economics theories:

#### Prospect Theory (Kahneman & Tversky, 1979)
- Value function: `v(x) = x^α` for gains, `v(x) = -λ(-x)^α` for losses
- Parameters: α = 0.88 (diminishing sensitivity), λ = 2.25 (loss aversion)
- Key insight: Users overweight losses relative to equivalent gains

#### Mental Accounting (Thaler, 1985)
- Different account types have different reference point update rates:
  - **Windfall** (η=0.10): Quick adaptation, treat subsidies as "free money"
  - **Income** (η=0.35): Slow adaptation, treat subsidies as earned income
  - **Price-sensitive**: Mixed, responds strongly to discount framing
  - **Deal-seeker**: Actively seeks and redeems coupons
- Reference point update: `R_{t+1} = R_t + η × (subsidy - R_t)`

#### Bounded Rationality (Simon, 1955)
- Finite decision precision: agents don't perfectly optimize
- Discount factor reduces effective rationality by noise

### 2.2 Mesa ABM Framework

Built on Mesa 3.x, the ABM simulation includes:

- **SubsidyAgent**: Each agent has a cognitive profile (price sensitivity, mental account type, fatigue level)
- **SubsidyModel**: Manages agent population, budget constraints, and strategy execution
- **MultiWorldModel**: Core innovation — runs parallel worlds with different strategies

**Strategies**:
| Strategy | Description |
|----------|-------------|
| Random | Uniform random subsidy allocation |
| Static | Fixed subsidy amount to all eligible users |
| Dynamic | Subsidy adjusted based on user response history |
| Cognitive | Theory-driven: prospect theory + mental accounting |

### 2.3 Multi-World Evaluation

The multi-world framework addresses a fundamental challenge in policy evaluation: **decoupling assumption risk from random noise**.

```
World A (static)     ────→  Results_A
World B (dynamic)    ────→  Results_B
World C (cognitive)  ────→  Results_C

Comparison: ΔROI, ΔGTV, robustness metrics
```

By running the same agent population under different policy assumptions, we can:
1. Quantify how much outcome variance is due to policy choice vs. stochastic noise
2. Identify which strategies are robust to parameter perturbation
3. Provide confidence intervals that account for model uncertainty

---

## 3. Social Network Contagion

### 3.1 Network Construction

Four methods for constructing social networks:

| Method | Description | Use Case |
|--------|-------------|----------|
| **Attribute similarity** | Edge weight = cosine similarity of user attributes | Demographic homophily |
| **Barabási–Albert** | Preferential attachment (scale-free) | Hub-dominated networks |
| **Watts–Strogatz** | Small-world with tunable clustering | Local clustering + shortcuts |
| **POI co-occurrence** | Users sharing POI visits are connected | Behavioral similarity |

### 3.2 SIR Contagion Model

Coupon redemption behavior spreads through the network via an SIR model:
- **S (Susceptible)**: Hasn't been exposed to coupon info from neighbors
- **I (Infected)**: Actively considering/using coupons (can influence neighbors)
- **R (Recovered)**: No longer spreading (immune / habituated)

### 3.3 Social Effect Estimation

Following Christakis & Fowler (2007), we estimate social contagion effects by comparing:
- **Original network** cascade size
- **Configuration model** (degree-preserving random graph) cascade size
- **Difference** = social effect attributable to network structure (not degree distribution)

---

## 4. Evaluation Metrics

### 4.1 Bootstrap Confidence Intervals
Efron's bootstrap with 1,000+ resamples for non-parametric CI estimation.

### 4.2 E-Value (VanderWeele & Ding, 2017)
The minimum strength of association that an unmeasured confounder would need with both treatment and outcome to explain away the observed effect.

```
E-value = RR + √(RR × (RR - 1))  for RR ≥ 1
```

### 4.3 Multi-World Robustness
- Strategy ranking stability across bootstrap resamples
- Jensen-Shannon divergence between strategy outcome distributions
- Policy regret under worst-case scenarios

---

## 5. Comparison with Existing Frameworks

| Feature | This Project | meituan-subsidy-efficiency | Standard A/B Testing |
|---------|-------------|---------------------------|---------------------|
| Causal identification | DAG-based (DoWhy) | L0-L4 heuristic levels | Randomized experiment |
| Heterogeneous effects | T/X/DR/S-Learner | L2 stratification | None (ATE only) |
| Agent simulation | Theory-driven (Mesa) | Rule-based | None |
| Network effects | SIR contagion | Not modeled | Not modeled |
| Robustness | Multi-world + E-value | Bootstrap only | p-value |
| Behavioral grounding | Prospect theory + Mental accounting | Ad-hoc rules | None |

---

## References

- Christakis, N. A., & Fowler, J. H. (2007). The spread of obesity in a large social network over 32 years. *New England Journal of Medicine*, 357(4), 370–379.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.
- Künzel, S. R., et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156–4165.
- Rubin, D. B. (1973). Matching to remove bias in observational studies. *Biometrics*, 29(1), 159–183.
- Simon, H. A. (1955). A behavioral model of rational choice. *The Quarterly Journal of Economics*, 69(1), 99–118.
- Thaler, R. H. (1985). Mental accounting and consumer choice. *Marketing Science*, 4(3), 199–214.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268–274.
