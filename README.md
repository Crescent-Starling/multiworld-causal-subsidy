# MultiWorld Causal Subsidy

> Causal inference + AI-driven simulation for coupon subsidy policy evaluation. Originated from the 6th Meituan Business Analysis Elite Competition, this project combines multi-world evaluation, uplift modeling, and behavioral-economics-grounded agent simulation for robust offline policy optimization.

**简体中文**：基于因果推断与AI仿真的优惠券补贴策略评估系统。起源于第六届美团商务分析精英大赛，集成多平行世界评估框架、Uplift 建模与行为经济学驱动的 Agent 仿真，支持鲁棒的离线策略优化。

---

## Highlights

- **Multi-World Evaluation** — Parallel simulation worlds with different subsidy strategies, enabling robust counterfactual comparison and risk quantification
- **Causal Inference Pipeline** — CausalML (T/X/DR/S-Learner) + DoWhy (causal graph + refutation) for rigorous treatment effect estimation
- **Theory-Driven AI Agents** — Mesa ABM with Prospect Theory (Kahneman & Tversky, 1979) and Mental Accounting (Thaler, 1985) agents, beyond ad-hoc heuristics
- **Social Network Contagion** — NetworkX-powered SIR contagion modeling for spillover effect estimation
- **Meituan Real-World Data** — Validated on Meituan coupon order data and user behavior sequences from the competition dataset (anonymized)

## Background

This project originated from the **6th Meituan Business Analysis Elite Competition** (美团商务分析精英大赛), where the task was to build user-level simulation capabilities for coupon subsidy strategy optimization.

Building upon the competition work, this repository presents a **complete redesign** with advanced frameworks:
- **Causal Inference**: CausalML (Uber) + DoWhy (Microsoft) for robust effect estimation
- **Agent-Based Modeling**: Mesa framework for formal ABM architecture
- **Behavioral Economics**: Prospect theory + Mental accounting + Bounded rationality
- **Network Effects**: Social contagion via NetworkX
- **LLM Agent**: AgentSociety (Tsinghua) for LLM-native simulation

## Data Source

The data used in this project comes from the **Meituan competition dataset** (publicly available, anonymized).
- **Coupon order data** (`神券订单数据样例.xlsx`): 950 order records with coupon type (free/paid), subsidy amount, inflation flag, and POI category
- **User behavior sequences** (`用户行为序列.xlsx`): 478 session-level behavior records (clicks, views, purchases) for sampled users

> **Note**: All data has been anonymized by Meituan. This project uses the data for research and educational purposes only.

For reproducibility, we also provide **synthetically generated data** (see `data/synthetic/`).

## Tech Stack

### Causal Inference
- `causalml` (Uber): Industrial-grade Uplift modeling (T/X/DR/S-Learner)
- `dowhy` (Microsoft): Causal graph + refutation
- Custom implementation: G-Net, MSM, PSM

### AI Simulation
- `mesa`: Formal ABM framework with theory-driven cognitive agents
- `networkx`: Social network modeling + contagion
- `agentsociety` (Tsinghua): LLM-native ABM (planned)

### Evaluation
- Bootstrap CI, E-value sensitivity analysis
- Multi-world robustness comparison

## Project Structure

```
src/
├── modeling/           # Causal inference models
│   ├── causalml_wrapper.py
│   ├── dowhy_causal_graph.py
│   └── psm_matcher.py
├── simulation/        # AI-driven ABM simulation
│   ├── mesa_agent_model.py
│   ├── network_contagion.py
│   ├── cognitive_agent_theory.py
│   └── llm_agent.py
├── evaluation/        # Evaluation metrics
│   └── metrics.py
└── features/          # Feature engineering

scripts/               # Run scripts
docs/                  # Documentation
data/                  # Data (Meituan anonymized + synthetic)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all demos
python scripts/run_all_demos.py

# Run causal inference comparison
python scripts/run_causalml_comparison.py

# Run Mesa ABM simulation
python scripts/run_mesa_simulation.py
```

## Methodology

See `docs/methodology.md` for detailed methodology and comparison with existing frameworks.

## License

MIT License (for code only). Data usage must comply with Meituan competition terms.
