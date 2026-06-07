# AI Subsidy Simulation System

> A simulation sandbox for subsidy policy evaluation using causal inference and agent-based modeling.

## Background

This project originated from the **6th Meituan Business Analysis Elite Competition** (美团商务分析精英大赛), where the task was to build user-level simulation capabilities for coupon subsidy strategy optimization.

Building upon the competition work, this repository presents a **complete redesign** with advanced frameworks:
- **Causal Inference**: CausalML (Uber) + DoWhy (Microsoft) for robust effect estimation
- **Agent-Based Modeling**: Mesa framework for formal ABM architecture
- **Behavioral Economics**: Prospect theory + Mental accounting (Phase 1 completed)
- **LLM Agent**: AgentSociety (Tsinghua) for LLM-native simulation

## Data Source

The data used in this project comes from the **Meituan competition dataset** (publicly available, anonymized). 
- **User profile data**: 100,000 users (anonymized)
- **Behavior sequence data**: User activity logs
- **Order data**: 315,000+ transaction records
- **Coupon data**: 2,312,000+ coupon records

> **Note**: All data has been anonymized by Meituan. This project uses the data for research and educational purposes only.

For reproducibility, we also provide **synthetically generated data** (see `data/synthetic/`).

## Tech Stack

### Causal Inference
- `causalml` (Uber): Industrial-grade Uplift modeling
- `dowhy` (Microsoft): Causal graph + refutation
- Custom implementation: T/X/DR-Learner, G-Net, MSM

### ABM Simulation
- `mesa`: Formal ABM framework
- `networkx`: Social network modeling

### LLM Agent
- `agentsociety` (Tsinghua): LLM-native ABM (planned)

### Evaluation
- Bootstrap CI
- PoliSim PN/PS causal probability (planned)

## Project Structure

```
src/
├── modeling/           # Causal inference models
│   ├── causalml_wrapper.py
│   ├── dowhy_causal_graph.py
│   └── psm_matcher.py
├── simulation/        # ABM simulation
│   ├── mesa_agent_model.py
│   ├── network_contagion.py
│   └── cognitive_agent_theory.py
├── evaluation/        # Evaluation metrics
│   ├── causal_sim_eval.py
│   └── multi_world_robustness.py
└── features/          # Feature engineering

scripts/               # Run scripts
docs/                  # Documentation
data/                  # Data (competition + synthetic)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run causal inference comparison
python scripts/run_causalml_comparison.py

# Run Mesa ABM simulation
python scripts/run_mesa_simulation.py
```

## Methodology

See `docs/methodology.md` for detailed methodology and comparison with existing frameworks.

## License

MIT License (for code only). Data usage must comply with Meituan competition terms.
