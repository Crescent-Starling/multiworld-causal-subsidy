"""
修复后验证：行为链 ROI 是否恢复正常
"""
import sys
sys.path.insert(0, '/Users/crescent/Desktop/multiworld-causal-subsidy')

import numpy as np
from src.simulation.mesa_agent_model import SubsidyModel

print("=" * 60)
print("修复后验证：行为链 ROI")
print("=" * 60)

# 先算理论链式概率
rates = {"clicked": 0.75, "carted": 0.65, "paid": 0.82, "redeemed": 0.70}
chain_paid = rates["clicked"] * rates["carted"] * rates["paid"]
chain_redeemed = chain_paid * rates["redeemed"]
print(f"\n理论链式概率（新默认值）:")
print(f"  clicked:  {rates['clicked']}")
print(f"  carted:   {rates['carted']} (条件概率)")
print(f"  paid:     {rates['paid']} (条件概率)")
print(f"  → 绝对 paid 概率: {chain_paid:.4f}")
print(f"  redeemed: {rates['redeemed']} (条件概率)")
print(f"  → 绝对 redeemed 概率: {chain_redeemed:.4f}")

# 跑仿真
n_agents = 500
n_rounds = 5
seed = 42

print(f"\n跑仿真: {n_agents} agents × {n_rounds} rounds...")
model = SubsidyModel(
    n_agents=n_agents,
    strategy='cognitive',
    budget_ratio=0.4,
    subsidy_amount=10.0,
    seed=seed,
    behavior_chain_enabled=True,
)

for r in range(n_rounds):
    model.step()
    res = model.round_results[-1]
    print(f"  Round {res['round']}: "
          f"ROI={res['roi']:.3f}, "
          f"ΔGTV={res['delta_gtv']:.1f}, "
          f"treated_GTV={res['treated_gtv']:.0f}, "
          f"control_GTV={res['control_gtv']:.0f}, "
          f"n_redeemed={res.get('n_redeemed',0)}, "
          f"chain_paid_rate={res.get('chain_paid_rate',0):.3f}, "
          f"chain_redeemed_rate={res.get('chain_redeemed_rate',0):.3f}")

print("\n--- 处理组 GTV 分解（最后一轮）---")
agents = list(model.agents)
treated = [a for a in agents if a._step_subsidized]
for a in treated[:5]:
    print(f"  Agent {a.agent_id}: paid={a.funnel_state['paid']}, "
          f"redeemed={a.funnel_state['redeemed']}, "
          f"step_gtv={a._step_gtv:.1f}, base_gtv={a.base_gtv:.1f}")

# 对比：单步模式 vs 行为链模式
print("\n--- 对比：单步 vs 行为链 (cognitive, 5 rounds) ---")
for bc_enabled in [False, True]:
    model2 = SubsidyModel(
        n_agents=500, strategy='cognitive',
        budget_ratio=0.4, subsidy_amount=10.0,
        seed=42, behavior_chain_enabled=bc_enabled,
    )
    for _ in range(5):
        model2.step()
    summary = model2.get_summary()
    avg_roi = summary['roi'].mean()
    avg_delta = summary['delta_gtv'].mean()
    print(f"  behavior_chain={bc_enabled}: "
          f"avg_ROI={avg_roi:.3f}, avg_ΔGTV={avg_delta:.1f}")
