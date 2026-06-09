"""
完整验证：修复后行为链模式 ROI 稳定性测试
多策略对比 + 多轮
"""
import sys
sys.path.insert(0, '/Users/crescent/Desktop/multiworld-causal-subsidy')

import numpy as np
from src.simulation.mesa_agent_model import SubsidyModel

print("=" * 70)
print("修复后完整验证：行为链模式 ROI 稳定性")
print("=" * 70)

strategies = ['random', 'static', 'dynamic', 'cognitive']
n_agents = 500
n_rounds = 8
seed = 42
budget_ratio = 0.4
subsidy_amount = 10.0

results = {}
for strat in strategies:
    model = SubsidyModel(
        n_agents=n_agents,
        strategy=strat,
        budget_ratio=budget_ratio,
        subsidy_amount=subsidy_amount,
        seed=seed,
        behavior_chain_enabled=True,
        base_chain_rates={
            'clicked': 0.75, 'carted': 0.65,
            'paid': 0.82, 'redeemed': 0.70
        },
    )
    for _ in range(n_rounds):
        model.step()
    summary = model.get_summary()
    results[strat] = summary
    avg_roi = summary['roi'].mean()
    avg_delta = summary['delta_gtv'].mean()
    avg_redeem = summary['redemption_rate'].mean()
    print(f"\n{strat.upper():12s}: "
          f"avg_ROI={avg_roi:.3f}, "
          f"avg_ΔGTV={avg_delta:.0f}, "
          f"avg_redemption={avg_redeem:.3f}")
    print(f"  ROI range: [{summary['roi'].min():.3f}, {summary['roi'].max():.3f}]")
    print(f"  Redemption rate range: [{summary['redemption_rate'].min():.3f}, {summary['redemption_rate'].max():.3f}]")
    # 打印每轮 ROI
    for _, row in summary.iterrows():
        print(f"    R{int(row['round'])}: "
              f"ROI={row['roi']:.3f}, "
              f"δGTV={row['delta_gtv']:.0f}, "
              f"paid_rate={row.get('chain_paid_rate', 0):.3f}, "
              f"redeem_rate={row.get('chain_redeemed_rate', 0):.3f}")

# 检查所有轮次 ROI 是否都为正
print("\n" + "=" * 70)
print("ROI 正负检查:")
all_positive = True
for strat, summary in results.items():
    neg_rounds = summary[summary['roi'] < 0]
    if len(neg_rounds) > 0:
        print(f"  {strat}: {len(neg_rounds)} 轮 ROI 为负!")
        all_positive = False
    else:
        print(f"  {strat}: 所有轮次 ROI >= 0 ✓")
print("=" * 70)
