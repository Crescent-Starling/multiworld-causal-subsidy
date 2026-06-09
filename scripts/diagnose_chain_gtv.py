"""
诊断行为链 ROI 偏低的根因
对比处理组和对照组在各漏斗状态下的 GTV 期望
"""
import sys
sys.path.insert(0, '/Users/crescent/Desktop/multiworld-causal-subsidy')

import numpy as np
import pandas as pd

from src.simulation.mesa_agent_model import SubsidyModel

# 用较大的Agent数和较多轮次，获得稳定统计
n_agents = 500
n_rounds = 3
seed = 42

print("=" * 60)
print("诊断：行为链 GTV 计算逻辑")
print("=" * 60)

model = SubsidyModel(
    n_agents=n_agents,
    strategy='cognitive',
    budget_ratio=0.4,
    subsidy_amount=10.0,
    seed=seed,
    behavior_chain_enabled=True,
    base_chain_rates={
        'clicked': 0.55, 'carted': 0.33,
        'paid': 0.42, 'redeemed': 0.64
    },
)

# 手动跑1轮，然后分析Agent级别数据
model.step()

agents = list(model.agents)
treated = [a for a in agents if a._step_subsidized]
control  = [a for a in agents if not a._step_subsidized]

print(f"\n处理组: {len(treated)} 人, 对照组: {len(control)} 人")

# ---- 处理组 GTV 分解 ----
print("\n--- 处理组 GTV 分解 ---")
paid_t = [a for a in treated if a.funnel_state.get('paid', False)]
not_paid_t = [a for a in treated if not a.funnel_state.get('paid', False)]
redeemed_t = [a for a in treated if a.funnel_state.get('redeemed', False)]

print(f"处理组 paid={len(paid_t)}, not_paid={len(not_paid_t)}, redeemed={len(redeemed_t)}")
if paid_t:
    print(f"  已付款处理组 avg_GTV: {np.mean([a._step_gtv for a in paid_t]):.2f}")
if not_paid_t:
    print(f"  未付款处理组 avg_GTV: {np.mean([a._step_gtv for a in not_paid_t]):.2f}")
if redeemed_t:
    print(f"  已核销处理组 avg_GTV: {np.mean([a._step_gtv for a in redeemed_t]):.2f}")

# ---- 对照组 GTV 分解 ----
print("\n--- 对照组 GTV 分解 ---")
# 对照组全部走 _step_gtv = base_gtv * 0.3
ctrl_gtvs = [a._step_gtv for a in control]
print(f"对照组 avg_GTV: {np.mean(ctrl_gtvs):.2f}")
print(f"对照组 GTV 公式: base_gtv * 0.3")

# ---- 理论 GTV 期望 ----
print("\n--- 理论 GTV 期望（处理组）---")
# base_chain_rates 是各步条件概率
# 实际链式概率 = clicked * carted * paid * redeemed
p_click = 0.55
p_cart  = 0.33
p_pay   = 0.42
p_redeem = 0.64
chain_paid_rate = p_click * p_cart * p_pay
chain_redeem_rate = chain_paid_rate * p_redeem
print(f"链式 paid 概率: {p_click} * {p_cart} * {p_pay} = {chain_paid_rate:.4f}")
print(f"链式 redeemed 概率: {chain_paid_rate:.4f} * {p_redeem} = {chain_redeem_rate:.4f}")

# 处理组 GTV 期望
# paid 用户: GTV = base_gtv + subsidy*0.5*(redeemed?1:0)
# not_paid 用户: GTV = base_gtv * 0.3
avg_base_gtv = np.mean([a.base_gtv for a in agents])
avg_subsidy = np.mean([a._step_subsidy_amount for a in treated]) if treated else 0
print(f"\navg_base_gtv: {avg_base_gtv:.2f}")
print(f"avg_subsidy (处理组): {avg_subsidy:.2f}")

# 期望 GTV（处理组，按链式概率）
e_paid     = chain_paid_rate
e_redeemed = chain_redeem_rate
e_not_paid = 1 - e_paid

# 注意：redeemed 是 paid 的子集
gtv_if_paid_redeemed   = avg_base_gtv + avg_subsidy * 0.5
gtv_if_paid_not_redeemed = avg_base_gtv   # paid 但未核销：代码第377行 subsidized 时 paid=True 就有 base_gtv + subsidy*0.5*0
# 等等，看代码第376-379行：
#   if self.funnel_state["paid"]:
#       self._step_gtv = self.base_gtv + self.subsidy_amount * 0.5 * int(self.redeemed)
#   else:
#       self._step_gtv = self.base_gtv * 0.3
# 所以 paid=True, redeemed=False 时: GTV = base_gtv + 0 = base_gtv
# paid=True, redeemed=True 时:  GTV = base_gtv + subsidy*0.5

gtv_paid_redeemed   = avg_base_gtv + avg_subsidy * 0.5
gtv_paid_not_redeemed = avg_base_gtv
gtv_not_paid        = avg_base_gtv * 0.3

e_gt = (e_redeemed * gtv_paid_redeemed +
        (e_paid - e_redeemed) * gtv_paid_not_redeemed +
        e_not_paid * gtv_not_paid)
print(f"\n处理组期望GTV/人: {e_gt:.2f}")
print(f"  其中: {e_redeemed:.4f} 概率全额GTV={gtv_paid_redeemed:.2f}")
print(f"         {(e_paid - e_redeemed):.4f} 概率 base_gtv={gtv_paid_not_redeemed:.2f}")
print(f"         {e_not_paid:.4f} 概率 0.3*base={gtv_not_paid:.2f}")

# 对照组 GTV 期望
e_gc = avg_base_gtv * 0.3
print(f"\n对照组期望GTV/人: {e_gc:.2f} (base_gtv*0.3)")

# delta_GTV 期望
e_delta = e_gt - e_gc
print(f"\n期望 delta_GTV/人: {e_delta:.2f}")
print(f"期望 ROI = (delta_GTV - subsidy) / subsidy = ({e_delta:.2f} - {avg_subsidy:.2f}) / {avg_subsidy:.2f} = {(e_delta - avg_subsidy)/avg_subsidy:.4f}")

# ---- 实际观察 ----
print("\n--- 实际观察（本轮）---")
result = model.round_results[-1]
for k in ['roi', 'delta_gtv', 'total_subsidy', 'n_treated', 'n_redeemed',
          'treated_gtv', 'control_gtv', 'chain_clicked_rate',
          'chain_carted_rate', 'chain_paid_rate', 'chain_redeemed_rate']:
    if k in result:
        print(f"  {k}: {result[k]}")

# ---- 关键诊断：处理组 vs 对照组人均GTV ----
avg_gt_treated = result['treated_gtv'] / max(result['n_treated'], 1)
avg_gt_control = result['control_gtv'] / max(len(control), 1)
print(f"\n处理组人均GTV: {avg_gt_treated:.2f}")
print(f"对照组人均GTV: {avg_gt_control:.2f}")
print(f"人均delta: {avg_gt_treated - avg_gt_control:.2f}")
