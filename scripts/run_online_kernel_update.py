"""
在线核更新 + 多平行世界仿真示例

演示：
1. 创建多平行世界（认知策略 vs 随机策略）
2. 每轮结束后用KernelOnlineUpdater更新Agent核参数
3. 用MultiWorldKernelTracker追踪核参数演化轨迹
4. 对比两种策略下，用户参数的不同演化路径
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.mesa_agent_model import (
    MultiWorldModel,
    StrategyType,
)
from src.simulation.kernel_online_upater import (
    KernelOnlineUpdater,
    MultiWorldKernelTracker,
)
from src.simulation.behavioral_kernel import (
    UserKernel,
    ChainStep,
)


# =========================================================================
# 辅助函数：为Agent初始化UserKernel
# =========================================================================

def init_kernels_for_world(model: "SubsidyModel") -> None:
    """
    为世界中所有Agent初始化UserKernel（如果还没有）

    模拟真实场景：用户逐渐积累行为数据，
    平台从行为链中提取其个性化核参数。
    """
    rng = model.rng

    for agent in model.agents:
        if agent.user_kernel is not None:
            continue  # 已有核，跳过

        # 从Agent当前参数初始化一个"粗糙核"
        theta = {}
        beta = {}
        for step in ChainStep.funnel_steps():
            # 用模型base_chain_rates + 小的个体差异
            base = model.base_chain_rates.get(step.value, 0.5)
            theta[step.value] = float(np.clip(
                base + rng.normal(0, 0.05), 0.01, 0.99
            ))

            # 初始beta为小随机数（尚未观测到补贴效应）
            beta[step.value] = float(rng.normal(0, 0.02))

        gamma = {
            "alpha": agent.alpha,
            "lambda_": agent.lambda_,
            "fatigue_rate": 0.1,
            "account_eta": 0.2,
        }

        agent.user_kernel = UserKernel(
            user_id=str(agent.agent_id),
            theta=theta,
            beta=beta,
            gamma=gamma,
        )


# =========================================================================
# 主实验
# =========================================================================

def main():
    print("=" * 60)
    print("在线核更新 + 多平行世界仿真 示例")
    print("=" * 60)

    N_AGENTS = 200
    N_ROUNDS = 20
    SEED = 42

    # ---- 1. 创建多平行世界 ----
    print("\n[1] 创建多平行世界...")
    mwm = MultiWorldModel(
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
        seed=SEED,
    )
    print(f"    共享Agent配置数: {len(mwm._agent_configs)}")

    # ---- 2. 创建在线更新器和追踪器 ----
    print("\n[2] 初始化KernelOnlineUpdater和MultiWorldKernelTracker...")
    updater = KernelOnlineUpdater(
        learning_rate_theta=0.15,
        learning_rate_beta=0.08,
        forgetting_factor=0.95,
        enable_dynamic_beta=True,
        enable_dynamic_gamma=True,
    )
    tracker = MultiWorldKernelTracker()
    print("    完成。")

    # ---- 3. 运行世界A：认知策略（会塑造用户参数） ----
    print("\n[3] 运行世界A：认知策略 (cognitive) ...")
    world_a = mwm.add_world(
        world_name="cognitive",
        strategy="cognitive",
        budget_ratio=0.3,
        subsidy_amount=10.0,
        kernel_updater=updater,
        kernel_tracker=tracker,
    )
    # 为Agent初始化核（如果SubsidyModel没有自动做）
    # 注意：需要在世界创建后、运行前初始化
    print(f"    世界A ROI: {world_a.final_metrics.get('avg_roi', 0):.4f}")
    print(f"    世界A 累计ΔGTV: {world_a.final_metrics.get('cumulative_delta_gtv', 0):.2f}")

    # ---- 4. 运行世界B：随机策略（对照） ----
    print("\n[4] 运行世界B：随机策略 (random) ...")
    world_b = mwm.add_world(
        world_name="random",
        strategy="random",
        budget_ratio=0.3,
        subsidy_amount=10.0,
        kernel_updater=updater,
        kernel_tracker=tracker,
    )
    print(f"    世界B ROI: {world_b.final_metrics.get('avg_roi', 0):.4f}")
    print(f"    世界B 累计ΔGTV: {world_b.final_metrics.get('cumulative_delta_gtv', 0):.2f}")

    # ---- 5. 分析核参数演化 ----
    print("\n[5] 分析核参数演化轨迹...")
    print("-" * 40)

    # 5.1 查看某个Agent的跨世界对比
    SAMPLE_AGENT = 0
    traj_df = tracker.get_world_comparison_df(SAMPLE_AGENT)
    if not traj_df.empty:
        print(f"\n  Agent {SAMPLE_AGENT} 的核参数跨世界对比（前5轮）:")
        print(traj_df.head(10).to_string(index=False))

    # 5.2 疲劳累积对比
    fatigue_df = tracker.analyze_fatigue_accumulation()
    if not fatigue_df.empty:
        # 计算每个世界每轮的平局疲劳度
        avg_fatigue = (
            fatigue_df.groupby(["world", "round"])["fatigue"]
            .mean()
            .reset_index()
        )
        print(f"\n  各世界平均疲劳度演化（每5轮抽样）:")
        sampled = avg_fatigue[avg_fatigue["round"] % 5 == 0]
        print(sampled.to_string(index=False))

    # 5.3 theta_clicked 演化对比
    print(f"\n  theta_clicked 跨世界演化（Agent {SAMPLE_AGENT}）:")
    if not traj_df.empty:
        theta_evol = traj_df[["world", "round", "theta_clicked", "fatigue"]].copy()
        print(theta_evol.to_string(index=False))

    # ---- 6. 策略对比总结 ----
    print("\n[6] 策略对比总结")
    print("-" * 40)
    comparison = mwm.compare_worlds()
    print(comparison.to_string(index=False))

    print("\n" + "=" * 60)
    print("实验完成。核参数演化数据已记录到MultiWorldKernelTracker。")
    print("=" * 60)

    return mwm, updater, tracker


if __name__ == "__main__":
    main()
