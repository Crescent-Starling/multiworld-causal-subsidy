"""
行为链核提取验证与演示脚本
==========================

验证流程：
1. 生成具有已知 ground-truth 参数的合成行为链数据
2. 用 BehavioralKernelExtractor 提取核参数
3. 对比提取参数与 ground-truth（评估收缩估计精度）
4. 验证差分隐私噪声的影响
5. 用核驱动 Agent 运行仿真，对比 ROI

运行：
  python scripts/validate_behavioral_kernel.py
  python scripts/validate_behavioral_kernel.py --n-users 200 --epsilon 1.0
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.behavioral_kernel import (
    BehavioralKernelExtractor,
    KernelPopulationSampler,
    UserKernel,
    ChainStep,
    kernel_modulated_prob,
)
from src.simulation.mesa_agent_model import (
    SubsidyAgent,
    SubsidyModel,
    MultiWorldModel,
    AgentConfig,
)


# ===========================================================================
# Step 1: 合成数据生成器
# ===========================================================================

def generate_synthetic_chains(
    n_users: int = 100,
    n_sessions_per_user: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, UserKernel]]:
    """
    生成合成行为链数据，返回 (DataFrame, ground_truth_kernels)

    数据格式：每个用户有 n_sessions_per_user 个会话（session），
    每个会话中用户沿漏斗 browse→click→cart→pay→redeem 前进，
    每步有条件概率决定是否继续。

    theta 的语义：P(step | prev_step=True)，即条件转换率。
    """
    rng = np.random.RandomState(seed)

    # 总体参数（条件概率语义）
    pop_theta = {"clicked": 0.75, "carted": 0.65, "paid": 0.82, "redeemed": 0.70}
    pop_beta = {"clicked": 0.10, "carted": 0.05, "paid": 0.08, "redeemed": 0.15}

    rows = []
    ground_truth = {}

    for i in range(n_users):
        # 个体化参数（从总体分布中采样）
        theta_i = {}
        for step, mu in pop_theta.items():
            a = mu * 20  # 控制方差
            b = (1 - mu) * 20
            theta_i[step] = float(np.clip(rng.beta(a, b), 0.1, 0.95))

        beta_i = {}
        for step, mu in pop_beta.items():
            beta_i[step] = float(rng.normal(mu, 0.03))

        gamma_i = {
            "alpha": float(rng.normal(0.88, 0.05)),
            "lambda_": float(rng.normal(2.25, 0.2)),
            "fatigue_rate": float(rng.uniform(0.05, 0.25)),
            "account_eta": float(rng.uniform(0.15, 0.45)),
            "price_sensitivity": float(rng.beta(2, 5)),
        }

        ground_truth[f"user_{i}"] = UserKernel(
            user_id=f"user_{i}",
            theta=theta_i,
            beta=beta_i,
            gamma=gamma_i,
        )

        # 生成该用户的多个会话
        for j in range(n_sessions_per_user):
            is_subsidized = rng.random() < 0.4  # 40% 概率受补贴
            subsidy = rng.uniform(5, 20) if is_subsidized else 0.0

            # 漏斗序贯决策：每步基于条件概率决定是否继续
            steps_reached = ["browsed"]  # 始终浏览
            for step in ChainStep.funnel_steps():
                step_name = step.value
                base_p = theta_i[step_name]
                if is_subsidized:
                    # 补贴提升转换率
                    p = float(np.clip(base_p + beta_i[step_name] * subsidy / 10.0, 0.01, 0.99))
                else:
                    p = base_p * 0.7  # 无补贴时率较低

                if rng.random() < p:
                    steps_reached.append(step_name)
                else:
                    break  # 漏斗中断

            # 记录达成的每个步骤
            for step_name in steps_reached:
                rows.append({
                    "user_id": f"user_{i}",
                    "action": step_name,
                    "subsidy_amount": subsidy,
                    "session_id": j,
                })

    df = pd.DataFrame(rows)
    return df, ground_truth


# ===========================================================================
# Step 2: 评估核提取精度
# ===========================================================================

def evaluate_extraction_accuracy(
    extracted: list[UserKernel],
    ground_truth: dict[str, UserKernel],
) -> dict:
    """评估提取参数与 ground-truth 的偏差"""
    steps = [s.value for s in ChainStep.funnel_steps()]
    theta_errors = {step: [] for step in steps}
    beta_errors = {step: [] for step in steps}

    for kernel in extracted:
        gt = ground_truth.get(kernel.user_id)
        if gt is None:
            continue
        for step in steps:
            theta_errors[step].append(abs(kernel.get_theta(step) - gt.get_theta(step)))
            beta_errors[step].append(abs(kernel.get_beta(step) - gt.get_beta(step)))

    result = {"theta_mae": {}, "beta_mae": {}}
    for step in steps:
        result["theta_mae"][step] = float(np.mean(theta_errors[step])) if theta_errors[step] else 0.0
        result["beta_mae"][step] = float(np.mean(beta_errors[step])) if beta_errors[step] else 0.0

    result["overall_theta_mae"] = float(np.mean([result["theta_mae"][s] for s in steps]))
    result["overall_beta_mae"] = float(np.mean([result["beta_mae"][s] for s in steps]))

    return result


# ===========================================================================
# Step 3: 差分隐私噪声影响评估
# ===========================================================================

def evaluate_dp_impact(
    kernels_no_dp: list[UserKernel],
    kernels_dp: list[UserKernel],
) -> dict:
    """评估 DP 噪声对参数的影响"""
    steps = [s.value for s in ChainStep.funnel_steps()]

    theta_perturbations = {step: [] for step in steps}
    for k1, k2 in zip(kernels_no_dp, kernels_dp):
        for step in steps:
            theta_perturbations[step].append(abs(k1.get_theta(step) - k2.get_theta(step)))

    return {
        "theta_mean_perturbation": {
            step: float(np.mean(theta_perturbations[step]))
            for step in steps
        },
        "overall_mean_perturbation": float(np.mean([
            np.mean(theta_perturbations[step]) for step in steps
        ])),
    }


# ===========================================================================
# Step 4: 核驱动仿真对比
# ===========================================================================

def run_kernel_simulation(
    kernels: list[UserKernel],
    n_rounds: int = 30,
    seed: int = 42,
) -> dict:
    """用核驱动 Agent 运行仿真，与基线对比"""
    # 生成 AgentConfig
    rng = np.random.RandomState(seed)
    configs = []
    for i, kernel in enumerate(kernels[:500]):  # 最多 500 个
        ps = kernel.get_gamma("price_sensitivity", 0.5)
        income = int(np.clip(5 - ps * 4, 1, 5))
        configs.append(AgentConfig(
            agent_id=i,
            price_sensitivity=ps,
            income_level=income,
            city_tier=3,
            base_gtv=rng.lognormal(3.5, 0.8),
            behavior_chain_enabled=True,
            user_kernel=kernel,
        ))

    # 基线配置（无核参数）
    rng2 = np.random.RandomState(seed)
    baseline_configs = []
    for i in range(len(configs)):
        ps = rng2.beta(2, 5)
        income = int(rng2.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.35, 0.25, 0.1]))
        baseline_configs.append(AgentConfig(
            agent_id=i,
            price_sensitivity=ps,
            income_level=income,
            city_tier=int(rng2.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.30, 0.20, 0.10])),
            base_gtv=rng2.lognormal(3.5, 0.8),
            behavior_chain_enabled=True,
            user_kernel=None,
        ))

    # 运行仿真
    def run_single(configs_list, label):
        model = SubsidyModel(
            n_agents=len(configs_list),
            strategy="cognitive",
            budget_ratio=0.3,
            subsidy_amount=10.0,
            seed=seed,
            agent_configs=configs_list,
            behavior_chain_enabled=True,
        )
        result = model.run(n_rounds=n_rounds)
        return {
            "label": label,
            "avg_roi": result.final_metrics["avg_roi"],
            "cumulative_delta_gtv": result.final_metrics["cumulative_delta_gtv"],
            "avg_redemption_rate": result.final_metrics["avg_redemption_rate"],
        }

    kernel_result = run_single(configs, "kernel-driven")
    baseline_result = run_single(baseline_configs, "baseline (no kernel)")

    return {"kernel_driven": kernel_result, "baseline": baseline_result}


# ===========================================================================
# Step 5: 种群合成采样验证
# ===========================================================================

def validate_population_synthesis(kernels: list[UserKernel], seed: int = 42) -> dict:
    """验证种群合成采样器"""
    sampler = KernelPopulationSampler(random_state=seed)
    sampler.fit(kernels)

    # 采样新种群
    synthetic = sampler.sample(100)

    # 比较原始核与合成核的分布
    steps = [s.value for s in ChainStep.funnel_steps()]

    comparison = {"original_mean": {}, "synthetic_mean": {}, "ks_test": {}}
    for step in steps:
        orig_vals = [k.get_theta(step) for k in kernels]
        synth_vals = [k.get_theta(step) for k in synthetic]

        comparison["original_mean"][step] = float(np.mean(orig_vals))
        comparison["synthetic_mean"][step] = float(np.mean(synth_vals))

        # KS 检验（分布相似性）
        from scipy.stats import ks_2samp
        try:
            stat, pval = ks_2samp(orig_vals, synth_vals)
            comparison["ks_test"][step] = {"statistic": float(stat), "p_value": float(pval)}
        except Exception:
            comparison["ks_test"][step] = {"statistic": float("nan"), "p_value": float("nan")}

    comparison["n_original"] = len(kernels)
    comparison["n_synthetic"] = len(synthetic)
    comparison["distribution_stats"] = sampler.get_distribution_stats()

    return comparison


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate behavioral kernel extraction")
    parser.add_argument("--n-users", type=int, default=100, help="Number of synthetic users")
    parser.add_argument("--n-events", type=int, default=30, help="Events per user")
    parser.add_argument("--epsilon", type=float, default=float("inf"), help="DP epsilon (inf=no noise)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip simulation comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("  Behavioral Kernel Extraction — Validation & Demo")
    print("=" * 70)

    # Step 1: 生成合成数据
    print(f"\n[Step 1] Generating synthetic behavior chains ({args.n_users} users, {args.n_events} events/user)")
    df, ground_truth = generate_synthetic_chains(
        n_users=args.n_users,
        n_sessions_per_user=args.n_events,
        seed=args.seed,
    )
    print(f"  Generated {len(df)} behavior events from {args.n_users} users")
    print(f"  Steps distribution:")
    for step in ChainStep.funnel_steps():
        n = int((df["action"] == step.value).sum())
        print(f"    {step.value}: {n} events")

    # Step 2: 核参数提取（无 DP）
    print(f"\n[Step 2] Extracting kernel parameters (no DP)")
    extractor_no_dp = BehavioralKernelExtractor(
        min_samples=5,
        dp_epsilon=float("inf"),
        random_state=args.seed,
    )
    kernels_no_dp = extractor_no_dp.fit_extract(df)
    print(f"  Extracted {len(kernels_no_dp)} user kernels")

    # 评估精度
    accuracy = evaluate_extraction_accuracy(kernels_no_dp, ground_truth)
    print(f"\n  Extraction accuracy (vs ground truth):")
    print(f"    Overall theta MAE: {accuracy['overall_theta_mae']:.4f}")
    print(f"    Overall beta MAE:  {accuracy['overall_beta_mae']:.4f}")
    print(f"    Per-step theta MAE:")
    for step in ChainStep.funnel_steps():
        print(f"      {step.value}: {accuracy['theta_mae'][step.value]:.4f}")

    # 总体参数
    pop_summary = extractor_no_dp.get_population_summary()
    print(f"\n  Population parameters:")
    for step in ChainStep.funnel_steps():
        mu, tau = pop_summary["theta"][step.value]["mu"], pop_summary["theta"][step.value]["tau"]
        print(f"    {step.value}: mu={mu:.3f}, tau={tau:.1f}")

    # Step 3: DP 噪声影响
    if args.epsilon < float("inf"):
        print(f"\n[Step 3] DP noise impact (epsilon={args.epsilon})")
        extractor_dp = BehavioralKernelExtractor(
            min_samples=5,
            dp_epsilon=args.epsilon,
            random_state=args.seed,
        )
        kernels_dp = extractor_dp.fit_extract(df)
        dp_impact = evaluate_dp_impact(kernels_no_dp, kernels_dp)
        print(f"  Overall mean perturbation: {dp_impact['overall_mean_perturbation']:.4f}")
        for step in ChainStep.funnel_steps():
            print(f"    {step.value}: {dp_impact['theta_mean_perturbation'][step.value]:.4f}")
    else:
        print(f"\n[Step 3] DP noise: skipped (epsilon=inf, no noise)")

    # Step 4: 种群合成采样
    print(f"\n[Step 4] Population synthesis validation")
    pop_comparison = validate_population_synthesis(kernels_no_dp, seed=args.seed)
    print(f"  Original n={pop_comparison['n_original']}, Synthetic n={pop_comparison['n_synthetic']}")
    print(f"  Theta distribution comparison:")
    for step in ChainStep.funnel_steps():
        s = step.value
        print(f"    {s}: original_mean={pop_comparison['original_mean'][s]:.3f} "
              f"vs synthetic_mean={pop_comparison['synthetic_mean'][s]:.3f}")
        ks = pop_comparison["ks_test"].get(s, {})
        pval = ks.get("p_value", float("nan"))
        print(f"         KS test: p={pval:.3f} {'(similar)' if pval > 0.05 else '(different)'}")

    # Step 5: 仿真对比
    if not args.skip_simulation:
        print(f"\n[Step 5] Kernel-driven simulation comparison")
        sim_results = run_kernel_simulation(kernels_no_dp, seed=args.seed)
        kd = sim_results["kernel_driven"]
        bl = sim_results["baseline"]
        print(f"  Kernel-driven Agent:")
        print(f"    ROI={kd['avg_roi']:.3f}, ΔGTV={kd['cumulative_delta_gtv']:.1f}, "
              f"RedemptionRate={kd['avg_redemption_rate']:.2%}")
        print(f"  Baseline Agent (no kernel):")
        print(f"    ROI={bl['avg_roi']:.3f}, ΔGTV={bl['cumulative_delta_gtv']:.1f}, "
              f"RedemptionRate={bl['avg_redemption_rate']:.2%}")
        roi_diff = kd["avg_roi"] - bl["avg_roi"]
        print(f"  ROI difference: {roi_diff:+.3f}")

    # 示例核参数展示
    print(f"\n[Demo] Example user kernels:")
    for kernel in kernels_no_dp[:3]:
        print(f"  {kernel}")
        print(f"    beta: {kernel.beta}")
        print(f"    gamma: {kernel.gamma}")

    print("\n" + "=" * 70)
    print("  Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
