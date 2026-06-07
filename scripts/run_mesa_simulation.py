"""
Mesa ABM仿真运行脚本

功能：
1. 生成合成用户画像数据
2. 运行4种Agent策略对比（静态/动态/认知/随机）
3. 多平行世界仿真
4. 输出结果到 output/

用法：
    python scripts/run_mesa_simulation.py
"""

import sys
import os

# 添加项目根目录到sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import seaborn as sns

from src.simulation.mesa_agent_model import (
    SubsidyAgent,
    SubsidyModel,
    MultiWorldModel,
    StrategyType,
    SimulationResult,
)
from src.simulation.cognitive_agent_theory import prospect_value


# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
N_AGENTS = 500
N_ROUNDS = 30
SEED = 42


def ensure_output_dir():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single_strategy_demo():
    """
    单策略演示：运行认知策略模型，输出过程指标
    """
    print("=" * 60)
    print("1. 单策略演示：认知策略（前景理论+心理账户+有限理性）")
    print("=" * 60)

    model = SubsidyModel(
        n_agents=N_AGENTS,
        strategy="cognitive",
        seed=SEED,
    )
    result = model.run(n_rounds=N_ROUNDS)

    fm = result.final_metrics
    print(f"\n仿真完成：{fm['total_rounds']} 轮")
    print(f"平均ROI: {fm['avg_roi']:.4f}")
    print(f"累计ΔGTV: {fm['cumulative_delta_gtv']:.2f} 元")
    print(f"平均覆盖率: {fm['avg_coverage']:.4f}")
    print(f"平均核销率: {fm['avg_redemption_rate']:.4f}")
    print(f"最终平均疲劳度: {fm['final_avg_fatigue']:.4f}")
    print(f"最终平均参考点: {fm['final_avg_reference_point']:.4f}")
    print(f"总补贴支出: {fm['total_subsidy_spent']:.2f} 元")

    return result


def run_multi_world_comparison():
    """
    多平行世界仿真：4种策略对比
    """
    print("\n" + "=" * 60)
    print("2. 多平行世界仿真：4种策略对比")
    print("=" * 60)

    multi_world = MultiWorldModel(
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
        seed=SEED,
    )

    # 运行所有世界
    results = multi_world.run_all_strategies()

    # 对比分析
    comparison_df = multi_world.compare_worlds()
    print("\n多世界策略对比：")
    print(comparison_df.to_string(index=False))

    # 稳健性分析
    robustness = multi_world.robustness_analysis()
    print(f"\n最优策略: {robustness.get('best_strategy', 'N/A')}")
    print(f"ROI变异系数: {robustness.get('roi_cv', 0):.4f}")

    return multi_world, results


def plot_prospect_value_function():
    """
    绘制前景理论价值函数
    展示Kahneman & Tversky (1979)的S型价值函数
    """
    print("\n绘制前景理论价值函数...")

    x = np.linspace(-50, 50, 500)
    y = [prospect_value(xi, alpha=0.88, lambda_=2.25) for xi in x]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, y, "b-", linewidth=2, label="v(x)")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_xlabel("补贴金额（相对参考点）", fontsize=12)
    ax.set_ylabel("主观价值", fontsize=12)
    ax.set_title(
        "前景理论价值函数 (Kahneman & Tversky, 1979)\n"
        r"$\alpha$=0.88, $\lambda$=2.25",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 标注关键特征
    ax.annotate(
        "损失厌恶\n($\\lambda$=2.25)",
        xy=(-30, prospect_value(-30, alpha=0.88, lambda_=2.25)),
        xytext=(-40, 40),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )
    ax.annotate(
        "边际递减",
        xy=(30, prospect_value(30, alpha=0.88, lambda_=2.25)),
        xytext=(20, -30),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="blue"),
        color="blue",
    )

    path = os.path.join(OUTPUT_DIR, "prospect_value_function.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {path}")


def plot_multi_world_results(multi_world: MultiWorldModel):
    """
    绘制多世界仿真对比图
    """
    print("\n绘制多世界对比图...")

    step_df = multi_world.get_step_comparison()

    sns.set_style("whitegrid")
    strategy_colors = {
        "random": "#d62728",
        "static": "#1f77b4",
        "dynamic": "#ff7f0e",
        "cognitive": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. ROI对比
    ax = axes[0, 0]
    for strategy, color in strategy_colors.items():
        data = step_df[step_df["strategy"] == strategy]
        if len(data) > 0:
            ax.plot(data["round"], data["roi"], label=strategy, color=color, linewidth=2)
    ax.set_xlabel("仿真轮次")
    ax.set_ylabel("ROI")
    ax.set_title("ROI 随时间变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. ΔGTV对比
    ax = axes[0, 1]
    for strategy, color in strategy_colors.items():
        data = step_df[step_df["strategy"] == strategy]
        if len(data) > 0:
            ax.plot(data["round"], data["delta_gtv"], label=strategy, color=color, linewidth=2)
    ax.set_xlabel("仿真轮次")
    ax.set_ylabel("ΔGTV (元)")
    ax.set_title("ΔGTV 随时间变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 疲劳度对比
    ax = axes[1, 0]
    for strategy, color in strategy_colors.items():
        data = step_df[step_df["strategy"] == strategy]
        if len(data) > 0:
            ax.plot(data["round"], data["avg_fatigue"], label=strategy, color=color, linewidth=2)
    ax.set_xlabel("仿真轮次")
    ax.set_ylabel("平均疲劳度")
    ax.set_title("疲劳脱敏累积曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 参考点对比
    ax = axes[1, 1]
    for strategy, color in strategy_colors.items():
        data = step_df[step_df["strategy"] == strategy]
        if len(data) > 0:
            ax.plot(data["round"], data["avg_reference_point"], label=strategy, color=color, linewidth=2)
    ax.set_xlabel("仿真轮次")
    ax.set_ylabel("平均心理参考点")
    ax.set_title("心理账户参考点演化 (Thaler, 1985)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("多平行世界仿真：4种补贴策略对比", fontsize=15, y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "multi_world_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {path}")


def plot_strategy_bar_chart(multi_world: MultiWorldModel):
    """
    绘制策略最终指标柱状图
    """
    print("\n绘制策略对比柱状图...")

    comparison_df = multi_world.compare_worlds()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    strategies = comparison_df["world"].tolist()
    colors_map = {
        "random": "#d62728",
        "static": "#1f77b4",
        "dynamic": "#ff7f0e",
        "cognitive": "#2ca02c",
    }
    colors = [colors_map.get(s, "#888888") for s in strategies]

    # 1. 平均ROI
    ax = axes[0]
    bars = ax.bar(strategies, comparison_df["avg_roi"], color=colors)
    ax.set_title("平均ROI")
    ax.set_ylabel("ROI")
    for bar, val in zip(bars, comparison_df["avg_roi"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    # 2. 累计ΔGTV
    ax = axes[1]
    bars = ax.bar(strategies, comparison_df["cumulative_delta_gtv"], color=colors)
    ax.set_title("累计ΔGTV (元)")
    ax.set_ylabel("ΔGTV")
    for bar, val in zip(bars, comparison_df["cumulative_delta_gtv"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:.0f}", ha="center", fontsize=9)

    # 3. 最终疲劳度
    ax = axes[2]
    bars = ax.bar(strategies, comparison_df["final_avg_fatigue"], color=colors)
    ax.set_title("最终平均疲劳度")
    ax.set_ylabel("Fatigue")
    for bar, val in zip(bars, comparison_df["final_avg_fatigue"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=10)

    fig.suptitle("4种补贴策略最终指标对比", fontsize=14)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "strategy_bar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {path}")


def save_results(
    single_result: SimulationResult,
    multi_world: MultiWorldModel,
):
    """保存所有结果到output目录"""
    print("\n" + "=" * 60)
    print("3. 保存结果")
    print("=" * 60)

    # 1. 保存单策略详细步骤指标
    single_df = pd.DataFrame(single_result.round_metrics)
    single_path = os.path.join(OUTPUT_DIR, "cognitive_strategy_rounds.csv")
    single_df.to_csv(single_path, index=False)
    print(f"  已保存: {single_path}")

    # 2. 保存多世界对比结果
    comparison_df = multi_world.compare_worlds()
    compare_path = os.path.join(OUTPUT_DIR, "multi_world_comparison.csv")
    comparison_df.to_csv(compare_path, index=False)
    print(f"  已保存: {compare_path}")

    # 3. 保存多世界逐步对比
    step_df = multi_world.get_step_comparison()
    step_path = os.path.join(OUTPUT_DIR, "multi_world_step_metrics.csv")
    step_df.to_csv(step_path, index=False)
    print(f"  已保存: {step_path}")

    # 4. 保存JSON汇总
    summary = {
        "config": {
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "seed": SEED,
        },
        "strategies_compared": ["random", "static", "dynamic", "cognitive"],
        "final_comparison": comparison_df.to_dict(orient="records"),
    }
    summary_path = os.path.join(OUTPUT_DIR, "simulation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {summary_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("Mesa ABM 仿真系统")
    print(f"Agent数量: {N_AGENTS}, 仿真轮数: {N_ROUNDS}, 随机种子: {SEED}")

    ensure_output_dir()

    # 1. 单策略演示
    single_result = run_single_strategy_demo()

    # 2. 多世界对比
    multi_world, results = run_multi_world_comparison()

    # 3. 绘图
    plot_prospect_value_function()
    plot_multi_world_results(multi_world)
    plot_strategy_bar_chart(multi_world)

    # 4. 保存结果
    save_results(single_result, multi_world)

    print("\n" + "=" * 60)
    print("仿真完成！所有结果已保存到 output/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
