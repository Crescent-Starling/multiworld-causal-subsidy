"""
行为链完整运行脚本

端到端演示 B 线（行为链升级）的完整流程：
  1. 生成行为链合成数据（参数来自美团数据校准）
  2. 多结果 CATE 估计（各步骤独立 X-Learner）
  3. 中介分析（补贴→核销的直接/间接效应分解）
  4. 行为链 Agent 仿真（对比 cognitive / cate_chain 策略）
  5. 输出结果与可视化

用法：
    python scripts/run_behavior_chain.py
    python scripts/run_behavior_chain.py --n-agents 200 --n-rounds 10
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 项目根目录（从 scripts/ 向上一级）
ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.features.data_generator import (
    SyntheticDataConfig,
    generate_all_data,
    calibrate_from_meituan,
)
from src.modeling.multi_outcome_cate import MultiOutcomeCATE
from src.modeling.mediation_analyzer import BehaviorChainMediator
from src.simulation.mesa_agent_model import SubsidyModel, StrategyType


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _safe_fmt(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return format(v, fmt)


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def plot_funnel(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    steps: list = None,
    save_path: str = None,
) -> None:
    """绘制行为链漏斗对比图（处理组 vs 对照组）"""
    if steps is None:
        steps = ["browsed", "clicked", "carted", "paid", "redeemed"]

    ctrl = df[df[treatment_col] == 0]
    treat = df[df[treatment_col] == 1]

    ctrl_rates = [ctrl[s].mean() if s in ctrl.columns else 1.0 for s in steps]
    treat_rates = [treat[s].mean() if s in treat.columns else 1.0 for s in steps]

    x = np.arange(len(steps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_ctrl  = ax.bar(x - width/2, ctrl_rates,  width, label="对照组（无补贴）", color="#5B8DB8", alpha=0.85)
    bars_treat = ax.bar(x + width/2, treat_rates, width, label="处理组（有补贴）", color="#E07B54", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=11)
    ax.set_ylabel("转化率", fontsize=11)
    ax.set_title("行为链漏斗转化率：处理组 vs 对照组", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)

    # 标注数值
    for bar in bars_ctrl:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9)
    for bar in bars_treat:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [图] 漏斗图已保存: {save_path}")
    plt.close()


def plot_cate_heatmap(
    cate_summary: dict,
    save_path: str = None,
) -> None:
    """绘制各步骤 ATE 柱状图"""
    steps = list(cate_summary.keys())
    ates  = [cate_summary[s] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in ates]
    bars = ax.bar(steps, ates, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, ates):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.001 if val >= 0 else val - 0.002,
                f"{val:.4f}", ha="center", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ATE（平均处理效应）", fontsize=11)
    ax.set_title("行为链各步骤 ATE（补贴的增量效应）", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [图] CATE 柱状图已保存: {save_path}")
    plt.close()


def plot_simulation_comparison(
    results: dict,
    save_path: str = None,
) -> None:
    """绘制策略 ROI 对比图"""
    strategies = list(results.keys())
    rois = [results[s].final_metrics.get("avg_roi", 0) for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(strategies)))
    bars = ax.bar(strategies, rois, color=colors, alpha=0.88, edgecolor="white")

    for bar, val in zip(bars, rois):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(val, 0) + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("平均 ROI", fontsize=11)
    ax.set_title("行为链场景：各策略 ROI 对比", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=15, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [图] 策略对比图已保存: {save_path}")
    plt.close()


def plot_chain_funnel_sim(
    model: SubsidyModel,
    save_path: str = None,
) -> None:
    """绘制仿真漏斗轨迹（各轮次平均）"""
    df = model.get_summary()
    chain_cols = [c for c in df.columns if c.startswith("chain_")]
    if not chain_cols:
        return

    steps = [c.replace("chain_", "").replace("_rate", "") for c in chain_cols]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (col, step) in enumerate(zip(chain_cols, steps)):
        ax.plot(df["round"], df[col], marker="o", label=step, linewidth=1.5, markersize=4)

    ax.set_xlabel("仿真轮次", fontsize=11)
    ax.set_ylabel("漏斗各步转化率", fontsize=11)
    ax.set_title("行为链仿真：各轮次漏斗转化率轨迹", fontsize=13, fontweight="bold")
    ax.legend(loc="right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [图] 仿真漏斗轨迹图已保存: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_behavior_chain_pipeline(
    n_agents: int = 200,
    n_rounds: int = 10,
    n_users: int = 800,
    seed: int = 42,
    output_dir: str = "output/behavior_chain",
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Step 0: 从美团数据校准参数 ----
    print_section("Step 0: 从美团数据校准行为链转换率")
    rates = calibrate_from_meituan(
        order_path=str(ROOT / "data/神券订单数据样例.xlsx"),
        behavior_path=str(ROOT / "data/用户行为序列.xlsx"),
    )
    print("  校准转换率：")
    for k, v in rates.items():
        print(f"    {k:12s}: {v:.3f}")

    # ---- Step 1: 生成行为链合成数据 ----
    print_section("Step 1: 生成行为链合成数据")
    config = SyntheticDataConfig(
        n_users=n_users,
        enable_behavior_chain=True,
        random_state=seed,
    )
    config.base_conversion_rates = rates
    all_data = generate_all_data(config)
    bc_df = all_data["behavior_chain"]
    print(f"  行为链数据形状: {bc_df.shape}")
    print(f"  各步骤整体转化率：")
    for step in ["browsed", "clicked", "carted", "paid", "redeemed"]:
        print(f"    {step:12s}: 全量={bc_df[step].mean():.3f}  "
              f"处理组={bc_df[bc_df['treatment']==1][step].mean():.3f}  "
              f"对照组={bc_df[bc_df['treatment']==0][step].mean():.3f}")

    # 漏斗图
    plot_funnel(
        bc_df,
        save_path=os.path.join(output_dir, "funnel_control_vs_treated.png"),
    )

    # ---- Step 2: 多结果 CATE 估计 ----
    print_section("Step 2: 多结果 CATE 估计（行为链各步）")
    feature_cols = ["price_sensitivity", "income_level"]
    cate_estimator = MultiOutcomeCATE(
        outcome_names=["clicked", "carted", "paid", "redeemed"],
        learner_type="xlearner",
        sequential=False,
    )
    bc_with_cate = cate_estimator.fit_predict(bc_df, feature_cols)
    print("\n  ATE 摘要（各步骤平均处理效应）：")
    summary = cate_estimator.summary()
    print(summary.to_string(index=False))

    # CATE 质量评估
    eval_res = cate_estimator.evaluate_all_steps(bc_with_cate)
    print("\n  CATE 质量（与真实 CATE 相关性）：")
    for step, info in eval_res.items():
        print(f"    {step:10s}: corr={_safe_fmt(info.get('corr'),'4f')}  "
              f"mae={_safe_fmt(info.get('mae'),'4f')}  "
              f"ate={_safe_fmt(info.get('ate'), '4f')}")

    # 保存 ATE 柱状图
    ate_dict = {s: cate_estimator.ate_estimates.get(s, 0.0)
                for s in ["clicked", "carted", "paid", "redeemed"]}
    plot_cate_heatmap(ate_dict, save_path=os.path.join(output_dir, "cate_ate_by_step.png"))

    # ---- Step 3: 中介分析 ----
    print_section("Step 3: 中介分析（补贴→核销的直接/间接效应）")
    mediator = BehaviorChainMediator(
        treatment_col="treatment",
        outcome_col="redeemed",
        mediator_cols=["clicked", "carted", "paid"],
        feature_cols=feature_cols,
    )
    med_results = mediator.fit_analyze(bc_df, n_bootstrap=100, seed=seed)
    mediator.plot_mediation_summary()
    med_df = med_results["summary"]
    med_df.to_csv(os.path.join(output_dir, "mediation_results.csv"), index=False)
    print(f"  中介分析结果已保存: {output_dir}/mediation_results.csv")

    # ---- Step 4: 行为链 Agent 仿真 ----
    print_section("Step 4: 行为链 Agent 仿真（多策略对比）")

    # 构建 CATE 评分映射（agent_id → {step: cate}）
    # bc_with_cate 每行是一条曝光记录；用户可能有多条，取均值
    chain_cate_scores: dict[int, dict] = {}
    n_sim_agents = n_agents
    cate_steps = ["clicked", "carted", "paid", "redeemed"]
    for agent_id in range(n_sim_agents):
        idx = agent_id % len(bc_with_cate)
        chain_cate_scores[agent_id] = {
            step: float(bc_with_cate[f"cate_{step}"].iloc[idx])
            for step in cate_steps
        }

    # 单步骤 CATE（用于 CATE_DRIVEN，取 redeemed）
    cate_scores_single = {
        aid: v["redeemed"] for aid, v in chain_cate_scores.items()
    }

    strategies_to_run = {
        "cognitive":  {"strategy": "cognitive",  "behavior_chain_enabled": False},
        "cate_driven":{"strategy": "cate_driven","behavior_chain_enabled": False},
        "chain_cognitive": {"strategy": "cognitive",  "behavior_chain_enabled": True},
        "cate_chain": {"strategy": "cate_chain", "behavior_chain_enabled": True},
    }

    sim_results = {}
    for name, params in strategies_to_run.items():
        print(f"\n  运行策略: {name}")
        model = SubsidyModel(
            n_agents=n_sim_agents,
            strategy=params["strategy"],
            budget_ratio=0.4,
            subsidy_amount=10.0,
            seed=seed,
            behavior_chain_enabled=params["behavior_chain_enabled"],
            base_chain_rates=rates,
            cate_scores=cate_scores_single,
            chain_cate_scores=chain_cate_scores,
        )
        for _ in range(n_rounds):
            model.step()

        df_m = model.get_summary()
        avg_roi  = float(df_m["roi"].mean())
        avg_rdm  = float(df_m["redemption_rate"].mean())
        cum_gtv  = float(df_m["delta_gtv"].sum())
        print(f"    avg_roi={avg_roi:.3f}  avg_redeem={avg_rdm:.3f}  cumΔGTV={cum_gtv:.1f}")

        # 构建 SimulationResult 兼容格式
        from src.simulation.mesa_agent_model import SimulationResult, StrategyType as ST
        result = SimulationResult(
            strategy=ST(params["strategy"]),
            round_metrics=model.round_results,
            final_metrics={
                "avg_roi": avg_roi,
                "avg_redemption_rate": avg_rdm,
                "cumulative_delta_gtv": cum_gtv,
            },
            behavior_chain_metrics={
                col.replace("chain_","").replace("_rate",""): float(df_m[col].mean())
                for col in df_m.columns if col.startswith("chain_")
            } if params["behavior_chain_enabled"] else {},
        )
        sim_results[name] = result

        # 行为链漏斗轨迹图
        if params["behavior_chain_enabled"]:
            plot_chain_funnel_sim(
                model,
                save_path=os.path.join(output_dir, f"funnel_sim_{name}.png"),
            )

    # 策略对比图
    plot_simulation_comparison(
        sim_results,
        save_path=os.path.join(output_dir, "strategy_roi_comparison.png"),
    )

    # ---- Step 5: 输出摘要报告 ----
    print_section("Step 5: 摘要报告")
    print(f"\n  {'策略':<18} {'avg_ROI':>10} {'avg_核销率':>10} {'cumΔGTV':>12}")
    print(f"  {'-'*52}")
    for name, result in sim_results.items():
        m = result.final_metrics
        print(f"  {name:<18} {m['avg_roi']:>10.3f} {m['avg_redemption_rate']:>10.3f} "
              f"{m['cumulative_delta_gtv']:>12.1f}")

    print()
    print(f"  行为链 CATE 摘要：")
    for step, ate in ate_dict.items():
        print(f"    {step:12s} ATE = {ate:.4f}")

    # 保存结果 JSON
    output_json = {
        "calibrated_rates": rates,
        "cate_ate_by_step": ate_dict,
        "mediation": {
            "total_effect": round(med_results["total_effect"], 4),
            "direct_effect": round(med_results["direct_effect"], 4),
            "indirect_effects": {k: round(v, 4) for k, v in med_results["indirect_effects"].items()},
            "proportion_mediated": round(med_results["proportion_mediated"], 4),
        },
        "simulation": {
            name: {
                "avg_roi": round(r.final_metrics["avg_roi"], 4),
                "avg_redemption_rate": round(r.final_metrics["avg_redemption_rate"], 4),
                "cumulative_delta_gtv": round(r.final_metrics["cumulative_delta_gtv"], 2),
                "behavior_chain_metrics": r.behavior_chain_metrics,
            }
            for name, r in sim_results.items()
        },
    }
    json_path = os.path.join(output_dir, "behavior_chain_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print(f"\n  完整结果已保存: {json_path}")
    print(f"  可视化图表目录: {output_dir}/")
    print()

    return output_json


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B线行为链仿真运行脚本")
    parser.add_argument("--n-agents", type=int, default=200, help="仿真 Agent 数量")
    parser.add_argument("--n-rounds", type=int, default=10, help="仿真轮次")
    parser.add_argument("--n-users", type=int, default=800, help="合成数据用户数")
    parser.add_argument("--seed",     type=int, default=42,  help="随机种子")
    parser.add_argument("--output-dir", type=str,
                        default="output/behavior_chain", help="输出目录")
    args = parser.parse_args()

    run_behavior_chain_pipeline(
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        n_users=args.n_users,
        seed=args.seed,
        output_dir=args.output_dir,
    )
