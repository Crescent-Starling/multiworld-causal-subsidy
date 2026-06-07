"""
综合运行脚本：一键运行所有模块的演示

Usage:
    python scripts/run_all_demos.py [--quick] [--no-git]

--quick: 使用小数据量快速测试
--no-git: 跳过 Git 提交
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.data_generator import generate_all_data, SyntheticDataConfig


def separator(title: str) -> str:
    """格式化分隔符"""
    line = "=" * 60
    return f"\n{line}\n{title}\n{line}\n"


def run_causal_inference_demo(causal_data: pd.DataFrame, verbose: bool = True) -> dict:
    """运行因果推断模块演示"""
    if verbose:
        print("PHASE 1: Causal Inference (CausalML + DoWhy + PSM)")

    results = {"causalml": {}, "dowhy": None, "psm": None}
    feature_cols = [c for c in causal_data.columns if c not in ("treatment", "outcome", "true_cate")]

    # 1. CausalML
    if verbose:
        print("\n--- 1.1 CausalML Meta-Learner Comparison ---")
    try:
        from src.modeling.causalml_wrapper import CausalMLWrapper, CausalMLConfig

        for learner_type in ["tlearner", "xlearner", "drlearner", "slearner"]:
            config = CausalMLConfig(learner_type=learner_type)
            wrapper = CausalMLWrapper(config)
            result = wrapper.fit_predict(causal_data, feature_cols, "treatment", "outcome")
            cate = result["cate_causalml"].values.flatten()
            true_cate = causal_data["true_cate"].values
            corr = np.corrcoef(cate, true_cate)[0, 1]
            results["causalml"][learner_type] = {"ate_mean": float(cate.mean()), "corr": float(corr)}

            if verbose:
                print(f"  {learner_type.upper():12s} | ATE={cate.mean():.4f} | Corr={corr:.4f}")
    except Exception as e:
        if verbose:
            print(f"  CausalML FAILED: {e}")

    # 2. DoWhy
    if verbose:
        print("\n--- 1.2 DoWhy Causal Graph ---")
    try:
        from src.modeling.dowhy_causal_graph import SubsidyCausalGraph, DoWhyConfig

        config = DoWhyConfig()
        cg = SubsidyCausalGraph(config)
        ate_result = cg.estimate_ate(causal_data, "treatment", "outcome")
        results["dowhy"] = ate_result
        if verbose:
            print(f"  ATE = {ate_result.get('ate', 'N/A'):.4f} (true=2.0)")
            print(f"  Method: {ate_result.get('method', 'N/A')}")
    except Exception as e:
        if verbose:
            print(f"  DoWhy FAILED: {e}")

    # 3. PSM
    if verbose:
        print("\n--- 1.3 PSM Matching ---")
    try:
        from src.modeling.psm_matcher import PSMMatcher, PSMConfig

        config = PSMConfig(method="nearest")
        matcher = PSMMatcher(config)
        matcher.compute_propensity_scores(causal_data, "treatment", feature_cols)
        matched_df = matcher.match(causal_data, "treatment")
        psm_ate = matcher.estimate_ate_after_matching(matched_df, "outcome", "treatment")
        results["psm"] = {"ate": float(psm_ate), "quality": matcher.match_quality}
        if verbose:
            print(f"  PSM ATE = {psm_ate:.4f} (true=2.0)")
            if matcher.match_quality:
                smd_max = matcher.match_quality.get('smd_matched_max', 'N/A')
                smd_mean = matcher.match_quality.get('smd_matched_mean', 'N/A')
                smd_max_str = f"{smd_max:.4f}" if isinstance(smd_max, float) else str(smd_max)
                smd_mean_str = f"{smd_mean:.4f}" if isinstance(smd_mean, float) else str(smd_mean)
                print(f"  Match quality: SMD_max={smd_max_str}, SMD_mean={smd_mean_str}")
    except Exception as e:
        if verbose:
            print(f"  PSM FAILED: {e}")

    return results


def run_theory_demo(user_profiles: pd.DataFrame, verbose: bool = True) -> dict:
    """运行认知 Agent 理论演示"""
    if verbose:
        print(separator("PHASE 1.5: Cognitive Agent Theory"))

    results = {}
    try:
        from src.simulation.cognitive_agent_theory import TheoreticalCognitiveAgent, MentalAccountType
        import numpy as np

        agents = {}
        for account_type in MentalAccountType:
            agent = TheoreticalCognitiveAgent(
                agent_id=f"demo_{account_type.value}",
                mental_account=account_type,
            )
            agents[account_type.value] = agent

        if verbose:
            print("\n  8-round simulation by mental account type:")
        for account_type, agent in agents.items():
            redemptions = 0
            for r in range(8):
                decided = agent.decide(subsidy_amount=10.0)
                agent.update_state(was_subsidized=True, redeemed=decided)
                if decided:
                    redemptions += 1
            results[account_type] = redemptions
            if verbose:
                print(f"  {account_type:20s} | redemptions={redemptions}/8 | fatigue={agent.fatigue:.4f}")

    except Exception as e:
        if verbose:
            print(f"  Cognitive Agent Theory FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def run_abm_demo(user_profiles: pd.DataFrame, verbose: bool = True) -> dict:
    """运行 ABM 仿真模块演示"""
    if verbose:
        print(separator("PHASE 2: ABM Simulation (Mesa + NetworkX)"))

    results = {"mesa": {}, "network": {}}

    # Mesa ABM
    if verbose:
        print("\n--- 2.1 Mesa ABM Simulation ---")
    try:
        from src.simulation.mesa_agent_model import SubsidyModel

        n_agents = min(500, len(user_profiles))
        model = SubsidyModel(n_agents=n_agents, strategy="cognitive", seed=42)

        model_results = []
        for r in range(8):
            model.step()
            r_result = model.collect_results()
            model_results.append(r_result)
            if verbose:
                print(f"  Round {r+1}: ROI={r_result['roi']:.2f}, Coverage={r_result['coverage']:.2%}")

        results_df = pd.DataFrame(model_results)
        results["mesa"] = {
            "roi_mean": float(results_df["roi"].mean()),
            "delta_gtv_sum": float(results_df["delta_gtv"].sum()),
        }
        if verbose:
            print(f"\n  Final ROI: {results['mesa']['roi_mean']:.2f}")
            print(f"  Final ΔGTV: {results['mesa']['delta_gtv_sum']:.1f}")
    except Exception as e:
        if verbose:
            print(f"  Mesa ABM FAILED: {e}")

    # NetworkX
    if verbose:
        print("\n--- 2.2 NetworkX Social Contagion ---")
    try:
        from src.simulation.network_contagion import SocialNetwork, SocialContagion

        sn = SocialNetwork()
        G = sn.build_barabasi_albert(500, 3, seed=42)
        if verbose:
            print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        sc = SocialContagion()
        seed_nodes = list(range(50))
        contagion_result = sc.propagate(G, seed_nodes, contagion_rate=0.15, n_steps=10)
        results["network"]["cascade_size"] = contagion_result["cascade_size"]
        results["network"]["cascade_ratio"] = contagion_result["cascade_ratio"]
        if verbose:
            print(f"  Contagion: seed={len(seed_nodes)}, "
                  f"final_infected={contagion_result['cascade_size']} "
                  f"({contagion_result['cascade_ratio']:.1%})")
    except Exception as e:
        if verbose:
            print(f"  NetworkX FAILED: {e}")

    return results


def run_multiworld_demo(verbose: bool = True) -> dict:
    """运行多平行世界仿真演示"""
    if verbose:
        print(separator("PHASE 2.5: Multi-World Simulation"))

    results = {}
    try:
        from src.simulation.mesa_agent_model import MultiWorldModel

        mw = MultiWorldModel(n_agents=200, n_rounds=5, seed=42)
        world_results = mw.run_all_strategies()

        # 对比
        comparison = mw.compare_worlds()
        results["comparison"] = comparison

        # 稳健性
        robustness = mw.robustness_analysis()
        results["robustness"] = robustness

        if verbose:
            print("\n  World comparison:")
            print(comparison.to_string(index=False))
            print(f"\n  Robustness: {robustness}")
    except Exception as e:
        if verbose:
            print(f"  Multi-World FAILED: {e}")

    return results


def run_llm_agent_demo(verbose: bool = True) -> dict:
    """运行 LLM Agent 演示（优先使用真实 API）"""
    if verbose:
        print(separator("PHASE 3: LLM Agent Simulation"))

    results = {}
    try:
        from src.simulation.llm_agent import LLMClient, LLMAgentSociety
        import os

        # 自动检测可用后端
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        has_deepseek = bool(os.environ.get("DEEPSEEK_API_KEY"))
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

        if has_deepseek:
            backend = "deepseek"
            api_key = os.environ["DEEPSEEK_API_KEY"]
            use_mock = False
        elif has_openai:
            backend = "openai"
            api_key = os.environ["OPENAI_API_KEY"]
            use_mock = False
        elif has_anthropic:
            backend = "anthropic"
            api_key = os.environ["ANTHROPIC_API_KEY"]
            use_mock = False
        else:
            backend = "mock"
            api_key = None
            use_mock = True

        mode_label = f"{backend.upper()} (真实LLM)" if not use_mock else "Mock (规则回退)"
        if verbose:
            print(f"  Backend: {mode_label}")

        # 创建 LLM 客户端
        llm_client = LLMClient(
            backend=backend,
            api_key=api_key,
        )

        # 创建 Agent 社会（小规模的演示）
        society = LLMAgentSociety(
            n_agents=4,
            use_mock=use_mock,
            backend=backend,
            api_key=api_key,
            seed=42,
        )

        # 运行 3 轮，递增门槛以观察决策分化
        scenarios = [(20, 50), (15, 100), (10, 150)]
        sim_results = []

        for subsidy, threshold in scenarios:
            r_result = society.run_round(subsidy_amount=subsidy, threshold=threshold)
            sim_results.append(r_result)
            if verbose:
                print(f"  Round {r_result['round']}: subsidy=¥{subsidy}, "
                      f"threshold=¥{threshold}, "
                      f"redemption={r_result['redemption_rate']:.0%} "
                      f"({r_result['n_redeemed']}/{society.n_agents})")

        results["sim_results"] = sim_results

        # 按心理账户汇总
        traj_df = society.get_trajectory_df()
        results["trajectory_df"] = traj_df  # 传给 save_results 保存
        if not traj_df.empty and verbose:
            print(f"\n  各心理账户核销率:")
            for acc, group in traj_df.groupby("mental_account"):
                rate = group["redeemed"].mean()
                print(f"    {acc:<25s}: {rate:.0%} ({int(group['redeemed'].sum())}/{len(group)})")

    except Exception as e:
        if verbose:
            print(f"  LLM Agent FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def run_evaluation_demo(causal_data: pd.DataFrame, verbose: bool = True) -> dict:
    """运行评估模块演示"""
    if verbose:
        print(separator("EVALUATION: Metrics & Robustness"))

    results = {}
    try:
        from src.evaluation.metrics import bootstrap_ci, compute_roi, e_value, multi_world_robustness

        outcome = causal_data["outcome"].values
        ci = bootstrap_ci(outcome, statistic=np.mean, n_bootstrap=1000, ci=0.95)
        results["bootstrap_ci"] = ci
        if verbose:
            print(f"  Bootstrap CI for mean outcome: [{ci[0]:.4f}, {ci[1]:.4f}]")

        roi = compute_roi(subsidy_cost=100, incremental_gtv=250)
        results["roi"] = roi
        if verbose:
            print(f"  ROI: {roi:.2f}")

        ev = e_value(rr=1.5)
        results["e_value"] = ev
        if verbose:
            print(f"  E-value for RR=1.5: {ev:.4f}")

    except Exception as e:
        if verbose:
            print(f"  Evaluation FAILED: {e}")

    return results


def save_results(all_results: dict, output_dir: str = "output") -> None:
    """保存所有结果到 output/ 目录"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存因果推断结果
    if "causal" in all_results:
        causal = all_results["causal"]
        rows = []
        for method, vals in causal.get("causalml", {}).items():
            rows.append({"method": f"CausalML_{method}", "ate_mean": vals.get("ate_mean"), "corr": vals.get("corr")})
        if causal.get("dowhy"):
            rows.append({"method": "DoWhy", "ate": causal["dowhy"].get("ate")})
        if causal.get("psm"):
            rows.append({"method": "PSM", "ate": causal["psm"].get("ate")})

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(output_dir, "causal_results.csv"), index=False)

    # 保存 ABM 结果
    if "abm" in all_results:
        abm = all_results["abm"]

        # Mesa 结果
        if "mesa" in abm:
            mesa_rows = [{"metric": k, "value": v} for k, v in abm["mesa"].items()]
            if mesa_rows:
                pd.DataFrame(mesa_rows).to_csv(
                    os.path.join(output_dir, "mesa_results.csv"), index=False
                )

        # Network 结果
        if "network" in abm:
            network_rows = [{"metric": k, "value": v} for k, v in abm["network"].items()]
            if network_rows:
                pd.DataFrame(network_rows).to_csv(
                    os.path.join(output_dir, "network_results.csv"), index=False
                )

    # 保存 Multi-World 对比结果
    if "multiworld" in all_results and "comparison" in all_results.get("multiworld", {}):
        comparison = all_results["multiworld"]["comparison"]
        if isinstance(comparison, pd.DataFrame):
            comparison.to_csv(os.path.join(output_dir, "multi_world_comparison.csv"), index=False)

    # 保存理论 Agent 结果
    if "theory" in all_results:
        theory_rows = [{"account_type": k, "redemptions": v} for k, v in all_results["theory"].items()]
        if theory_rows:
            pd.DataFrame(theory_rows).to_csv(
                os.path.join(output_dir, "theory_agent_results.csv"), index=False
            )

    # 保存评估结果
    if "eval" in all_results:
        eval_rows = [{"metric": k, "value": str(v)} for k, v in all_results["eval"].items()]
        if eval_rows:
            pd.DataFrame(eval_rows).to_csv(
                os.path.join(output_dir, "evaluation_results.csv"), index=False
            )

    # 保存 LLM Agent 轨迹（含 reasoning chain）
    if "llm" in all_results:
        llm = all_results["llm"]
        traj_df = llm.get("trajectory_df")
        if traj_df is not None and not traj_df.empty:
            traj_path = os.path.join(output_dir, "llm_agent_trajectories.csv")
            traj_df.to_csv(traj_path, index=False, encoding="utf-8-sig")
            print(f"  LLM trajectories saved to: {traj_path}")

            # 同时保存完整 JSON（含完整 reasoning，CSV 会截断长文本）
            import json
            json_path = os.path.join(output_dir, "llm_agent_trajectories.json")
            traj_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
            print(f"  LLM trajectories (JSON, full reasoning) saved to: {json_path}")

    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run all demos")
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument("--no-git", action="store_true", help="Skip Git operations")
    args = parser.parse_args()

    print(separator("AI Subsidy Simulation System - Full Demo"))

    # 生成合成数据
    if args.quick:
        config = SyntheticDataConfig(n_users=2000, n_orders=8000)
        print("\n[Quick mode] Using small data...")
    else:
        config = SyntheticDataConfig(n_users=10000, n_orders=50000)

    print("\nGenerating synthetic data...")
    data = generate_all_data(config)

    user_profiles = data["user_profiles"]
    orders = data["orders"]
    causal_data = data["causal_data"]

    print(f"  User profiles: {user_profiles.shape}")
    print(f"  Orders: {orders.shape}")
    print(f"  Causal data: {causal_data.shape}")

    # 保存合成数据
    os.makedirs("data/synthetic", exist_ok=True)
    for name, df in data.items():
        path = os.path.join("data/synthetic", f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    all_results = {}
    start_time = time.time()

    # 运行各模块
    all_results["causal"] = run_causal_inference_demo(causal_data)
    all_results["theory"] = run_theory_demo(user_profiles)
    all_results["abm"] = run_abm_demo(user_profiles)
    all_results["multiworld"] = run_multiworld_demo()
    all_results["llm"] = run_llm_agent_demo()
    all_results["eval"] = run_evaluation_demo(causal_data)

    elapsed = time.time() - start_time
    print(separator(f"All demos completed in {elapsed:.1f}s"))

    # 保存结果
    save_results(all_results)

    return all_results


if __name__ == "__main__":
    main()
