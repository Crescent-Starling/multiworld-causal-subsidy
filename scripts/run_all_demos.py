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
                print(f"  Match quality: SMD={matcher.match_quality.get('smd_matched', 'N/A'):.4f}")
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

        agents = {}
        for account_type in MentalAccountType:
            agent = TheoreticalCognitiveAgent(
                agent_id=f"demo_{account_type.value}",
                mental_account=account_type,
                price_sensitivity=np.random.uniform(0.2, 0.8),
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
        infected_history = sc.propagate(G, seed_nodes, contagion_rate=0.15, n_steps=10)
        results["network"]["cascade_size"] = infected_history[-1]
        if verbose:
            print(f"  Contagion: seed={len(seed_nodes)}, final_infected={infected_history[-1]}")
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
    """运行 LLM Agent 演示"""
    if verbose:
        print(separator("PHASE 3: LLM Agent Simulation"))

    results = {}
    try:
        from src.simulation.llm_agent import LLMAgentSociety

        society = LLMAgentSociety(n_agents=5, use_mock=True)
        sim_results = society.run_simulation(n_rounds=3)
        results["sim_results"] = sim_results

        if verbose:
            print(f"  LLM Agent simulation completed: {len(sim_results)} rounds")
            for r_result in sim_results:
                print(f"  Round {r_result['round']}: redemption_rate={r_result['redemption_rate']:.2%}")
    except Exception as e:
        if verbose:
            print(f"  LLM Agent FAILED: {e}")

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
    if "abm" in all_results and "mesa" in all_results["abm"]:
        pass  # Mesa 结果已经在内存中

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
