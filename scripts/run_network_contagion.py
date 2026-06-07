"""
运行社会网络传染仿真

生成合成数据，构建 3 种社会网络（属性相似度 / BA无标度 / WS小世界），
运行 SIR 社会传染仿真和 Agent 模型仿真，输出结果和可视化。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.network_contagion import (
    NetworkContagionAgent,
    SocialContagion,
    SocialNetwork,
)


def generate_synthetic_data(n_users: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成合成用户数据

    生成用于构建社会网络的合成用户属性数据。

    Parameters
    ----------
    n_users : int
        用户数
    seed : int
        随机种子

    Returns
    -------
    pd.DataFrame
        用户属性数据
    """
    rng = np.random.RandomState(seed)

    df = pd.DataFrame({
        "user_id": range(n_users),
        # 消费频率（0-1标准化）
        "consumption_freq": rng.beta(2, 5, n_users),
        # 价格敏感度（0-1）
        "price_sensitivity": rng.beta(3, 3, n_users),
        # 平均消费金额
        "avg_spending": rng.lognormal(3, 0.8, n_users),
        # 偏好品类（独热编码简化为连续值）
        "pref_food": rng.beta(2, 2, n_users),
        "pref_drink": rng.beta(1, 3, n_users),
        "pref_entertainment": rng.beta(1, 4, n_users),
        # 活跃时段（0-24小时）
        "active_hour": rng.uniform(8, 22, n_users),
        # 地理位置（二维坐标）
        "lat": rng.normal(39.9, 0.05, n_users),
        "lng": rng.normal(116.4, 0.05, n_users),
    })

    # 生成 POI 共现数据
    poi_records = []
    n_pois = 50
    for user_id in range(n_users):
        # 每个用户随机访问 3-15 个 POI
        n_visits = rng.randint(3, 16)
        visited_pois = rng.choice(n_pois, size=n_visits, replace=True)
        for poi_id in visited_pois:
            poi_records.append({
                "user_id": user_id,
                "poi_id": poi_id,
            })

    poi_df = pd.DataFrame(poi_records)

    return df, poi_df


def run_experiment():
    """运行完整的社会网络传染实验"""

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "network_contagion")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("社会网络传染仿真实验")
    print("Social Network Contagion Simulation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. 生成合成数据
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. 生成合成数据")
    print("=" * 70)
    n_users = 500
    user_df, poi_df = generate_synthetic_data(n_users=n_users)
    print(f"  用户数: {len(user_df)}, POI记录数: {len(poi_df)}")

    # 保存合成数据
    user_df.to_csv(os.path.join(output_dir, "synthetic_users.csv"), index=False)
    poi_df.to_csv(os.path.join(output_dir, "synthetic_poi.csv"), index=False)

    # ------------------------------------------------------------------
    # 2. 构建 3 种社会网络
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. 构建社会网络")
    print("=" * 70)

    sn = SocialNetwork()

    # --- 2a. 属性相似度网络 ---
    print("\n[2a] 属性相似度网络")
    feature_cols = [
        "consumption_freq", "price_sensitivity", "avg_spending",
        "pref_food", "pref_drink", "pref_entertainment", "active_hour",
    ]
    t0 = time.time()
    G_attr = sn.build_from_attributes(user_df, feature_cols, threshold=0.85)
    print(f"  耗时: {time.time() - t0:.2f}s")

    # --- 2b. BA 无标度网络 ---
    print("\n[2b] BA 无标度网络")
    sn_ba = SocialNetwork()
    t0 = time.time()
    G_ba = sn_ba.build_barabasi_albert(n=n_users, m=3, seed=42)
    print(f"  耗时: {time.time() - t0:.2f}s")

    # --- 2c. WS 小世界网络 ---
    print("\n[2c] WS 小世界网络")
    sn_ws = SocialNetwork()
    t0 = time.time()
    G_ws = sn_ws.build_watts_strogatz(n=n_users, k=6, p=0.3, seed=42)
    print(f"  耗时: {time.time() - t0:.2f}s")

    # --- 2d. POI 共现网络 ---
    print("\n[2d] POI 共现网络")
    sn_poi = SocialNetwork()
    t0 = time.time()
    G_poi = sn_poi.build_from_coorcurrence(poi_df, "user_id", "poi_id")
    print(f"  耗时: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. 网络拓扑对比
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. 网络拓扑统计对比")
    print("=" * 70)

    networks = {
        "属性相似度": G_attr,
        "BA无标度": G_ba,
        "WS小世界": G_ws,
        "POI共现": G_poi,
    }

    topology_stats = []
    for name, G in networks.items():
        degrees = [d for _, d in G.degree()]
        stats = {
            "网络类型": name,
            "节点数": G.number_of_nodes(),
            "边数": G.number_of_edges(),
            "平均度": np.mean(degrees),
            "最大度": max(degrees) if degrees else 0,
            "聚类系数": nx.average_clustering(G),
            "连通分量": nx.number_connected_components(G),
        }
        if nx.is_connected(G):
            stats["平均路径长度"] = nx.average_shortest_path_length(G)
        else:
            stats["平均路径长度"] = float("inf")
        topology_stats.append(stats)
        print(f"\n  [{name}]")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    topology_df = pd.DataFrame(topology_stats)
    topology_df.to_csv(os.path.join(output_dir, "network_topology.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. SIR 社会传染仿真
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. SIR 社会传染仿真")
    print("=" * 70)

    sc = SocialContagion()

    # 选择种子节点（随机选取 5% 的用户）
    rng = np.random.RandomState(42)
    n_seeds = max(int(n_users * 0.05), 1)
    seed_nodes = list(rng.choice(n_users, size=n_seeds, replace=False))

    # 价格敏感度字典
    price_sensitivity = {
        i: float(user_df.loc[i, "price_sensitivity"])
        for i in range(n_users)
    }

    sir_results = {}
    for name, G in networks.items():
        print(f"\n[{name}] SIR传播")
        result = sc.propagate(
            G, seed_nodes,
            contagion_rate=0.1,
            n_steps=20,
            price_sensitivity=price_sensitivity,
            seed=42,
        )
        sir_results[name] = result

    # 保存 SIR 时间序列
    for name, result in sir_results.items():
        ts_df = pd.DataFrame(result["time_series"])
        ts_df.to_csv(os.path.join(output_dir, f"sir_ts_{name}.csv"), index=False)

    # 绘制对比 SIR 曲线
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"属性相似度": "blue", "BA无标度": "red", "WS小世界": "green", "POI共现": "orange"}
    for name, result in sir_results.items():
        ts = result["time_series"]
        steps = [t["step"] for t in ts]
        i_vals = [t["I"] for t in ts]
        ax.plot(steps, i_vals, "-", color=colors.get(name, "black"),
                linewidth=2, label=f"{name} (级联={result['cascade_size']})")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Infected Users", fontsize=12)
    ax.set_title("SIR Social Contagion Comparison Across Networks", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sir_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[SIR] 对比曲线已保存")

    # ------------------------------------------------------------------
    # 5. 级联规模统计
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. 级联规模统计（50次重复）")
    print("=" * 70)

    cascade_stats = []
    for name, G in networks.items():
        print(f"\n[{name}] 级联规模统计")
        stats = sc.compute_cascade_size(
            G, seed_nodes,
            contagion_rate=0.1,
            n_repeats=50,
            seed=42,
        )
        stats["网络类型"] = name
        cascade_stats.append(stats)

    cascade_df = pd.DataFrame(cascade_stats)
    cascade_df.to_csv(os.path.join(output_dir, "cascade_stats.csv"), index=False)

    # ------------------------------------------------------------------
    # 6. 社会效应估计
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. 社会效应估计（同伴效应）")
    print("=" * 70)

    social_effects = []
    for name, G in networks.items():
        print(f"\n[{name}] 社会效应估计")
        estimate = sc.estimate_social_effect(
            G, seed_nodes,
            outcome_attr="redeem",
            n_simulations=50,
            seed=42,
        )
        estimate["网络类型"] = name
        social_effects.append(estimate)

    social_df = pd.DataFrame(social_effects)
    social_df.to_csv(os.path.join(output_dir, "social_effects.csv"), index=False)

    # ------------------------------------------------------------------
    # 7. Agent 级仿真（社会压力对核销的影响）
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. NetworkContagionAgent 仿真")
    print("=" * 70)

    # 对比不同社会压力因子下的核销率
    pressure_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    agent_results = []

    for factor in pressure_factors:
        print(f"\n[social_pressure_factor = {factor:.1f}]")

        # 使用 BA 网络
        base_prob = {
            i: float(user_df.loc[i, "price_sensitivity"] * 0.4 + 0.1)
            for i in range(n_users)
        }

        agent = NetworkContagionAgent(
            G=G_ba,
            base_probability=base_prob,
            social_pressure_factor=factor,
            seed=42,
        )

        # 初始种子：5% 的用户已核销
        initial = list(rng.choice(n_users, size=n_seeds, replace=False))
        result = agent.simulate(n_steps=10, initial_redeemed=initial)

        agent_results.append({
            "social_pressure_factor": factor,
            "total_redeemed": result["total_redeemed"],
            "redeem_rate": result["redeem_rate"],
            "social_lift": result["social_lift"],
        })

    agent_df = pd.DataFrame(agent_results)
    agent_df.to_csv(os.path.join(output_dir, "agent_simulation.csv"), index=False)

    print("\n社会压力因子 vs 核销率:")
    print(agent_df[["social_pressure_factor", "redeem_rate", "social_lift"]].to_string(index=False))

    # 绘制社会压力因子与核销率关系图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.plot(agent_df["social_pressure_factor"], agent_df["redeem_rate"] * 100,
             "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Social Pressure Factor", fontsize=12)
    ax1.set_ylabel("Redeem Rate (%)", fontsize=12)
    ax1.set_title("Social Pressure vs Redeem Rate", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(
        [str(f) for f in agent_df["social_pressure_factor"]],
        agent_df["social_lift"] * 100,
        color="steelblue", alpha=0.7, edgecolor="white",
    )
    ax2.set_xlabel("Social Pressure Factor", fontsize=12)
    ax2.set_ylabel("Social Lift (%)", fontsize=12)
    ax2.set_title("Social Lift by Pressure Factor", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "social_pressure_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[Agent] 社会压力分析图已保存")

    # ------------------------------------------------------------------
    # 8. 网络可视化
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. 网络可视化")
    print("=" * 70)

    for name, sn_obj in [
        ("属性相似度", sn),
        ("BA无标度", sn_ba),
        ("WS小世界", sn_ws),
    ]:
        save_path = os.path.join(output_dir, f"network_{name}.png")
        sn_obj.visualize(save_path=save_path, max_nodes=300)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("实验完成！输出文件汇总:")
    print("=" * 70)
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f:<40s} ({size_kb:.1f} KB)")
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    run_experiment()
