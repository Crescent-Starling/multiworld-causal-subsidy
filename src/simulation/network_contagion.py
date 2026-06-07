"""
社会网络传染模型

基于 NetworkX 构建用户社会网络，实现社会传染（Social Contagion）模型。
用于模拟补贴信息在社交网络中的传播过程，以及网络同伴效应对用户核销行为的影响。

参考文献:
- Christakis, N. A., & Fowler, J. H. (2007). The spread of obesity in a
  large social network over 32 years. New England Journal of Medicine, 357(4), 370-379.
- Centola, D., & Macy, M. (2007). Complex contagions and the weakness of
  long ties. American Journal of Sociology, 113(3), 702-734.
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of
  'small-world' networks. Nature, 393(6684), 440-442.
- Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random
  networks. Science, 286(5439), 509-512.
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免依赖 GUI
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
#  SocialNetwork — 社会网络构建
# ===========================================================================

class SocialNetwork:
    """社会网络构建器

    支持多种网络构建方式:
    1. 基于用户属性相似度的虚拟社会网络
    2. Barabási-Albert 无标度网络
    3. Watts-Strogatz 小世界网络
    4. 基于 POI 共现的行为网络
    """

    def __init__(self):
        self.graph: Optional[nx.Graph] = None

    # -----------------------------------------------------------------------
    #  基于属性相似度构建网络
    #  参考: Centola & Macy (2007) — 同质性驱动的社交网络
    # -----------------------------------------------------------------------
    def build_from_attributes(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        threshold: float = 0.5,
    ) -> nx.Graph:
        """基于用户属性相似度构建虚拟社会网络

        对每对用户计算余弦相似度，相似度超过阈值的用户之间建立连边，
        边权重等于余弦相似度。

        Parameters
        ----------
        df : pd.DataFrame
            用户属性数据，每行一个用户
        feature_cols : list[str]
            用于计算相似度的特征列名
        threshold : float
            连边阈值，仅相似度大于此值的用户对建立连边

        Returns
        -------
        nx.Graph
            构建好的社会网络图
        """
        n_users = len(df)
        features = df[feature_cols].values.astype(float)

        # 标准化特征（Min-Max 到 [0, 1]），使余弦相似度更有意义
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

        G = nx.Graph()
        G.add_nodes_from(range(n_users))

        # 为节点添加属性
        for i, row in df.iterrows():
            G.nodes[i].update(row.to_dict())

        edge_count = 0
        # 计算所有用户对的余弦相似度（只计算上三角避免重复）
        for i in range(n_users):
            for j in range(i + 1, n_users):
                # 计算余弦相似度：cosine distance → similarity
                vec_i = features[i]
                vec_j = features[j]
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)

                if norm_i == 0 or norm_j == 0:
                    continue

                similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)

                if similarity > threshold:
                    G.add_edge(i, j, weight=float(similarity))
                    edge_count += 1

        self.graph = G
        print(f"[属性相似度网络] 节点: {G.number_of_nodes()}, "
              f"边: {G.number_of_edges()}, "
              f"平均度: {np.mean([d for _, d in G.degree()]):.1f}")
        return G

    # -----------------------------------------------------------------------
    #  Barabási-Albert 无标度网络
    #  参考: Barabási & Albert (1999)
    # -----------------------------------------------------------------------
    def build_barabasi_albert(
        self,
        n: int = 1000,
        m: int = 3,
        seed: Optional[int] = 42,
    ) -> nx.Graph:
        """生成 Barabási-Albert 无标度网络

        BA模型模拟优先连接机制：新节点更倾向于连接已有高度节点，
        产生幂律度分布（少数节点拥有大量连接）。

        Parameters
        ----------
        n : int
            节点数
        m : int
            每个新节点添加的边数
        seed : int, optional
            随机种子

        Returns
        -------
        nx.Graph
            BA 无标度网络
        """
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        self.graph = G

        degrees = [d for _, d in G.degree()]
        print(f"[BA无标度网络] 节点: {G.number_of_nodes()}, "
              f"边: {G.number_of_edges()}, "
              f"平均度: {np.mean(degrees):.1f}, "
              f"最大度: {max(degrees)}")
        return G

    # -----------------------------------------------------------------------
    #  Watts-Strogatz 小世界网络
    #  参考: Watts & Strogatz (1998)
    # -----------------------------------------------------------------------
    def build_watts_strogatz(
        self,
        n: int = 1000,
        k: int = 6,
        p: float = 0.3,
        seed: Optional[int] = 42,
    ) -> nx.Graph:
        """生成 Watts-Strogatz 小世界网络

        WS模型从一个环形正则图开始，以概率 p 重新连接每条边，
        产生高聚类系数 + 短平均路径长度的"小世界"特性。

        Parameters
        ----------
        n : int
            节点数
        k : int
            每个节点的初始邻居数（必须为偶数）
        p : float
            边重连概率
        seed : int, optional
            随机种子

        Returns
        -------
        nx.Graph
            WS 小世界网络
        """
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        self.graph = G

        avg_path = (
            nx.average_shortest_path_length(G)
            if nx.is_connected(G)
            else float("inf")
        )
        clustering = nx.average_clustering(G)

        print(f"[WS小世界网络] 节点: {G.number_of_nodes()}, "
              f"边: {G.number_of_edges()}, "
              f"平均聚类系数: {clustering:.4f}, "
              f"平均路径长度: {avg_path:.2f}")
        return G

    # -----------------------------------------------------------------------
    #  基于 POI 共现构建网络
    #  如果两个用户在同一时间段访问了同一 POI，则建立连边
    # -----------------------------------------------------------------------
    def build_from_coorcurrence(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        poi_col: str = "poi_id",
    ) -> nx.Graph:
        """基于 POI 共现构建行为网络

        如果两个用户访问过相同的 POI，则在他们之间建立连边，
        边权重为共同访问的 POI 数量（归一化后）。

        Parameters
        ----------
        df : pd.DataFrame
            用户-POI 访问记录
        user_col : str
            用户ID列名
        poi_col : str
            POI ID列名

        Returns
        -------
        nx.Graph
            基于 POI 共现的网络
        """
        # 构建用户-POI 二部图的投影
        B = nx.Graph()
        user_nodes = df[user_col].unique()
        poi_nodes = df[poi_col].unique()

        B.add_nodes_from(user_nodes, bipartite=0)
        B.add_nodes_from(poi_nodes, bipartite=1)

        for _, row in df.iterrows():
            B.add_edge(row[user_col], row[poi_col])

        # 投影到用户侧
        G = nx.projected_graph(B, user_nodes)

        # 计算共现权重
        for u, v in G.edges():
            # 共同邻居 = 共同访问的 POI 数量
            common_pois = len(list(nx.common_neighbors(B, u, v)))
            G[u][v]["weight"] = float(common_pois)

        # 归一化权重到 [0, 1]
        if G.number_of_edges() > 0:
            max_weight = max(d["weight"] for _, _, d in G.edges(data=True))
            if max_weight > 0:
                for u, v, d in G.edges(data=True):
                    G[u][v]["weight"] = d["weight"] / max_weight

        self.graph = G

        degrees = [d for _, d in G.degree()]
        print(f"[POI共现网络] 用户节点: {G.number_of_nodes()}, "
              f"边: {G.number_of_edges()}, "
              f"平均度: {np.mean(degrees):.1f}" if degrees else "[POI共现网络] 空图")
        return G

    # -----------------------------------------------------------------------
    #  网络可视化
    # -----------------------------------------------------------------------
    def visualize(
        self,
        save_path: str = "output/network_visualization.png",
        max_nodes: int = 500,
    ) -> None:
        """网络可视化

        对于大型网络，仅显示子图以提高可读性。

        Parameters
        ----------
        save_path : str
            图片保存路径
        max_nodes : int
            可视化最大节点数
        """
        if self.graph is None:
            raise ValueError("请先构建网络（调用 build_* 方法）")

        G = self.graph

        # 大图取子图
        if G.number_of_nodes() > max_nodes:
            nodes = list(G.nodes())[:max_nodes]
            G_sub = G.subgraph(nodes).copy()
            print(f"[可视化] 网络过大，取前 {max_nodes} 个节点的子图")
        else:
            G_sub = G

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ---- 左图：网络拓扑 ----
        ax = axes[0]
        pos = nx.spring_layout(G_sub, seed=42, k=1.5 / np.sqrt(G_sub.number_of_nodes()))

        degrees = dict(G_sub.degree())
        node_sizes = [max(10, degrees[n] * 5) for n in G_sub.nodes()]

        nx.draw_networkx_edges(G_sub, pos, alpha=0.2, ax=ax)
        nx.draw_networkx_nodes(
            G_sub, pos, node_size=node_sizes,
            node_color=list(degrees.values()),
            cmap="YlOrRd", alpha=0.7, ax=ax,
        )
        ax.set_title("Social Network Topology", fontsize=14)
        ax.axis("off")

        # ---- 右图：度分布 ----
        ax2 = axes[1]
        all_degrees = [d for _, d in G.degree()]
        ax2.hist(all_degrees, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax2.set_xlabel("Degree", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Degree Distribution", fontsize=14)
        ax2.axvline(np.mean(all_degrees), color="red", linestyle="--",
                     label=f"Mean={np.mean(all_degrees):.1f}")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[可视化] 网络图已保存至 {save_path}")


# ===========================================================================
#  SocialContagion — 社会传染模型（SIR）
# ===========================================================================

class SocialContagion:
    """社会传染模型

    基于 SIR（Susceptible → Infected → Recovered）框架模拟补贴信息
    在社交网络中的传播过程。

    参考: Christakis & Fowler (2007); Centola & Macy (2007)

    状态定义:
    - S (Susceptible): 尚未感知到补贴信息的用户
    - I (Infected): 已感知补贴信息，可能受影响的用户
    - R (Recovered): 已核销或已过期的用户（不再传播）
    """

    # 节点状态常量
    SUSCEPTIBLE = "S"
    INFECTED = "I"
    RECOVERED = "R"

    def __init__(self):
        self.history: list[dict] = []  # 每步的状态记录

    # -----------------------------------------------------------------------
    #  SIR 传播模拟
    # -----------------------------------------------------------------------
    def propagate(
        self,
        G: nx.Graph,
        seed_nodes: list[int],
        contagion_rate: float = 0.1,
        n_steps: int = 20,
        price_sensitivity: Optional[dict[int, float]] = None,
        seed: Optional[int] = 42,
    ) -> dict:
        """社会传染 SIR 传播模拟

        补贴信息从种子节点出发，沿社交网络传播。
        传播概率 = 基础传染率 × 边权重 × 接收者价格敏感度。

        Parameters
        ----------
        G : nx.Graph
            社会网络
        seed_nodes : list[int]
            初始种子节点（最先感知补贴信息的用户）
        contagion_rate : float
            基础传染率
        n_steps : int
            模拟步数
        price_sensitivity : dict[int, float], optional
            节点 -> 价格敏感度 [0, 1]，敏感度越高越容易被"传染"
        seed : int, optional
            随机种子

        Returns
        -------
        dict
            包含:
            - 'states': 最终各节点状态 {node: state}
            - 'time_series': 每步的 S/I/R 计数
            - 'cascade_size': 最终感染总人数（含种子）
        """
        rng = np.random.RandomState(seed)

        # 初始化所有节点为 Susceptible
        states = {node: self.SUSCEPTIBLE for node in G.nodes()}

        # 种子节点设为 Infected
        for node in seed_nodes:
            if node in states:
                states[node] = self.INFECTED

        # 默认价格敏感度 = 1.0（所有节点相同）
        if price_sensitivity is None:
            price_sensitivity = {node: 1.0 for node in G.nodes()}

        time_series = []
        self.history = []

        for step in range(n_steps):
            s_count = sum(1 for s in states.values() if s == self.SUSCEPTIBLE)
            i_count = sum(1 for s in states.values() if s == self.INFECTED)
            r_count = sum(1 for s in states.values() if s == self.RECOVERED)

            time_series.append({
                "step": step,
                "S": s_count,
                "I": i_count,
                "R": r_count,
            })

            # 如果没有 Infected 节点，传播结束
            if i_count == 0:
                break

            new_states = dict(states)

            for node in list(G.nodes()):
                if states[node] != self.INFECTED:
                    continue

                # 尝试传染给每个 Susceptible 邻居
                for neighbor in G.neighbors(node):
                    if states[neighbor] != self.SUSCEPTIBLE:
                        continue

                    # 获取边权重
                    edge_data = G.get_edge_data(node, neighbor)
                    weight = edge_data["weight"] if edge_data else 1.0

                    # 传播概率 = 基础传染率 × 边权重 × 接收者价格敏感度
                    prob = contagion_rate * weight * price_sensitivity.get(neighbor, 1.0)
                    prob = min(prob, 1.0)  # 概率上限

                    if rng.random() < prob:
                        new_states[neighbor] = self.INFECTED

                # Infected 节点以一定概率转为 Recovered（信息衰减）
                # 恢复概率随步数增加而增加
                recovery_prob = 0.05 + 0.02 * step
                if rng.random() < recovery_prob:
                    new_states[node] = self.RECOVERED

            states = new_states

        # 统计最终结果
        final_infected = [
            node for node, s in states.items()
            if s == self.INFECTED or s == self.RECOVERED
        ]

        result = {
            "states": states,
            "time_series": time_series,
            "cascade_size": len(final_infected),
            "seed_size": len(seed_nodes),
            "cascade_ratio": len(final_infected) / max(G.number_of_nodes(), 1),
        }

        self.history = time_series
        print(f"[SIR传播] 种子节点: {len(seed_nodes)}, "
              f"级联规模: {result['cascade_size']}, "
              f"级联比例: {result['cascade_ratio']:.2%}")
        return result

    # -----------------------------------------------------------------------
    #  级联规模计算
    # -----------------------------------------------------------------------
    def compute_cascade_size(
        self,
        G: nx.Graph,
        seed_nodes: list[int],
        contagion_rate: float = 0.1,
        n_repeats: int = 50,
        seed: Optional[int] = 42,
    ) -> dict:
        """通过多次模拟计算平均级联规模

        Parameters
        ----------
        G : nx.Graph
            社会网络
        seed_nodes : list[int]
            种子节点列表
        contagion_rate : float
            基础传染率
        n_repeats : int
            重复模拟次数
        seed : int, optional
            随机种子

        Returns
        -------
        dict
            包含 mean, std, min, max, median
        """
        cascade_sizes = []

        for i in range(n_repeats):
            result = self.propagate(
                G, seed_nodes, contagion_rate,
                n_steps=20, seed=seed + i,
            )
            cascade_sizes.append(result["cascade_size"])

        cascade_sizes = np.array(cascade_sizes)
        stats = {
            "mean": float(np.mean(cascade_sizes)),
            "std": float(np.std(cascade_sizes)),
            "min": int(np.min(cascade_sizes)),
            "max": int(np.max(cascade_sizes)),
            "median": float(np.median(cascade_sizes)),
            "n_repeats": n_repeats,
        }

        print(f"[级联统计] 平均: {stats['mean']:.1f} ± {stats['std']:.1f}, "
              f"中位数: {stats['median']:.1f}, "
              f"范围: [{stats['min']}, {stats['max']}]")
        return stats

    # -----------------------------------------------------------------------
    #  社会效应估计（网络同伴效应）
    #  参考: Christakis & Fowler (2007) 的社会网络效应分析框架
    # -----------------------------------------------------------------------
    def estimate_social_effect(
        self,
        G: nx.Graph,
        treatment_nodes: list[int],
        outcome_attr: str = "outcome",
        n_simulations: int = 100,
        seed: Optional[int] = 42,
    ) -> dict:
        """估计社会效应（网络同伴效应）

        通过对比「有社交网络」vs「无社交网络（随机连接）」条件下的
        结果差异，估计社会传染效应的强度。

        方法:
        1. 在原始网络上运行 SIR 传播
        2. 在度保持的随机图上运行 SIR 传播（控制度分布）
        3. 两者级联规模之差即为社会效应估计

        参考: Christakis & Fowler (2007) — 区分同质性与因果传染效应

        Parameters
        ----------
        G : nx.Graph
            原始社会网络
        treatment_nodes : list[int]
            处理组节点（种子节点）
        outcome_attr : str
            结果属性名（用于标注节点）
        n_simulations : int
            每种条件下重复模拟次数
        seed : int, optional
            随机种子

        Returns
        -------
        dict
            包含原始网络和随机网络的级联统计，以及社会效应估计
        """
        rng = np.random.RandomState(seed)

        # --- 在原始网络上模拟 ---
        original_cascades = []
        for i in range(n_simulations):
            result = self.propagate(
                G, treatment_nodes, contagion_rate=0.1,
                n_steps=20, seed=seed + i,
            )
            original_cascades.append(result["cascade_size"])

        # --- 构建度保持的随机图（configuration model）---
        try:
            G_random = nx.configuration_model(
                [d for _, d in G.degree()],
                seed=seed,
            )
            # configuration_model 返回 MultiGraph，转为简单图
            G_random = nx.Graph(G_random)
            G_random.remove_edges_from(nx.selfloop_edges(G_random))

            random_cascades = []
            for i in range(n_simulations):
                # 确保种子节点在随机图中
                valid_seeds = [n for n in treatment_nodes if n in G_random.nodes()]
                if not valid_seeds:
                    random_cascades.append(0)
                    continue

                result = self.propagate(
                    G_random, valid_seeds, contagion_rate=0.1,
                    n_steps=20, seed=seed + 1000 + i,
                )
                random_cascades.append(result["cascade_size"])
        except Exception:
            # 如果 configuration model 失败，用 Erdős–Rényi 作为后备
            n = G.number_of_nodes()
            p = 2 * G.number_of_edges() / (n * (n - 1)) if n > 1 else 0
            G_random = nx.erdos_renyi_graph(n, p, seed=seed)

            random_cascades = []
            for i in range(n_simulations):
                valid_seeds = [n for n in treatment_nodes if n in G_random.nodes()]
                if not valid_seeds:
                    random_cascades.append(0)
                    continue

                result = self.propagate(
                    G_random, valid_seeds, contagion_rate=0.1,
                    n_steps=20, seed=seed + 1000 + i,
                )
                random_cascades.append(result["cascade_size"])

        original_mean = np.mean(original_cascades)
        random_mean = np.mean(random_cascades)
        social_effect = original_mean - random_mean

        estimate = {
            "original_network_mean": float(original_mean),
            "original_network_std": float(np.std(original_cascades)),
            "random_network_mean": float(random_mean),
            "random_network_std": float(np.std(random_cascades)),
            "social_effect": float(social_effect),
            "social_effect_ratio": float(social_effect / max(random_mean, 1)),
            "n_simulations": n_simulations,
            "method": "configuration_model",
        }

        print(f"[社会效应估计] 原始网络级联均值: {original_mean:.1f}, "
              f"随机网络级联均值: {random_mean:.1f}, "
              f"社会效应: {social_effect:+.1f} "
              f"({social_effect / max(random_mean, 1):+.1%})")
        return estimate

    # -----------------------------------------------------------------------
    #  可视化 SIR 传播时间序列
    # -----------------------------------------------------------------------
    def plot_sir_curve(
        self,
        time_series: list[dict],
        save_path: str = "output/sir_curve.png",
    ) -> None:
        """绘制 SIR 传播曲线

        Parameters
        ----------
        time_series : list[dict]
            propagate() 返回的 time_series
        save_path : str
            图片保存路径
        """
        steps = [t["step"] for t in time_series]
        s_vals = [t["S"] for t in time_series]
        i_vals = [t["I"] for t in time_series]
        r_vals = [t["R"] for t in time_series]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, s_vals, "b-", label="Susceptible", linewidth=2)
        ax.plot(steps, i_vals, "r-", label="Infected", linewidth=2)
        ax.plot(steps, r_vals, "g-", label="Recovered", linewidth=2)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Number of Users", fontsize=12)
        ax.set_title("SIR Social Contagion Curve", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[可视化] SIR 曲线已保存至 {save_path}")


# ===========================================================================
#  NetworkContagionAgent — 结合 Agent 模型的社会传染
# ===========================================================================

class NetworkContagionAgent:
    """网络传染 Agent 模型

    将社会网络与个体 Agent 模型结合，模拟每个用户（Agent）在受到
    社交网络影响下的补贴核销行为。

    核心机制:
    - 每个 Agent 有独立的核销概率（受自身属性影响）
    - Agent 的核销概率受邻居行为的影响（社会压力）
    - 邻居核销率越高 → 自身核销概率越大

    参考: Christakis & Fowler (2007) — 社会网络中的行为传播
    """

    def __init__(
        self,
        G: nx.Graph,
        base_probability: Optional[dict[int, float]] = None,
        social_pressure_factor: float = 0.3,
        seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        G : nx.Graph
            社会网络
        base_probability : dict[int, float], optional
            节点 -> 基础核销概率（无社交影响时的概率）
        social_pressure_factor : float
            社会压力因子 [0, 1]，
            控制邻居行为对自身核销概率的影响程度
        seed : int, optional
            随机种子
        """
        self.G = G
        self.social_pressure_factor = social_pressure_factor
        self.rng = np.random.RandomState(seed)

        # 默认基础核销概率
        if base_probability is None:
            self.base_probability = {
                node: 0.3 for node in G.nodes()
            }
        else:
            self.base_probability = base_probability

        # Agent 状态
        self.redeemed: dict[int, bool] = {node: False for node in G.nodes()}
        self.redeem_step: dict[int, int] = {}  # 记录核销发生的步数

        # 历史记录
        self.history: list[dict] = []

    def compute_social_pressure(self, node: int) -> float:
        """计算节点受到的社会压力

        社会压力 = 邻居中已核销的比例

        参考: Centola & Macy (2007) — 复杂传染需要多个社会接触的"强化"

        Parameters
        ----------
        node : int
            目标节点

        Returns
        -------
        float
            社会压力值 [0, 1]
        """
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return 0.0

        redeemed_neighbors = sum(
            1 for n in neighbors if self.redeemed.get(n, False)
        )
        return redeemed_neighbors / len(neighbors)

    def compute_adjusted_probability(self, node: int) -> float:
        """计算经社交压力调整后的核销概率

        P_adjusted = P_base × (1 - social_pressure_factor)
                   + social_pressure × social_pressure_factor

        社会压力因子越大，邻居行为对自身影响越强。
        极端情况:
        - factor=0: 完全独立决策
        - factor=1: 完全随大流

        Parameters
        ----------
        node : int
            目标节点

        Returns
        -------
        float
            调整后的核销概率
        """
        p_base = self.base_probability.get(node, 0.3)
        pressure = self.compute_social_pressure(node)

        p_adjusted = (
            p_base * (1 - self.social_pressure_factor)
            + pressure * self.social_pressure_factor
        )
        return p_adjusted

    def simulate(
        self,
        n_steps: int = 10,
        initial_redeemed: Optional[list[int]] = None,
    ) -> dict:
        """运行 Agent 仿真

        Parameters
        ----------
        n_steps : int
            模拟步数
        initial_redeemed : list[int], optional
            初始已核销的节点（如补贴领取后立即核销的用户）

        Returns
        -------
        dict
            包含:
            - 'redeemed': 最终核销状态
            - 'total_redeemed': 总核销人数
            - 'time_series': 每步累计核销人数
            - 'social_lift': 社会效应提升率
        """
        # 初始化
        self.redeemed = {node: False for node in self.G.nodes()}
        self.redeem_step = {}
        self.history = []

        if initial_redeemed:
            for node in initial_redeemed:
                if node in self.G.nodes():
                    self.redeemed[node] = True
                    self.redeem_step[node] = 0

        total_nodes = self.G.number_of_nodes()
        time_series = []

        for step in range(1, n_steps + 1):
            new_redeemed_this_step = 0

            # 随机顺序处理节点，避免顺序偏差
            nodes = list(self.G.nodes())
            self.rng.shuffle(nodes)

            for node in nodes:
                if self.redeemed[node]:
                    continue

                p_adjusted = self.compute_adjusted_probability(node)

                if self.rng.random() < p_adjusted:
                    self.redeemed[node] = True
                    self.redeem_step[node] = step
                    new_redeemed_this_step += 1

            cumulative_redeemed = sum(1 for v in self.redeemed.values() if v)
            time_series.append({
                "step": step,
                "new_redeemed": new_redeemed_this_step,
                "cumulative_redeemed": cumulative_redeemed,
                "redeem_rate": cumulative_redeemed / total_nodes,
            })

            # 如果所有节点都已核销，提前终止
            if cumulative_redeemed == total_nodes:
                break

        self.history = time_series

        # 计算社会效应提升率：对比无社交影响的理论核销率
        theoretical_redeemed = sum(
            1 - (1 - self.base_probability.get(n, 0.3)) ** n_steps
            for n in self.G.nodes()
        )
        actual_redeemed = sum(1 for v in self.redeemed.values() if v)
        social_lift = (
            (actual_redeemed - theoretical_redeemed) / max(theoretical_redeemed, 1)
        )

        result = {
            "redeemed": dict(self.redeemed),
            "total_redeemed": actual_redeemed,
            "redeem_rate": actual_redeemed / total_nodes,
            "time_series": time_series,
            "social_lift": float(social_lift),
            "theoretical_redeemed": float(theoretical_redeemed),
        }

        print(f"[Agent仿真] 步数: {len(time_series)}, "
              f"总核销: {actual_redeemed}/{total_nodes} ({result['redeem_rate']:.1%}), "
              f"社会提升: {social_lift:+.1%}")
        return result

    def plot_redeem_curve(
        self,
        time_series: list[dict],
        save_path: str = "output/agent_redeem_curve.png",
    ) -> None:
        """绘制 Agent 核销曲线

        Parameters
        ----------
        time_series : list[dict]
            simulate() 返回的 time_series
        save_path : str
            图片保存路径
        """
        steps = [t["step"] for t in time_series]
        rates = [t["redeem_rate"] * 100 for t in time_series]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(steps, rates, color="steelblue", alpha=0.7, edgecolor="white")
        ax.plot(steps, rates, "ro-", markersize=6, linewidth=2, label="Redeem Rate")
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Redeem Rate (%)", fontsize=12)
        ax.set_title(
            f"Agent Redeem Curve (social_pressure_factor={self.social_pressure_factor})",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[可视化] 核销曲线已保存至 {save_path}")
