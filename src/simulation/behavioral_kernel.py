"""
行为链核提取模块 (Behavioral Kernel Extraction)
================================================

核心思想：从每个用户的长期行为链数据中，提取一组紧凑的、可脱敏的参数化表示，
作为仿真 Agent 的"核"（Kernel），替代硬编码的全局参数。

三层架构：
  Layer 1 (Raw Chain): 用户原始行为序列 (browse→click→cart→pay→redeem)
  Layer 2 (Estimation): 从序列中估计三组个体级参数
    - theta_i: 基础漏斗转换率 P(step_{t+1} | step_t, X_i)
    - beta_i:  补贴敏感度 Delta P(step | subsidy)
    - gamma_i: 认知调制参数 (心理账户、疲劳等)
  Layer 3 (Kernel): 加差分隐私噪声后的可发布参数包

方法论参考：
- ECUP (Entire Chain Uplift Modeling, WWW 2024): 全链路条件概率建模
- Bayesian Hierarchical Estimation: 小样本下的个体参数收缩
- Differential Privacy (Dwork et al., 2006): 参数发布时的隐私保护
- Li & Kannan (2016): 多触点归因模型

作者注：
真实业务场景中，平台拥有每个用户的长期行为链，可以直接拟合个体参数。
竞赛数据样本量有限时，采用层级贝叶斯收缩——借总体信息补个体不足。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


# ===========================================================================
# 数据结构定义
# ===========================================================================

class ChainStep(str, Enum):
    """行为链步骤枚举"""
    CLICKED = "clicked"
    CARTED = "carted"
    PAID = "paid"
    REDEEMED = "redeemed"

    @classmethod
    def funnel_steps(cls) -> list["ChainStep"]:
        """有序漏斗步骤列表"""
        return [cls.CLICKED, cls.CARTED, cls.PAID, cls.REDEEMED]


@dataclass
class UserKernel:
    """
    用户行为核——个体级参数化表示

    三个分量：
    - theta: 基础漏斗转换率 (dict[ChainStep, float])
    - beta: 补贴敏感度 (dict[ChainStep, float])
    - gamma: 认知调制参数 (dict[str, float])
    - metadata: 可选元数据 (e.g., sample_size, confidence)

    数学形式：
      P(step_{t+1} | step_t, subsidy) = sigmoid(
          logit(theta[step]) + beta[step] * subsidy_effect + gamma_modifiers
      )

    其中 subsidy_effect = subsidy * price_sensitivity (前景理论调制)
    """
    user_id: str
    theta: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    gamma: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # 隐私保护参数
    _epsilon: float = float("inf")  # DP epsilon (inf = 无噪声)

    @property
    def steps(self) -> list[str]:
        """有序漏斗步骤名"""
        return [s.value for s in ChainStep.funnel_steps()]

    def get_theta(self, step: str, default: float = 0.5) -> float:
        """获取某步骤的基础转换率"""
        return self.theta.get(step, default)

    def get_beta(self, step: str, default: float = 0.0) -> float:
        """获取某步骤的补贴敏感度"""
        return self.beta.get(step, default)

    def get_gamma(self, key: str, default: float = 0.0) -> float:
        """获取认知调制参数"""
        return self.gamma.get(key, default)

    def to_vector(self) -> np.ndarray:
        """将核参数展开为向量（用于分布拟合）"""
        vec = []
        for step in self.steps:
            vec.append(self.theta.get(step, 0.5))
        for step in self.steps:
            vec.append(self.beta.get(step, 0.0))
        for key in sorted(self.gamma.keys()):
            vec.append(self.gamma[key])
        return np.array(vec)

    @classmethod
    def from_vector(cls, vec: np.ndarray, user_id: str = "",
                    gamma_keys: Optional[list[str]] = None) -> "UserKernel":
        """从向量重建核参数"""
        if gamma_keys is None:
            gamma_keys = ["alpha", "lambda_", "fatigue_rate", "account_eta"]

        n_steps = len(ChainStep.funnel_steps())
        theta = {}
        for i, step in enumerate(ChainStep.funnel_steps()):
            theta[step.value] = float(np.clip(vec[i], 0.01, 0.99)) if i < len(vec) else 0.5

        beta = {}
        for i, step in enumerate(ChainStep.funnel_steps()):
            idx = n_steps + i
            beta[step.value] = float(vec[idx]) if idx < len(vec) else 0.0

        gamma = {}
        for i, key in enumerate(gamma_keys):
            idx = 2 * n_steps + i
            gamma[key] = float(vec[idx]) if idx < len(vec) else 0.0

        return cls(user_id=user_id, theta=theta, beta=beta, gamma=gamma)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "user_id": self.user_id,
            "theta": self.theta,
            "beta": self.beta,
            "gamma": self.gamma,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        theta_str = ", ".join(f"{k}={v:.3f}" for k, v in self.theta.items())
        return f"UserKernel(id={self.user_id}, theta={{{theta_str}}})"


# ===========================================================================
# Layer 2: 参数估计算法
# ===========================================================================

class BehavioralKernelExtractor:
    """
    行为链核提取器

    从用户行为链数据中提取三组个体级参数：
    1. theta (基础转换率): 层级贝叶斯估计，小样本向总体均值收缩
    2. beta (补贴敏感度): 基于处理/对照差异的条件效应估计
    3. gamma (认知调制): 从行为模式推断的心理账户参数

    核心算法：
    -------

    (1) theta 估计——层级贝叶斯收缩 (Empirical Bayes):
        对于用户 i，步骤 s 的基础转换率：
          theta_{i,s} | mu_s, tau_s ~ Beta(mu_s * tau_s, (1-mu_s) * tau_s)
          x_{i,s} | theta_{i,s} ~ Binomial(n_{i,s}, theta_{i,s})

        后验均值（James-Stein 收缩）：
          hat_theta_{i,s} = (n_{i,s} * hat_theta_MLE + tau_s * mu_s) / (n_{i,s} + tau_s)

        其中 tau_s 从总体矩估计得到，控制收缩强度。

    (2) beta 估计——条件效应分解：
        对于有处理/对照标记的数据：
          beta_{i,s} = P(step_s | treated) - P(step_s | control)

        小样本时收缩到总体均值：
          hat_beta_{i,s} = (n_i * beta_MLE + tau_beta * mu_beta) / (n_i + tau_beta)

    (3) gamma 估计——行为模式推断：
        从行为链中提取认知参数：
          - alpha (前景曲率): 从支付/核销的非线性补贴响应推断
          - fatigue_rate: 从补贴脱敏速度推断
          - account_eta: 从参考点更新速度推断

    参数：
    ------
    - shrinkage_factor: 收缩强度 (0=纯MLE, 1=纯总体均值)，默认自动估计
    - min_samples: 最小样本量，低于此值强收缩到总体
    - dp_epsilon: 差分隐私参数，inf=不加噪声
    """

    def __init__(
        self,
        shrinkage_factor: Optional[float] = None,
        min_samples: int = 5,
        dp_epsilon: float = float("inf"),
        random_state: int = 42,
    ):
        self.shrinkage_factor = shrinkage_factor  # None=自动估计
        self.min_samples = min_samples
        self.dp_epsilon = dp_epsilon
        self.rng = np.random.RandomState(random_state)

        # 拟合后存储的总体参数
        self._population_theta: dict[str, tuple[float, float]] = {}  # {step: (mu, tau)}
        self._population_beta: dict[str, tuple[float, float]] = {}
        self._population_gamma: dict[str, tuple[float, float]] = {}

    # ---- 核心接口 ----

    def fit_extract(
        self,
        behavior_chains: pd.DataFrame,
        user_col: str = "user_id",
        step_col: str = "action",
        time_col: str = "timestamp",
        subsidy_col: Optional[str] = "subsidy_amount",
        additional_cols: Optional[list[str]] = None,
    ) -> list[UserKernel]:
        """
        从行为链数据中提取所有用户的核参数

        参数：
        - behavior_chains: 行为链 DataFrame，每行一条行为记录
        - user_col: 用户ID列名
        - step_col: 行为步骤列名 (值应为 clicked/carted/paid/redeemed)
        - time_col: 时间戳列名
        - subsidy_col: 补贴金额列名（有则估计 beta，无则 beta=0）
        - additional_cols: 额外特征列名（用于 gamma 估计）

        返回：
        - List[UserKernel]
        """
        # Step 1: 构建每个用户的行为链统计
        user_stats = self._compute_user_stats(
            behavior_chains, user_col, step_col, time_col, subsidy_col
        )

        # Step 2: 估计总体参数（层级先验）
        self._estimate_population_params(user_stats)

        # Step 3: 对每个用户进行层级贝叶斯收缩估计
        kernels = []
        for user_id, stats in user_stats.items():
            kernel = self._extract_single_kernel(user_id, stats)
            kernels.append(kernel)

        # Step 4: 差分隐私噪声（如果 epsilon < inf）
        if self.dp_epsilon < float("inf"):
            kernels = self._apply_dp_noise(kernels)

        return kernels

    def extract_from_sequences(
        self,
        sequences: dict[str, list[dict]],
        has_subsidy: bool = True,
    ) -> list[UserKernel]:
        """
        从预格式化的行为序列中提取核参数

        参数：
        - sequences: {user_id: [{"step": "clicked", "subsidy": 10.0, ...}, ...]}
        - has_subsidy: 序列中是否包含补贴信息

        返回：
        - List[UserKernel]
        """
        # 转换为 DataFrame 格式
        rows = []
        for user_id, events in sequences.items():
            for event in events:
                row = {"user_id": user_id, "action": event.get("step", "")}
                if has_subsidy and "subsidy" in event:
                    row["subsidy_amount"] = event["subsidy"]
                if "timestamp" in event:
                    row["timestamp"] = event["timestamp"]
                rows.append(row)

        if not rows:
            return []

        df = pd.DataFrame(rows)
        subsidy_col = "subsidy_amount" if has_subsidy and "subsidy_amount" in df.columns else None
        return self.fit_extract(df, subsidy_col=subsidy_col)

    # ---- 内部实现 ----

    def _compute_user_stats(
        self,
        df: pd.DataFrame,
        user_col: str,
        step_col: str,
        time_col: str,
        subsidy_col: Optional[str],
    ) -> dict[str, dict]:
        """
        计算每个用户的行为链统计量

        核心：theta 的语义是**条件概率** P(step | prev_step)，
        即"到达了前一步的会话中，有多少继续到当前步"。

        漏斗逻辑：browsed → clicked → carted → paid → redeemed
        - P(clicked | browsed) = n_clicked / n_sessions
        - P(carted  | clicked) = n_carted  / n_clicked
        - P(paid    | carted)  = n_paid    / n_carted
        - P(redeemed| paid)    = n_redeemed/ n_paid

        返回 {user_id: {step: {"n_reached": int, "n_prior": int, ...}}}
        """
        user_stats = {}
        steps = ChainStep.funnel_steps()
        # 漏斗步骤链：browsed 是 clicked 的前驱，clicked 是 carted 的前驱...
        step_chain = {
            "clicked": "browsed",
            "carted": "clicked",
            "paid": "carted",
            "redeemed": "paid",
        }

        for user_id, group in df.groupby(user_col):
            stats = {}
            actions = group[step_col].values

            # 统计各步骤出现次数
            step_counts = {}
            for action_val in actions:
                a = str(action_val).strip().lower()
                step_counts[a] = step_counts.get(a, 0) + 1

            n_sessions = step_counts.get("browsed", len(group))

            for step in steps:
                step_name = step.value
                n_reached = step_counts.get(step_name, 0)

                # 前驱步骤的计数
                prior_step = step_chain[step_name]
                n_prior = step_counts.get(prior_step, n_sessions)

                stats[step_name] = {
                    "n_reached": n_reached,
                    "n_prior": n_prior,
                    "n_sessions": n_sessions,
                }

            # 如果有补贴信息，分别统计处理/对照的条件概率
            if subsidy_col and subsidy_col in group.columns:
                treated = group[group[subsidy_col] > 0]
                control = group[group[subsidy_col] == 0]

                t_actions = treated[step_col].values
                c_actions = control[step_col].values

                t_counts = {}
                for a in t_actions:
                    t_counts[str(a).strip().lower()] = t_counts.get(str(a).strip().lower(), 0) + 1

                c_counts = {}
                for a in c_actions:
                    c_counts[str(a).strip().lower()] = c_counts.get(str(a).strip().lower(), 0) + 1

                n_t_sessions = t_counts.get("browsed", len(treated))
                n_c_sessions = c_counts.get("browsed", len(control))

                for step in steps:
                    step_name = step.value
                    prior_step = step_chain[step_name]
                    n_t_reached = t_counts.get(step_name, 0)
                    n_t_prior = t_counts.get(prior_step, n_t_sessions)
                    n_c_reached = c_counts.get(step_name, 0)
                    n_c_prior = c_counts.get(prior_step, n_c_sessions)

                    stats[step_name].update({
                        "n_t_reached": n_t_reached,
                        "n_t_prior": n_t_prior,
                        "n_t_sessions": n_t_sessions,
                        "n_c_reached": n_c_reached,
                        "n_c_prior": n_c_prior,
                        "n_c_sessions": n_c_sessions,
                    })

            stats["_user_id"] = user_id
            user_stats[str(user_id)] = stats

        return user_stats

    def _estimate_population_params(self, user_stats: dict[str, dict]) -> None:
        """
        估计总体先验参数 (mu, tau) 用于层级贝叶斯收缩

        对每一步 s:
          mu_s = 总体平均条件转换率
          tau_s = 总体精度参数（控制收缩强度）
        """
        steps = ChainStep.funnel_steps()

        for step in steps:
            step_name = step.value
            rates = []
            beta_rates = []

            for uid, stats in user_stats.items():
                s = stats.get(step_name, {})
                n_reached = s.get("n_reached", 0)
                n_prior = s.get("n_prior", 1)
                if n_prior > 0:
                    rates.append(n_reached / n_prior)

                # 补贴敏感度：处理组率 - 对照组率
                n_t_reached = s.get("n_t_reached", 0)
                n_t_prior = s.get("n_t_prior", 0)
                n_c_reached = s.get("n_c_reached", 0)
                n_c_prior = s.get("n_c_prior", 0)
                if n_t_prior > 0 and n_c_prior > 0:
                    rate_t = n_t_reached / n_t_prior
                    rate_c = n_c_reached / n_c_prior
                    beta_rates.append(rate_t - rate_c)

            # 总体均值
            mu = float(np.mean(rates)) if rates else 0.5
            # 总体精度（method of moments 估计 Beta 分布参数）
            var = float(np.var(rates)) if len(rates) > 1 else 0.01
            # tau = mu*(1-mu)/var - 1 (Beta分布的precision)
            tau = max(mu * (1 - mu) / max(var, 1e-6) - 1, 2.0)
            self._population_theta[step_name] = (mu, tau)

            # 补贴敏感度总体参数
            mu_beta = float(np.mean(beta_rates)) if beta_rates else 0.0
            var_beta = float(np.var(beta_rates)) if len(beta_rates) > 1 else 0.01
            tau_beta = max(1.0 / max(var_beta, 1e-6), 2.0)
            self._population_beta[step_name] = (mu_beta, tau_beta)

    def _extract_single_kernel(self, user_id: str, stats: dict) -> UserKernel:
        """
        对单个用户进行层级贝叶斯估计

        核心公式（James-Stein 收缩）：
          hat_theta = (n * MLE + tau * mu) / (n + tau)

        n 小时（样本少）→ 强收缩到总体均值
        n 大时（样本多）→ 接近 MLE
        """
        theta = {}
        beta = {}
        steps = ChainStep.funnel_steps()

        for step in steps:
            step_name = step.value
            s = stats.get(step_name, {})
            n_reached = s.get("n_reached", 0)
            n_prior = s.get("n_prior", 1)

            # MLE 估计（条件概率语义）
            mle = n_reached / max(n_prior, 1)

            # 层级收缩
            mu, tau = self._population_theta.get(step_name, (0.5, 10.0))

            # 数据驱动收缩：样本少→强收缩，样本多→弱收缩
            effective_n = max(n_prior, 1)
            shrinkage = tau / (effective_n + tau)

            # 最小样本量保护
            if n_prior < self.min_samples:
                shrinkage = max(shrinkage, 0.8)  # 强收缩到总体

            # 自定义收缩因子覆盖
            if self.shrinkage_factor is not None:
                shrinkage = self.shrinkage_factor

            # 后验均值
            theta[step_name] = float(np.clip(
                mle * (1 - shrinkage) + mu * shrinkage,
                0.01, 0.99
            ))

            # beta 估计（补贴敏感度）：处理组率 - 对照组率
            n_t_reached = s.get("n_t_reached", 0)
            n_t_prior = s.get("n_t_prior", 0)
            n_c_reached = s.get("n_c_reached", 0)
            n_c_prior = s.get("n_c_prior", 0)

            if n_t_prior > 0 and n_c_prior > 0:
                rate_t = n_t_reached / n_t_prior
                rate_c = n_c_reached / n_c_prior
                beta_mle = rate_t - rate_c

                mu_beta, tau_beta = self._population_beta.get(step_name, (0.0, 10.0))
                total_sub = n_t_prior + n_c_prior
                shrinkage_beta = tau_beta / (total_sub + tau_beta)
                if total_sub < self.min_samples:
                    shrinkage_beta = max(shrinkage_beta, 0.8)

                beta[step_name] = float(beta_mle * (1 - shrinkage_beta) + mu_beta * shrinkage_beta)
            else:
                beta[step_name] = 0.0

        # gamma 估计（认知调制参数）
        gamma = self._estimate_gamma(user_id, stats)

        return UserKernel(
            user_id=user_id,
            theta=theta,
            beta=beta,
            gamma=gamma,
            metadata={
                "n_sessions": stats.get(step_name, {}).get("n_sessions", 0),
                "extraction_method": "hierarchical_bayesian",
                "shrinkage": self.shrinkage_factor or "auto",
            },
        )

    def _estimate_gamma(self, user_id: str, stats: dict) -> dict[str, float]:
        """
        从行为模式推断认知调制参数

        推断逻辑（基于行为经济学理论）：
        - alpha (前景曲率): 核销率随补贴金额增加的凹度→alpha
        - fatigue_rate: 连续补贴后核销率下降的速度
        - account_eta: 参考点更新速率（从补贴脱敏模式推断）
        - price_sensitivity: 从补贴敏感度推断
        """
        gamma = {}

        # price_sensitivity: 从补贴是否显著提高各步率来推断
        beta_vals = []
        for step in ChainStep.funnel_steps():
            s = stats.get(step.value, {})
            n_t_reached = s.get("n_t_reached", 0)
            n_c_reached = s.get("n_c_reached", 0)
            n_t_prior = s.get("n_t_prior", 1)
            n_c_prior = s.get("n_c_prior", 1)
            if n_t_prior > 0 and n_c_prior > 0:
                beta_vals.append(n_t_reached / n_t_prior - n_c_reached / n_c_prior)

        if beta_vals:
            gamma["price_sensitivity"] = float(np.clip(
                np.mean(beta_vals) * 3.0 + 0.5, 0.0, 1.0
            ))
        else:
            gamma["price_sensitivity"] = 0.5

        # alpha: 默认前景理论值（需要更细粒度数据才能个体化）
        gamma["alpha"] = 0.88  # Kahneman & Tversky 标准值

        # lambda_: fatigue_rate, account_eta: 从行为链深度推断
        paid_stats = stats.get("paid", {})
        clicked_stats = stats.get("clicked", {})
        n_paid = paid_stats.get("n_reached", 0)
        n_clicked = clicked_stats.get("n_reached", 1)
        paid_rate = n_paid / max(n_clicked, 1)
        gamma["fatigue_rate"] = float(np.clip(0.15 * (1.0 - paid_rate) + 0.05, 0.01, 0.5))
        gamma["account_eta"] = float(np.clip(0.35 * paid_rate + 0.1, 0.05, 0.6))
        gamma["lambda_"] = 2.25  # 损失厌恶标准值

        return gamma

    def _apply_dp_noise(self, kernels: list[UserKernel]) -> list[UserKernel]:
        """
        差分隐私噪声注入 (Laplace 机制)

        对于每个参数 theta_{i,s}，注入 Laplace(0, Delta_f / epsilon) 噪声，
        其中 Delta_f 是灵敏度上界（参数变化的最大范围）。

        参考: Dwork et al. (2006), "Calibrating Noise to Sensitivity in Private Data Analysis"
        """
        epsilon = self.dp_epsilon
        if epsilon <= 0:
            return kernels

        # 灵敏度：参数在 [0,1] 范围内变化，Delta_f = 1
        sensitivity = 1.0
        scale = sensitivity / epsilon

        for kernel in kernels:
            # theta 参数加噪
            for step in kernel.theta:
                noise = self.rng.laplace(0, scale)
                kernel.theta[step] = float(np.clip(kernel.theta[step] + noise, 0.01, 0.99))

            # beta 参数加噪（范围可能超出 [0,1]，但仍裁剪到合理范围）
            for step in kernel.beta:
                noise = self.rng.laplace(0, scale)
                kernel.beta[step] = float(np.clip(kernel.beta[step] + noise, -1.0, 1.0))

            # gamma 参数加噪（敏感度可能不同，这里用相同 scale）
            for key in kernel.gamma:
                noise = self.rng.laplace(0, scale)
                # gamma 范围取决于具体参数，做简单裁剪
                kernel.gamma[key] = float(np.clip(kernel.gamma[key] + noise, -5.0, 10.0))

            kernel._epsilon = epsilon

        return kernels

    def get_population_summary(self) -> dict[str, Any]:
        """获取总体参数摘要（拟合后可用）"""
        summary = {"theta": {}, "beta": {}}
        for step, (mu, tau) in self._population_theta.items():
            summary["theta"][step] = {"mu": mu, "tau": tau}
        for step, (mu, tau) in self._population_beta.items():
            summary["beta"][step] = {"mu": mu, "tau": tau}
        return summary


# ===========================================================================
# Layer 3: 种群合成采样器
# ===========================================================================

class KernelPopulationSampler:
    """
    核种群合成采样器

    从已提取的核参数中拟合多变量分布，然后采样生成新的 Agent 种群。
    用途：当只有聚合统计量时，合成具有真实分布特征的仿真种群。

    方法：
    1. 将所有 UserKernel 向量化
    2. 拟合多变量正态分布（或核密度估计）
    3. 采样 → 反向映射回 UserKernel

    数学：
    设 K_i = (theta_i, beta_i, gamma_i) 为第 i 个用户的核向量
    拟合: K ~ N(mu, Sigma)
    采样: K_new ~ N(mu, Sigma)
    映射: K_new → UserKernel → AgentConfig
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
        self._fitted = False
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._gamma_keys: Optional[list[str]] = None
        self._n_kernels: int = 0

    def fit(self, kernels: list[UserKernel]) -> None:
        """
        从已提取的核参数中拟合多变量正态分布

        参数：
        - kernels: 已提取的 UserKernel 列表
        """
        if not kernels:
            raise ValueError("No kernels provided for fitting")

        # 确定gamma键名
        self._gamma_keys = sorted(kernels[0].gamma.keys()) if kernels[0].gamma else []

        # 向量化
        vectors = np.array([k.to_vector() for k in kernels])

        # 拟合多变量正态
        self._mean = np.mean(vectors, axis=0)
        self._cov = np.cov(vectors, rowvar=False)

        # 确保协方差矩阵正定
        if self._cov.ndim == 0:
            self._cov = np.array([[self._cov]])
        # 小样本修正：加对角正则化
        self._cov += np.eye(self._cov.shape[0]) * 1e-6

        self._n_kernels = len(kernels)
        self._fitted = True

    def sample(self, n: int, clip_theta: bool = True) -> list[UserKernel]:
        """
        从拟合的分布中采样新的核参数

        参数：
        - n: 采样数量
        - clip_theta: 是否裁剪 theta 到 [0.01, 0.99]

        返回：
        - List[UserKernel]
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before sample()")

        # 多变量正态采样
        vectors = self.rng.multivariate_normal(self._mean, self._cov, size=n)

        kernels = []
        for i, vec in enumerate(vectors):
            kernel = UserKernel.from_vector(vec, user_id=f"synthetic_{i}", gamma_keys=self._gamma_keys)

            if clip_theta:
                for step in kernel.theta:
                    kernel.theta[step] = float(np.clip(kernel.theta[step], 0.01, 0.99))

            kernel.metadata["source"] = "population_synthesis"
            kernel.metadata["n_reference_kernels"] = self._n_kernels
            kernels.append(kernel)

        return kernels

    def sample_agent_configs(
        self,
        n: int,
        base_gtv_rng: Optional[np.random.RandomState] = None,
    ) -> list["AgentConfig"]:
        """
        直接采样生成 AgentConfig 列表（可直接注入仿真模型）

        参数：
        - n: 采样数量
        - base_gtv_rng: 用于生成 base_gtv 的 RNG（如为 None 则自动创建）

        返回：
        - List[AgentConfig]
        """
        from src.simulation.mesa_agent_model import AgentConfig

        kernels = self.sample(n)

        if base_gtv_rng is None:
            base_gtv_rng = np.random.RandomState(42)

        configs = []
        for i, kernel in enumerate(kernels):
            # 从核参数映射到 AgentConfig
            ps = kernel.gamma.get("price_sensitivity", 0.5)
            # income_level 从 price_sensitivity 反推（大致映射）
            income = int(np.clip(5 - ps * 4, 1, 5))

            configs.append(AgentConfig(
                agent_id=i,
                price_sensitivity=ps,
                income_level=income,
                city_tier=3,  # 默认中等城市
                base_gtv=base_gtv_rng.lognormal(3.5, 0.8),
                alpha=kernel.gamma.get("alpha", 0.88),
                lambda_=kernel.gamma.get("lambda_", 2.25),
                decision_threshold=0.3,
                behavior_chain_enabled=True,
            ))

        return configs

    def get_distribution_stats(self) -> dict[str, Any]:
        """获取拟合的分布参数"""
        if not self._fitted:
            return {"fitted": False}

        steps = [s.value for s in ChainStep.funnel_steps()]
        theta_mean = {step: float(self._mean[i]) for i, step in enumerate(steps)}
        beta_mean = {step: float(self._mean[len(steps) + i]) for i, step in enumerate(steps)}

        return {
            "fitted": True,
            "n_reference_kernels": self._n_kernels,
            "theta_mean": theta_mean,
            "beta_mean": beta_mean,
            "gamma_mean": {k: float(v) for k, v in zip(
                self._gamma_keys or [],
                self._mean[2 * len(steps):] if self._mean is not None else []
            )},
        }


# ===========================================================================
# 便捷函数：从核参数驱动Agent决策概率
# ===========================================================================

def kernel_modulated_prob(
    base_rate: float,
    kernel: UserKernel,
    step: str,
    subsidy_amount: float = 0.0,
    price_sensitivity: float = 0.5,
    fatigue: float = 0.0,
    cognitive_load: float = 3.0,
) -> float:
    """
    用核参数调制行为链各步的决策概率

    数学公式：
      P(step | kernel, subsidy) = sigma(
          logit(base_rate)
          + beta[step] * log(1 + subsidy/10) * price_sensitivity
          + gamma[alpha] * prospect_value(subsidy)
          + gamma[fatigue_rate] * fatigue_discount
          + bounded_rationality_discount(cognitive_load)
      )

    参数：
    - base_rate: 从 kernel.theta[step] 获取的基础转换率
    - kernel: 用户核参数
    - step: 当前链步骤 (clicked/carted/paid/redeemed)
    - subsidy_amount: 补贴金额
    - price_sensitivity: 价格敏感度
    - fatigue: 疲劳度
    - cognitive_load: 认知负荷

    返回：
    - 决策概率 [0.01, 0.99]
    """
    from src.simulation.cognitive_agent_theory import (
        prospect_value,
        bounded_rationality_discount,
    )

    # 基础率（来自核）
    theta = kernel.get_theta(step, base_rate)

    # 转换到 logit 空间
    logit_base = np.log(max(theta, 1e-6) / max(1 - theta, 1e-6))

    # 补贴敏感度（来自核）
    # beta 是条件概率差值 (rate_treated - rate_control)，
    # 直接加到 logit 上会导致概率超出范围，需要缩放
    beta = kernel.get_beta(step, 0.0)
    # 缩放：beta 直接作为 logit 空间增量（而非概率空间增量）
    # 用 log(1 + subsidy/10) 做对数缩放，避免大额补贴过度放大
    subsidy_effect = beta * np.log1p(subsidy_amount / 10.0) * price_sensitivity

    # 前景理论调制
    alpha = kernel.get_gamma("alpha", 0.88)
    lambda_ = kernel.get_gamma("lambda_", 2.25)
    if subsidy_amount > 0:
        pv = prospect_value(subsidy_amount, alpha=alpha, lambda_=lambda_)
        prospect_effect = 0.05 * np.sign(pv) * np.log1p(abs(pv))
    else:
        prospect_effect = 0.0

    # 疲劳折扣
    fatigue_rate = kernel.get_gamma("fatigue_rate", 0.15)
    fatigue_discount = np.exp(-fatigue_rate * fatigue)

    # 有限理性折扣
    br = bounded_rationality_discount(cognitive_load)

    # 综合
    logit = (logit_base + subsidy_effect + prospect_effect) * br * fatigue_discount
    prob = 1.0 / (1.0 + np.exp(-logit))

    return float(np.clip(prob, 0.01, 0.99))


# ===========================================================================
# 上下文化核 (Contextual Kernel)
# ===========================================================================

@dataclass
class ContextConfig:
    """
    行为链决策的上下文配置

    描述当前决策所处的外部环境（商家、商品、时间、意图等），
    用于调制用户核参数——同一用户在不同上下文下行为概率不同。

    核心洞察：P(step | user, context) ≠ P(step | user)，
    各次发券不独立同分布，概率是上下文变量的函数。

    上下文维度：
    - merchant_category: 商家品类 (0=到餐, 1=闪购, 2=超市便利, 3=果蔬, 4=其他)
    - price_level: 商品价格水平 (0=低, 1=中, 2=高)
    - time_of_day: 时段 (0=上午, 1=下午, 2=晚间, 3=深夜)
    - session_intent: 会话意图 (0=闲逛, 1=搜索, 2=复购, 3=比价)
    - competition_intensity: 竞争激烈度 [0,1] (竞品优惠力度)
    """
    merchant_category: int = 0
    price_level: int = 1
    time_of_day: int = 0
    session_intent: int = 0
    competition_intensity: float = 0.3

    # 品类名称映射（用于可读性）
    MERCHANT_NAMES = {0: "到餐", 1: "闪购", 2: "超市便利", 3: "果蔬", 4: "其他"}
    PRICE_NAMES = {0: "低", 1: "中", 2: "高"}
    TIME_NAMES = {0: "上午", 1: "下午", 2: "晚间", 3: "深夜"}
    INTENT_NAMES = {0: "闲逛", 1: "搜索", 2: "复购", 3: "比价"}

    def to_vector(self) -> np.ndarray:
        """
        将上下文编码为特征向量 z_c

        编码方式：
        - merchant_category: one-hot (5维)
        - price_level: one-hot (3维)
        - time_of_day: one-hot (4维)
        - session_intent: one-hot (4维)
        - competition_intensity: 标准化 [0,1] (1维)
        总计 17维

        返回:
            np.ndarray of shape (17,)
        """
        z = np.zeros(17)
        # merchant_category one-hot (0-4 → idx 0-4)
        if 0 <= self.merchant_category <= 4:
            z[self.merchant_category] = 1.0
        # price_level one-hot (0-2 → idx 5-7)
        if 0 <= self.price_level <= 2:
            z[5 + self.price_level] = 1.0
        # time_of_day one-hot (0-3 → idx 8-11)
        if 0 <= self.time_of_day <= 3:
            z[8 + self.time_of_day] = 1.0
        # session_intent one-hot (0-3 → idx 12-15)
        if 0 <= self.session_intent <= 3:
            z[12 + self.session_intent] = 1.0
        # competition_intensity continuous (idx 16)
        z[16] = np.clip(self.competition_intensity, 0, 1)
        return z

    def describe(self) -> str:
        """人类可读的上下文描述"""
        return (
            f"{self.MERCHANT_NAMES.get(self.merchant_category, '?')}/"
            f"{self.PRICE_NAMES.get(self.price_level, '?')}价/"
            f"{self.TIME_NAMES.get(self.time_of_day, '?')}/"
            f"{self.INTENT_NAMES.get(self.session_intent, '?')}"
        )

    @classmethod
    def dim(cls) -> int:
        """上下文向量维度"""
        return 17


@dataclass
class ContextualKernel(UserKernel):
    """
    上下文化行为核——概率是上下文变量的函数

    继承 UserKernel，增加上下文调制权重矩阵：
    - W_theta: dict[step, ndarray(17,)]  上下文对基础转换率的调制
    - W_beta: dict[step, ndarray(17,)]   上下文对补贴敏感度的调制

    数学形式：
      P(step | user_i, context_c, subsidy_d) = sigma(
          logit(theta_base[step] + W_theta[step] · z_c)
          + (beta_base[step] + W_beta[step] · z_c) * g(d)
          + gamma_modifiers
      )

    其中：
    - z_c = ContextConfig.to_vector()，17维上下文特征
    - g(d) = log(1 + d/scale)，剂量-响应函数（边际递减）
    - W_theta · z_c：上下文对基础率的调制（如到餐品类click率更高）
    - W_beta · z_c：上下文对补贴效应的调制（如搜索意图下券更有效）

    关键洞察：同一用户在不同上下文下，行为概率不同；
    同一张券在不同场景下，效应也不同（context-treatment interaction）。
    """
    W_theta: dict[str, np.ndarray] = field(default_factory=dict)
    W_beta: dict[str, np.ndarray] = field(default_factory=dict)

    def contextual_theta(self, step: str, context: ContextConfig) -> float:
        """
        计算上下文化的基础转换率

        theta(step, context) = theta_base[step] + W_theta[step] · z_c
        """
        base = self.get_theta(step, 0.5)
        z_c = context.to_vector()
        W = self.W_theta.get(step)
        if W is not None and len(W) == len(z_c):
            modulation = float(W @ z_c)
        else:
            modulation = 0.0
        return float(np.clip(base + modulation, 0.01, 0.99))

    def contextual_beta(self, step: str, context: ContextConfig) -> float:
        """
        计算上下文化的补贴敏感度

        beta(step, context) = beta_base[step] + W_beta[step] · z_c
        """
        base = self.get_beta(step, 0.0)
        z_c = context.to_vector()
        W = self.W_beta.get(step)
        if W is not None and len(W) == len(z_c):
            modulation = float(W @ z_c)
        else:
            modulation = 0.0
        return float(base + modulation)

    def to_vector(self) -> np.ndarray:
        """将核参数+上下文权重展开为向量"""
        # 先调用父类的 to_vector
        base_vec = super().to_vector()
        # 展开上下文权重
        context_parts = []
        for step in self.steps:
            W = self.W_theta.get(step, np.zeros(ContextConfig.dim()))
            context_parts.append(W)
        for step in self.steps:
            W = self.W_beta.get(step, np.zeros(ContextConfig.dim()))
            context_parts.append(W)
        if context_parts:
            context_vec = np.concatenate(context_parts)
            return np.concatenate([base_vec, context_vec])
        return base_vec

    @classmethod
    def from_kernel_and_weights(
        cls,
        base_kernel: UserKernel,
        W_theta: Optional[dict[str, np.ndarray]] = None,
        W_beta: Optional[dict[str, np.ndarray]] = None,
    ) -> "ContextualKernel":
        """从基础核 + 上下文权重构建 ContextualKernel"""
        return cls(
            user_id=base_kernel.user_id,
            theta=base_kernel.theta,
            beta=base_kernel.beta,
            gamma=base_kernel.gamma,
            metadata=base_kernel.metadata,
            W_theta=W_theta or {},
            W_beta=W_beta or {},
        )


class ContextualKernelExtractor:
    """
    上下文化核提取器

    从带上下文标注的行为链数据中，同时估计：
    1. theta_base + W_theta (基础率 + 上下文调制)
    2. beta_base + W_beta (补贴敏感度 + 上下文交互)
    3. gamma (认知调制参数)

    估计方法：
    - theta_base: 层级贝叶斯收缩（同 BehavioralKernelExtractor）
    - W_theta: 对每个步骤，用 logistic 回归拟合 context → step_reached
    - beta_base + W_beta: 用差分法或交互项回归

    关键：当数据不足时，W 退化为零向量（等效于无上下文调制），
    保证了小样本下的稳健性。
    """

    def __init__(
        self,
        base_extractor: Optional[BehavioralKernelExtractor] = None,
        regularization: float = 1.0,
        min_context_samples: int = 20,
        random_state: int = 42,
    ):
        """
        参数：
        - base_extractor: 基础核提取器（如为None则自动创建）
        - regularization: 上下文权重的 L2 正则化强度（越大→W越小→越接近无上下文）
        - min_context_samples: 拟合上下文权重的最小样本量
        - random_state: 随机种子
        """
        self.base_extractor = base_extractor or BehavioralKernelExtractor(
            random_state=random_state
        )
        self.regularization = regularization
        self.min_context_samples = min_context_samples
        self.rng = np.random.RandomState(random_state)

    def fit_extract(
        self,
        behavior_chains: pd.DataFrame,
        user_col: str = "user_id",
        step_col: str = "action",
        time_col: str = "timestamp",
        subsidy_col: Optional[str] = "subsidy_amount",
        context_cols: Optional[list[str]] = None,
    ) -> list[ContextualKernel]:
        """
        从带上下文的行为链数据中提取上下文化核

        参数：
        - behavior_chains: 行为链 DataFrame
        - user_col: 用户ID列名
        - step_col: 行为步骤列名
        - time_col: 时间戳列名
        - subsidy_col: 补贴金额列名
        - context_cols: 上下文特征列名
            如果提供了，需包含: merchant_category, price_level,
            time_of_day, session_intent, competition_intensity

        返回：
        - List[ContextualKernel]
        """
        # Step 1: 用基础提取器获取 theta_base, beta_base, gamma
        base_kernels = self.base_extractor.fit_extract(
            behavior_chains, user_col, step_col, time_col, subsidy_col
        )

        # Step 2: 如果没有上下文列，退化为普通核
        if context_cols is None or len(behavior_chains) < self.min_context_samples:
            return [
                ContextualKernel.from_kernel_and_weights(k)
                for k in base_kernels
            ]

        # Step 3: 对每个用户拟合上下文权重
        context_kernels = []
        for kernel in base_kernels:
            W_theta, W_beta = self._fit_context_weights(
                behavior_chains, kernel.user_id, user_col,
                step_col, subsidy_col, context_cols
            )
            ck = ContextualKernel.from_kernel_and_weights(kernel, W_theta, W_beta)
            context_kernels.append(ck)

        return context_kernels

    def _fit_context_weights(
        self,
        df: pd.DataFrame,
        user_id: str,
        user_col: str,
        step_col: str,
        subsidy_col: Optional[str],
        context_cols: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        对单个用户拟合上下文调制权重

        方法：对每个漏斗步骤 s，
        用 L2 正则化 logistic 回归拟合:
          P(step_s reached | z_c) = sigma(W_s · z_c + b_s)

        正则化确保小样本下 W_s 不会过大。
        """
        user_df = df[df[user_col] == user_id]

        if len(user_df) < self.min_context_samples:
            return {}, {}

        # 构建上下文特征矩阵
        Z = self._build_context_matrix(user_df, context_cols)

        W_theta = {}
        W_beta = {}

        for step in ChainStep.funnel_steps():
            step_name = step.value

            # 目标：该步骤是否达成
            y = (user_df[step_col].str.strip().str.lower() == step_name).astype(float)

            # 仅当正负样本都有时才拟合
            if y.sum() < 2 or (1 - y).sum() < 2:
                W_theta[step_name] = np.zeros(ContextConfig.dim())
                W_beta[step_name] = np.zeros(ContextConfig.dim())
                continue

            # L2 正则化 logistic 回归（Ridge-style）
            W, intercept = self._ridge_logistic(Z, y)

            # W_theta: 上下文对基础率的调制
            # 注意：这里 W 是完整的权重（包括截距），上下文调制部分是 W 本身
            # 实际使用时：theta = theta_base + W · z_c
            # 所以 W_theta 是在 theta_base 基础上的增量
            W_theta[step_name] = W * self.regularization / (self.regularization + len(user_df) * 0.01)

            # W_beta: 上下文对补贴效应的调制
            # 需要有补贴数据才能估计
            if subsidy_col and subsidy_col in user_df.columns:
                # 交互项：context × subsidy
                subsidy_vals = user_df[subsidy_col].values
                Z_interaction = Z * subsidy_vals[:, np.newaxis]
                # 目标：处理组中该步骤是否达成
                treated = user_df[user_df[subsidy_col] > 0]
                if len(treated) >= 5:
                    y_treated = (treated[step_col].str.strip().str.lower() == step_name).astype(float)
                    Z_treated = self._build_context_matrix(treated, context_cols)
                    if y_treated.sum() >= 2 and (1 - y_treated).sum() >= 2:
                        W_int, _ = self._ridge_logistic(Z_interaction, y)
                        W_beta[step_name] = W_int * self.regularization / (self.regularization + len(user_df) * 0.01)
                    else:
                        W_beta[step_name] = np.zeros(ContextConfig.dim())
                else:
                    W_beta[step_name] = np.zeros(ContextConfig.dim())
            else:
                W_beta[step_name] = np.zeros(ContextConfig.dim())

        return W_theta, W_beta

    def _build_context_matrix(
        self,
        df: pd.DataFrame,
        context_cols: list[str],
    ) -> np.ndarray:
        """从 DataFrame 的上下文列构建特征矩阵"""
        # 尝试映射到 ContextConfig 的 to_vector 格式
        rows = []
        for _, row in df.iterrows():
            try:
                ctx = ContextConfig(
                    merchant_category=int(row.get("merchant_category", 0)),
                    price_level=int(row.get("price_level", 1)),
                    time_of_day=int(row.get("time_of_day", 0)),
                    session_intent=int(row.get("session_intent", 0)),
                    competition_intensity=float(row.get("competition_intensity", 0.3)),
                )
                rows.append(ctx.to_vector())
            except (ValueError, TypeError):
                rows.append(np.zeros(ContextConfig.dim()))
        return np.array(rows) if rows else np.zeros((1, ContextConfig.dim()))

    def _ridge_logistic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> tuple[np.ndarray, float]:
        """
        L2 正则化 logistic 回归（梯度下降）

        返回 (W, intercept) 其中 W 是特征权重
        """
        n, d = X.shape
        W = np.zeros(d)
        b = 0.0

        for _ in range(max_iter):
            # 前向传播
            logit = X @ W + b
            pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))

            # 梯度
            error = pred - y
            grad_W = (X.T @ error) / n + self.regularization * W
            grad_b = np.mean(error)

            # 更新
            W -= lr * grad_W
            b -= lr * grad_b

        return W, b


# ===========================================================================
# 上下文化核驱动决策概率
# ===========================================================================

def contextual_kernel_modulated_prob(
    base_rate: float,
    kernel: UserKernel,
    step: str,
    subsidy_amount: float = 0.0,
    price_sensitivity: float = 0.5,
    fatigue: float = 0.0,
    cognitive_load: float = 3.0,
    context: Optional[ContextConfig] = None,
    dose_scale: float = 10.0,
) -> float:
    """
    上下文化核驱动的决策概率

    数学公式（在 kernel_modulated_prob 基础上扩展）：

      P(step | user_i, context_c, subsidy_d) = sigma(
          logit(theta(step, context))        <-- 上下文化基础率
          + beta(step, context) * g(d)        <-- 上下文化补贴效应 × 剂量-响应
          + gamma_modifiers                   <-- 认知调制
      )

    其中：
    - theta(step, context) = theta_base[step] + W_theta[step] · z_c
    - beta(step, context) = beta_base[step] + W_beta[step] · z_c
    - g(d) = log(1 + d/dose_scale)，边际递减的剂量-响应函数
    - z_c = ContextConfig.to_vector()

    核心创新：
    1. Context modulation: 同一用户在不同商家/品类/时段下基础率不同
    2. Dose-response: 补贴不是"发/不发"二值，而是连续剂量，有边际递减
    3. Context-treatment interaction: 同一张券在不同场景下效应不同

    参数：
    - base_rate: 备用基础率（当 kernel 无该步骤时使用）
    - kernel: 用户核参数（UserKernel 或 ContextualKernel）
    - step: 当前链步骤
    - subsidy_amount: 补贴金额（连续处理变量）
    - price_sensitivity: 价格敏感度
    - fatigue: 疲劳度
    - cognitive_load: 认知负荷
    - context: 上下文配置（如为 None，退化为 kernel_modulated_prob）
    - dose_scale: 剂量-响应缩放因子

    返回：
    - 决策概率 [0.01, 0.99]
    """
    from src.simulation.cognitive_agent_theory import (
        prospect_value,
        bounded_rationality_discount,
    )

    # 确定基础率：如果有上下文且核支持上下文，使用上下文化基础率
    if context is not None and isinstance(kernel, ContextualKernel) and kernel.W_theta:
        theta = kernel.contextual_theta(step, context)
    else:
        theta = kernel.get_theta(step, base_rate)

    # 转换到 logit 空间
    logit_base = np.log(max(theta, 1e-6) / max(1 - theta, 1e-6))

    # 确定补贴敏感度：如果有上下文且核支持，使用上下文化敏感度
    if context is not None and isinstance(kernel, ContextualKernel) and kernel.W_beta:
        beta = kernel.contextual_beta(step, context)
    else:
        beta = kernel.get_beta(step, 0.0)

    # 剂量-响应函数：g(d) = log(1 + d/dose_scale)
    # 关键特性：
    # - g(0) = 0（无补贴无效应）
    # - g(d) 单调递增（多补贴多效应）
    # - g''(d) < 0（边际递减：第一张5元券的效果 > 第三张5元券）
    # - 剂量-响应曲线的形状由 dose_scale 控制：
    #   dose_scale 小 → 快速饱和，dose_scale 大 → 缓慢饱和
    dose_response = np.log1p(subsidy_amount / dose_scale)

    # 补贴效应 = beta × g(d) × price_sensitivity
    subsidy_effect = beta * dose_response * price_sensitivity

    # 前景理论调制
    alpha = kernel.get_gamma("alpha", 0.88)
    lambda_ = kernel.get_gamma("lambda_", 2.25)
    if subsidy_amount > 0:
        pv = prospect_value(subsidy_amount, alpha=alpha, lambda_=lambda_)
        prospect_effect = 0.05 * np.sign(pv) * np.log1p(abs(pv))
    else:
        prospect_effect = 0.0

    # 疲劳折扣
    fatigue_rate = kernel.get_gamma("fatigue_rate", 0.15)
    fatigue_discount = np.exp(-fatigue_rate * fatigue)

    # 有限理性折扣
    br = bounded_rationality_discount(cognitive_load)

    # 综合
    logit = (logit_base + subsidy_effect + prospect_effect) * br * fatigue_discount
    prob = 1.0 / (1.0 + np.exp(-logit))

    return float(np.clip(prob, 0.01, 0.99))


def optimal_dosage(
    kernel: UserKernel,
    step: str = "redeemed",
    budget: float = 20.0,
    price_sensitivity: float = 0.5,
    context: Optional[ContextConfig] = None,
    dose_scale: float = 10.0,
    n_grid: int = 100,
) -> tuple[float, float]:
    """
    最优补贴金额估计（剂量-响应优化）

    目标：maximize ROI(subsidy_d) = (GTV(d) - d) / d

    方法：在 [0, budget] 上网格搜索，找到 ROI 最高的剂量点。
    对于有上下文的核，在给定上下文下优化。

    参数：
    - kernel: 用户核参数
    - step: 优化的目标步骤（默认 redeemed，即核销）
    - budget: 最大补贴预算
    - price_sensitivity: 价格敏感度
    - context: 上下文配置
    - dose_scale: 剂量-响应缩放因子
    - n_grid: 搜索网格点数

    返回：
    - (optimal_dosage, expected_roi) 最优剂量和期望ROI
    """
    best_dose = 0.0
    best_roi = -float("inf")

    for d in np.linspace(0.1, budget, n_grid):
        # 在剂量 d 下的转换概率
        p = contextual_kernel_modulated_prob(
            base_rate=0.5,
            kernel=kernel,
            step=step,
            subsidy_amount=d,
            price_sensitivity=price_sensitivity,
            fatigue=0.0,
            cognitive_load=3.0,
            context=context,
            dose_scale=dose_scale,
        )

        # 期望 GTV 增量（简化模型）
        base_p = contextual_kernel_modulated_prob(
            base_rate=0.5,
            kernel=kernel,
            step=step,
            subsidy_amount=0.0,
            price_sensitivity=price_sensitivity,
            fatigue=0.0,
            cognitive_load=3.0,
            context=context,
            dose_scale=dose_scale,
        )

        delta_p = p - base_p
        # 期望增量 GTV（简化：假设 base_gtv ≈ 50，核销贡献为 delta_p * 50）
        expected_delta_gtv = delta_p * 50.0
        # ROI = (delta_gtv - cost) / cost
        if d > 0:
            roi = (expected_delta_gtv - d) / d
        else:
            roi = 0.0

        if roi > best_roi:
            best_roi = roi
            best_dose = d

    return best_dose, best_roi
