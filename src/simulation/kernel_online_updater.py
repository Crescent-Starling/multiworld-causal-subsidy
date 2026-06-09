"""
核在线更新模块 (Kernel Online Updater)
==============================================

多平行世界框架的核心价值：
  传统A/B实验是静态的——用户参数在实验期间被视为不变量。
  但现实中，补贴策略本身就是一种干预，会改变用户的行为参数：
    - 过度补贴 → 疲劳度累积 → theta_{paid} 下降
    - 持续补贴 → 参考点适应 → 补贴敏感度 beta 漂移
    - 长期暴露 → 心理账户迁移 → gamma 参数改变

KernelOnlineUpdater 实现：
  每一轮仿真结束后，根据Agent的行为序列，用贝叶斯更新规则
  修正其UserKernel参数，使下一轮仿真使用"被策略塑造后的核"。

设计动机：
----------
传统A/B实验的隐含假设是"用户参数是固定的"，即：
    Y_i(a) = f(X_i, θ_i)   其中 θ_i 不随实验进行而改变

但补贴策略的实验往往违反这个假设：
  - 疲劳效应：用户 i 在轮次 t 接受补贴后，轮次 t+1 的
    P(redeemed) 系统性下降
  - 参考点适应：持续补贴使参考点 R_i 上移，同等补贴的
    主观价值 V_i(t) 递减
  - 心理账户迁移：Windfall Spender → Routine Income，
    补贴的"意外之财"感知消失

这意味着：
  同一个用户，在不同实验阶段，其"真实θ_i"是不同的。
  传统A/B实验把这种动态变化归为"随机噪声"，
  导致效应估计偏误（attenuation bias）。

多平行世界解决方案：
----------------
  世界A（策略A）中的用户 i，其核参数 θ_i^A(t) 会随策略A
  的作用而演化；
  世界B（策略B）中的同一用户 i，其核参数 θ_i^B(t) 随
  策略B的作用而演化。

  这样，多平行世界不仅测量"策略对行为的即时效应"，
  还能观测"策略对用户参数的长期塑造效应"——
  这是传统A/B实验完全无法做到的。

参考文献：
- Chickering & Pearl (1996): "A Clinician's Tool for Analyzing
  Non-Compliance in Clinical Trials"
- Imai, K. et al. (2013): "Estimating the Effect of Treatments
  on the Treated: Challenges and Solutions"
- Hofmann, W. et al. (2014): "Habit Formation and Behavior
  Change" (心理账户迁移的行为学证据)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any


# ===========================================================================
# 在线核更新器
# ===========================================================================

class KernelOnlineUpdater:
    """
    在线核参数更新器

    每一轮仿真结束后，根据Agent的行为序列，
    用贝叶斯更新规则修正UserKernel参数。

    更新规则：
    ----------
    (1) theta 更新（基础转化率）：
        theta_{i,s}(t+1) = (n_i(t) * hat_theta_{i,s}(t) + tau * mu_s) / (n_i(t) + tau)
        其中 hat_theta_{i,s}(t) = 本轮观测到的条件转化率

    (2) beta 更新（补贴敏感度）：
        beta_{i,s}(t+1) = beta_{i,s}(t) + eta_beta * (observed_uplift - beta_{i,s}(t))
        其中 observed_uplift = P(step|subsidized) - P(step|not_subsidized)

    (3) gamma 更新（认知参数）：
        fatigue_rate: 从两轮间fatigue的差分估计
        account_eta: 从参考点移动速度估计

    核心洞察：
    -----------
    传统A/B实验假设用户参数是时不变的（time-invariant），
    但实际上补贴策略会产生"动态处理效应"（dynamic treatment effect）：

        用户 i 在轮次 1-10 接受策略 A → 疲劳累积 →
        轮次 11-20 时，即使策略不变，响应率也下降了

    多平行世界框架能够捕捉这种动态：
      世界A：策略A运行30轮 → 观测到theta_i的演化轨迹
      世界B：策略B运行30轮 → 观测到不同的theta_i演化轨迹
      世界C（对照组）：无补贴 → theta_i基本稳定

    这种"参数演化轨迹的对比"，
    是传统A/B实验（只给一个静态uplift数字）无法提供的。
    """

    def __init__(
        self,
        learning_rate_theta: float = 0.1,
        learning_rate_beta: float = 0.05,
        forgetting_factor: float = 0.95,
        min_observations: int = 3,
        enable_dynamic_beta: bool = True,
        enable_dynamic_gamma: bool = True,
    ):
        """
        初始化在线更新器

        参数：
        - learning_rate_theta: theta更新的学习率（越小越保守）
        - learning_rate_beta:  beta更新的学习率
        - forgetting_factor:   遗忘因子（旧数据的权重衰减）
        - min_observations:    触发更新所需的最少观测数
        - enable_dynamic_beta:  是否启用beta动态更新
        - enable_dynamic_gamma: 是否启用gamma动态更新
        """
        self.lr_theta = learning_rate_theta
        self.lr_beta = learning_rate_beta
        self.forgetting = forgetting_factor
        self.min_obs = min_observations
        self.enable_beta = enable_dynamic_beta
        self.enable_gamma = enable_dynamic_gamma

        # 每轮行为记录：{agent_id: [{"step": "clicked", "subsidized": bool, ...}]}
        self._behavior_history: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    # 主接口：更新所有Agent的核
    # ------------------------------------------------------------------

    def update_all_kernels(
        self,
        agents: list[Any],  # List[SubsidyAgent]
        world_name: str,
        round_num: int,
    ) -> dict[int, Any]:  # -> dict[int, UserKernel]
        """
        对所有Agent执行在线核更新

        参数：
        - agents: SubsidyAgent列表
        - world_name: 当前世界名称（用于记录）
        - round_num: 当前轮次

        返回：
        - {agent_id: updated_UserKernel}
          如果Agent没有核，返回None（跳过）
        """
        updated_kernels = {}

        for agent in agents:
            if agent.user_kernel is None:
                # 该Agent没有核参数（使用硬编码base_chain_rates）
                # 可以选择从当前行为初始化一个核
                kernel = self._init_kernel_from_behavior(agent)
                if kernel is None:
                    updated_kernels[agent.agent_id] = None
                    continue
                agent.user_kernel = kernel

            # 收集本轮行为数据
            behavior_data = self._collect_agent_behavior(agent, round_num)
            if behavior_data is None:
                updated_kernels[agent.agent_id] = agent.user_kernel
                continue

            # 执行更新
            new_kernel = self._update_single_kernel(
                agent.user_kernel, behavior_data, agent
            )
            agent.user_kernel = new_kernel
            updated_kernels[agent.agent_id] = new_kernel

        return updated_kernels

    # ------------------------------------------------------------------
    # 行为数据收集
    # ------------------------------------------------------------------

    def _collect_agent_behavior(
        self, agent: Any, round_num: int
    ) -> Optional[dict]:
        """
        从Agent当前状态收集行为数据

        返回：
        - {
            "round": int,
            "subsidized": bool,
            "subsidy_amount": float,
            "funnel_state": dict,
            "fatigue": float,
            "reference_point": float,
            "mental_account": str,
          }
        """
        return {
            "round": round_num,
            "subsidized": agent._step_subsidized,
            "subsidy_amount": agent._step_subsidy_amount,
            "funnel_state": agent.funnel_state.copy(),
            "fatigue": agent.fatigue,
            "reference_point": agent.reference_point,
            "mental_account": agent.mental_account.value,
        }

    def _init_kernel_from_behavior(self, agent: Any) -> Optional[Any]:
        """
        当Agent没有核时，从其行为链初始化一个

        这允许"无核Agent"在仿真过程中逐渐获得个性化核，
        模拟真实场景中"新用户"逐渐积累行为数据的过程。
        """
        # 从Agent的base_chain_rates或默认值初始化
        from .behavioral_kernel import UserKernel, ChainStep

        theta = {}
        beta = {}
        for step in ChainStep.funnel_steps():
            # 用Agent的base_chain_rates或默认值
            rate = agent.model.base_chain_rates.get(step.value, 0.5)
            theta[step.value] = rate
            beta[step.value] = 0.0  # 初始无补贴敏感度估计

        gamma = {
            "alpha": agent.alpha,
            "lambda_": agent.lambda_,
            "fatigue_rate": 0.1,
            "account_eta": 0.2,
        }

        kernel = UserKernel(
            user_id=str(agent.agent_id),
            theta=theta,
            beta=beta,
            gamma=gamma,
        )
        return kernel

    # ------------------------------------------------------------------
    # 单Agent核更新（核心算法）
    # ------------------------------------------------------------------

    def _update_single_kernel(
        self,
        old_kernel: Any,  # UserKernel
        behavior_data: dict,
        agent: Any,
    ) -> Any:  # -> UserKernel
        """
        对单个Agent执行贝叶斯在线核更新

        更新逻辑：
        ----------
        (1) theta更新：
            observed_rate = 本轮观测到的条件转化率
            new_theta = (1 - lr) * old_theta + lr * observed_rate
            这等价于指数加权移动平均（EWMA），
            是贝叶斯更新在共轭先验下的在线近似。

        (2) beta更新：
            observed_uplift = P(step|subsidized) - P(step|not_subsidized)
            new_beta = (1 - lr) * old_beta + lr * observed_uplift
            捕捉用户对补贴的敏感度变化。

        (3) gamma更新：
            fatigue_rate：从 fatigue(t) - fatigue(t-1) 估计
            account_eta：从参考点移动幅度估计

        为什么这是因果的？
        ---------------
        传统相关分析：看补贴金额与转化率的相关系数
        → 混淆偏差：本来就想买的用户，平台更愿意发券

        本方法：用多平行世界的设计
        - 世界A（高补贴策略）：观测 beta_i^A 的演化
        - 世界B（低补贴策略）：观测 beta_i^B 的演化
        - 世界C（无补贴对照）：观测 beta_i^C 的演化

        如果 beta_i^A(t) 随 t 递减（补贴效应衰减），
        且 beta_i^C(t) 保持稳定，
        这提供了"补贴导致敏感度下降"的因果证据——
        因为这是跨世界对比，控制了用户固有必要特征。
        """
        from .behavioral_kernel import UserKernel, ChainStep

        new_theta = dict(old_kernel.theta)
        new_beta = dict(old_kernel.beta)
        new_gamma = dict(old_kernel.gamma)

        funnel = behavior_data.get("funnel_state", {})
        subsidized = behavior_data.get("subsidized", False)
        subsidy_amt = behavior_data.get("subsidy_amount", 0.0)

        # ---- (1) theta更新 ----
        # 核心修正：区分"有补贴"和"无补贴"观测
        # - 无补贴时：observed直接反映基础转化率，正常更新theta
        # - 有补贴时：observed = theta + beta*effect，需要剥离补贴贡献
        #   使用"减去预测的beta贡献"来估计"纯基础theta"
        for step in ChainStep.funnel_steps():
            step_name = step.value
            if step_name in funnel:
                observed = 1.0 if funnel[step_name] else 0.0
                old_t = new_theta.get(step_name, 0.5)

                if subsidized and observed > 0:
                    # 有补贴且转化了：
                    # 观测到的概率 = theta + beta * subsidy_effect
                    # 我们需要估计"如果没有补贴，概率是多少"
                    old_b = new_beta.get(step_name, 0.0)
                    subsidy_effect = np.log1p(subsidy_amt / 10.0) / 5.0  # 与kernel_modulated_prob一致
                    predicted_subsidy_boost = old_b * subsidy_effect
                    # "剥离补贴贡献"后的基础概率估计
                    deconfounded = max(observed - predicted_subsidy_boost, 0.01)
                    # 使用较小的学习率（因为有补贴的观测不确定性更高）
                    effective_lr = self.lr_theta * 0.5
                    new_t = (1 - effective_lr) * old_t + effective_lr * deconfounded
                else:
                    # 无补贴时，观测直接反映基础theta
                    new_t = (1 - self.lr_theta) * old_t + self.lr_theta * observed

                new_theta[step_name] = float(np.clip(new_t, 0.01, 0.99))

        # ---- (2) beta更新 ----
        if self.enable_beta:
            for step in ChainStep.funnel_steps():
                step_name = step.value
                if step_name not in funnel:
                    continue

                observed = 1.0 if funnel[step_name] else 0.0
                old_b = new_beta.get(step_name, 0.0)
                old_t = new_theta.get(step_name, 0.5)

                if subsidized:
                    # 有补贴时：实际转化率 vs 预期基础转化率
                    # beta ≈ observed_rate - baseline_rate
                    implied_uplift = observed - old_t
                    # 补贴金额归一化
                    subsidy_norm = np.log1p(subsidy_amt / 10.0) / 5.0
                    if subsidy_norm > 0:
                        # 去归一化：implied_beta = implied_uplift / subsidy_norm
                        implied_beta = implied_uplift / subsidy_norm
                    else:
                        implied_beta = 0.0
                    new_b = (1 - self.lr_beta) * old_b + self.lr_beta * implied_beta
                else:
                    # 无补贴时：实际转化率提供baseline信息
                    # 如果实际 < baseline，可能暗示beta被高估
                    # 轻微向0收缩（因为无补贴观测不提供beta的直接证据）
                    new_b = old_b * 0.98  # 轻微衰减

                new_beta[step_name] = float(np.clip(new_b, -0.5, 0.5))

        # ---- (3) gamma更新 ----
        if self.enable_gamma:
            # fatigue_rate：从当前fatigue水平估计
            current_fatigue = behavior_data.get("fatigue", 0.0)
            old_fatigue_rate = new_gamma.get("fatigue_rate", 0.1)
            # 高fatigue → 高fatigue_rate（快速累积）
            current_f = behavior_data.get("fatigue", 0.0)
            rnd = max(behavior_data.get("round", 1), 1.0)
            implied_rate = min(current_f / rnd, 1.0)
            new_fatigue_rate = (1 - 0.1) * old_fatigue_rate + 0.1 * implied_rate
            new_gamma["fatigue_rate"] = float(np.clip(new_fatigue_rate, 0.01, 0.5))

            # account_eta：从心理账户类型推断
            acct = behavior_data.get("mental_account", "PRICE_SENSITIVE")
            if acct == "ROUTINE_INCOME":
                # 已迁移到理性账户 → 参考点更新慢
                new_gamma["account_eta"] = 0.05
            elif acct == "WINDFALL_SPENDER":
                # 仍是冲动型 → 参考点更新快
                new_gamma["account_eta"] = 0.3

        return UserKernel(
            user_id=old_kernel.user_id,
            theta=new_theta,
            beta=new_beta,
            gamma=new_gamma,
            metadata={
                **old_kernel.metadata,
                "last_updated_round": behavior_data.get("round", 0),
                "update_method": "online_bayes_ewma",
            },
        )


# ===========================================================================
# 多世界核演化追踪器
# ===========================================================================

@dataclass
class KernelTrajectory:
    """
    记录单个Agent在多个平行世界中的核参数演化轨迹

    用途：
    - 对比不同策略下，同一用户的theta/beta/gamma如何不同地演化
    - 提供"策略→用户参数塑造"的因果证据
    - 这是传统A/B实验完全无法提供的分析维度
    """
    agent_id: int
    world_trajectories: dict[str, list[dict]] = field(default_factory=dict)
    # world_trajectories[world_name] = [
    #   {"round": 1, "theta_clicked": 0.6, "beta_clicked": 0.1, "fatigue": 0.2, ...},
    #   {"round": 2, ...},
    # ]

    def add_snapshot(
        self,
        world_name: str,
        round_num: int,
        kernel: Any,  # UserKernel
        agent_state: dict,
    ) -> None:
        """记录一轮的核参数快照"""
        if world_name not in self.world_trajectories:
            self.world_trajectories[world_name] = []

        snapshot = {"round": round_num}
        for step_name, val in kernel.theta.items():
            snapshot[f"theta_{step_name}"] = val
        for step_name, val in kernel.beta.items():
            snapshot[f"beta_{step_name}"] = val
        snapshot["fatigue"] = agent_state.get("fatigue", 0.0)
        snapshot["reference_point"] = agent_state.get("reference_point", 0.0)
        snapshot["mental_account"] = agent_state.get("mental_account", "")

        self.world_trajectories[world_name].append(snapshot)

    def get_trajectory_df(self, world_name: str) -> pd.DataFrame:
        """将某个世界的轨迹转换为DataFrame"""
        if world_name not in self.world_trajectories:
            return pd.DataFrame()
        return pd.DataFrame(self.world_trajectories[world_name])

    def compare_worlds(self, step: str = "clicked") -> pd.DataFrame:
        """
        对比同一Agent在不同世界中的theta演化

        返回：
        - DataFrame: round | world | theta_clicked | fatigue | ...
        """
        rows = []
        for world_name, traj in self.world_trajectories.items():
            for snap in traj:
                rows.append({
                    "world": world_name,
                    "round": snap["round"],
                    f"theta_{step}": snap.get(f"theta_{step}", 0.5),
                    "fatigue": snap.get("fatigue", 0.0),
                    "mental_account": snap.get("mental_account", ""),
                })
        return pd.DataFrame(rows)


class MultiWorldKernelTracker:
    """
    多世界核演化追踪器（全局管理器）

    在每个世界的每轮结束后，
    记录所有Agent的核参数快照，
    用于后续分析"策略如何塑造用户参数"。

    核心价值：
    --------
    传统A/B实验只回答：
      "策略A比策略B好多少？"（一个静态数字）

    多平行世界+在线核更新可以回答：
      (1) "策略A和B的效应，在不同用户生命周期阶段是否不同？"
          → 看theta_i(t)轨迹的分叉点
      (2) "补贴策略是否导致用户'耐药性的产生'？"
          → 看beta_i(t)是否随t递减
      (3) "不同策略对用户心理账户迁移的不同影响？"
          → 看gamma_i(t)在不同世界中的演化差异
      (4) "哪种策略最'温和'，不会导致用户疲劳累积？"
          → 对比fatigue(t)轨迹

    这些都是传统A/B实验无法回答的因果动力学问题。
    """

    def __init__(self):
        self.tracker: dict[int, KernelTrajectory] = {}
        self.world_kernels: dict[str, dict[int, list[dict]]] = {}
        # world_kernels[world_name][agent_id] = [snapshot1, snapshot2, ...]

    def register_snapshot(
        self,
        world_name: str,
        round_num: int,
        agents: list[Any],
    ) -> None:
        """为某个世界的所有Agent记录核参数快照"""
        if world_name not in self.world_kernels:
            self.world_kernels[world_name] = {}

        for agent in agents:
            aid = agent.agent_id
            if aid not in self.world_kernels[world_name]:
                self.world_kernels[world_name][aid] = []

            snapshot = {"round": round_num}
            if agent.user_kernel is not None:
                for step_name, val in agent.user_kernel.theta.items():
                    snapshot[f"theta_{step_name}"] = val
                for step_name, val in agent.user_kernel.beta.items():
                    snapshot[f"beta_{step_name}"] = val
                snapshot["gamma"] = dict(agent.user_kernel.gamma)

            snapshot["fatigue"] = agent.fatigue
            snapshot["reference_point"] = agent.reference_point
            snapshot["mental_account"] = agent.mental_account.value
            snapshot["subsidized"] = agent._step_subsidized
            snapshot["subsidy_amount"] = agent._step_subsidy_amount

            self.world_kernels[world_name][aid].append(snapshot)

            # 同时更新每个Agent的跨世界轨迹
            if aid not in self.tracker:
                self.tracker[aid] = KernelTrajectory(agent_id=aid)
            self.tracker[aid].add_snapshot(
                world_name, round_num, agent.user_kernel,
                {"fatigue": agent.fatigue,
                 "reference_point": agent.reference_point,
                 "mental_account": agent.mental_account.value}
            )

    def analyze_beta_drift(self, agent_id: int) -> pd.DataFrame:
        """
        分析某个Agent的beta参数在不同世界中的漂移

        用途：
        - 检验"补贴导致敏感度下降"假设
        - 对比不同策略下的beta漂移速度
        """
        if agent_id not in self.tracker:
            return pd.DataFrame()

        traj = self.tracker[agent_id]
        return traj.compare_worlds(step="clicked")

    def analyze_fatigue_accumulation(self) -> pd.DataFrame:
        """
        分析所有Agent在所有世界中的疲劳累积模式

        返回：
        - DataFrame: world | round | avg_fatigue | ...
        """
        rows = []
        for world_name, agent_data in self.world_kernels.items():
            for aid, snapshots in agent_data.items():
                for snap in snapshots:
                    rows.append({
                        "world": world_name,
                        "agent_id": aid,
                        "round": snap["round"],
                        "fatigue": snap["fatigue"],
                        "theta_clicked": snap.get("theta_clicked", 0.5),
                        "beta_clicked": snap.get("beta_clicked", 0.0),
                        "subsidized": snap.get("subsidized", False),
                    })
        return pd.DataFrame(rows)

    def get_world_comparison_df(self, agent_id: int) -> pd.DataFrame:
        """获取某个Agent跨世界的参数对比DataFrame"""
        if agent_id not in self.tracker:
            return pd.DataFrame()
        return self.tracker[agent_id].compare_worlds()
