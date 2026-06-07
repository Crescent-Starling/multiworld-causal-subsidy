"""
基于Mesa 3.x的补贴策略Agent-Based仿真模型

核心创新：
1. 前景理论价值函数（Kahneman & Tversky, 1979）建模用户非理性决策
2. 心理账户参考点动态更新（Thaler, 1985）捕捉补贴脱敏
3. 有限理性折扣（Simon, 1955）建模认知约束
4. 疲劳脱敏累积机制模拟长期补贴效果递减
5. MultiWorldModel多平行世界仿真解耦假设风险与随机噪声

参考文献：
- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. Econometrica, 47(2), 263-291.
- Thaler, R. H. (1985). Mental Accounting and Consumer Choice. Marketing Science, 4(3), 199-214.
- Simon, H. A. (1955). A Behavioral Model of Rational Choice. The Quarterly Journal of Economics, 69(1), 99-118.
- Tversky, A., & Kahneman, D. (1992). Advances in Prospect Theory: Cumulative Representation of Uncertainty. Journal of Risk and Uncertainty, 5(4), 297-323.
- Mesa Framework: https://github.com/projectmesa/mesa
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from mesa import Agent, Model

# 认知Agent理论模块（前景理论、心理账户、有限理性、疲劳脱敏）
from src.simulation.cognitive_agent_theory import (
    prospect_discount,
    bounded_rationality_discount,
    fatigue_update,
    update_reference_point,
    check_account_transition,
    MentalAccountType,
    DEFAULT_ETA_DICT,
    DEFAULT_DESENS_RATES,
)


# ===========================================================================
# 枚举与数据类
# ===========================================================================

class StrategyType(str, Enum):
    """补贴策略类型"""
    RANDOM = "random"         # 随机策略：随机分配（基线）
    STATIC = "static"         # 静态策略：基于价格敏感度排序分配固定金额
    DYNAMIC = "dynamic"       # 动态策略：基于用户画像+疲劳度动态调整金额
    COGNITIVE = "cognitive"    # 认知策略：基于前景理论+心理账户+有限理性


@dataclass
class SimulationResult:
    """仿真结果数据类"""
    strategy: StrategyType
    round_metrics: list[dict] = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "round_metrics": self.round_metrics,
            "final_metrics": self.final_metrics,
        }


# ===========================================================================
# 心理账户类型映射
# ===========================================================================

def _map_mental_account(price_sensitivity: float, income_level: int) -> MentalAccountType:
    """基于用户画像映射心理账户类型"""
    if price_sensitivity > 0.5 and income_level <= 2:
        return MentalAccountType.WINDFALL_SPENDER
    elif price_sensitivity > 0.3:
        return MentalAccountType.PRICE_SENSITIVE
    elif income_level >= 4:
        return MentalAccountType.ROUTINE_INCOME
    else:
        return MentalAccountType.DEAL_SEEKER


# ===========================================================================
# SubsidyAgent
# ===========================================================================

class SubsidyAgent(Agent):
    """
    单用户仿真Agent

    基于行为经济学理论建模用户对补贴的响应决策：
    1. 前景理论价值函数 → 非线性主观价值评估（Kahneman & Tversky, 1979）
    2. 有限理性折扣 → 认知约束下的次优决策（Simon, 1955）
    3. 心理账户参考点更新 → 补贴脱敏效应（Thaler, 1985）
    4. 疲劳脱敏累积 → 长期补贴效果递减

    使用Mesa 3.x API：
    - super().__init__(model) 替代旧的 unique_id 参数
    - 通过 self.model 访问模型上下文
    """

    def __init__(
        self,
        model: Model,
        agent_id: int,
        price_sensitivity: float = 0.5,
        income_level: int = 3,
        city_tier: int = 3,
        mental_account: Optional[MentalAccountType] = None,
        alpha: float = 0.88,
        lambda_: float = 2.25,
        decision_threshold: float = 0.3,
    ):
        """
        初始化Agent

        参数：
        - model: Mesa Model实例
        - agent_id: Agent编号
        - price_sensitivity: 价格敏感度 [0, 1]
        - income_level: 收入等级 1-5
        - city_tier: 城市等级 1-5
        - mental_account: 心理账户类型
        - alpha: 前景理论价值函数曲率
        - lambda_: 损失厌恶系数
        - decision_threshold: 核销决策阈值
        """
        super().__init__(model)

        # Agent标识
        self.agent_id = agent_id

        # 用户画像
        self.price_sensitivity = price_sensitivity
        self.income_level = income_level
        self.city_tier = city_tier

        # 心理账户（Thaler, 1985）
        self.mental_account = mental_account or _map_mental_account(
            price_sensitivity, income_level
        )

        # 前景理论参数（Kahneman & Tversky, 1979）
        self.alpha = alpha       # 价值函数曲率
        self.lambda_ = lambda_   # 损失厌恶系数

        # 决策阈值
        self.decision_threshold = decision_threshold

        # 内部状态
        self.reference_point = 0.0    # 心理账户参考点
        self.fatigue = 0.0           # 疲劳度
        self.cognitive_load = 3.0 + (5 - income_level) * 0.5  # 认知负荷

        # 当前轮次状态
        self.subsidized = False
        self.subsidy_amount = 0.0
        self.redeemed = False
        self.base_gtv = np.random.lognormal(3.5, 0.8)  # 基础GTV

        # 累积指标
        self.total_subsidy = 0.0
        self.total_gtv = 0.0
        self.total_redemptions = 0

        # 本轮快照（用于指标收集，step()后仍可读取）
        self._step_subsidy_amount = 0.0
        self._step_subsidized = False

    def receive_subsidy(self, amount: float) -> None:
        """接收补贴"""
        self.subsidized = True
        self.subsidy_amount = amount
        self.total_subsidy += amount

    def step(self) -> None:
        """
        Agent每步决策逻辑

        决策流程：
        1. 前景理论价值评估
        2. 有限理性折扣
        3. 疲劳脱敏折扣
        4. 心理账户影响
        5. 综合决策（与阈值+噪声比较）
        """
        # 保存本轮快照
        self._step_subsidized = self.subsidized
        self._step_subsidy_amount = self.subsidy_amount

        if not self.subsidized:
            self.redeemed = False
            # 无补贴时仍更新状态（疲劳衰减、参考点回落）
            self._update_state()
            return

        # 步骤1: 前景理论价值（Kahneman & Tversky, 1979）
        pv = prospect_discount(
            subsidy=self.subsidy_amount,
            reference=self.reference_point,
            alpha=self.alpha,
            lambda_=self.lambda_,
        )

        # 步骤2: 归一化激活值（sigmoid变换）
        activation = 1.0 / (1.0 + np.exp(-pv / max(self.base_gtv, 1.0)))

        # 步骤3: 有限理性折扣（Simon, 1955）
        br = bounded_rationality_discount(self.cognitive_load)

        # 步骤4: 疲劳脱敏折扣
        fatigue_discount = np.exp(-0.3 * self.fatigue)

        # 步骤5: 综合决策
        effective = activation * br * fatigue_discount

        # 心理账户影响（Thaler, 1985）
        account_boost = {
            MentalAccountType.WINDFALL_SPENDER: 0.15,
            MentalAccountType.PRICE_SENSITIVE: 0.10,
            MentalAccountType.ROUTINE_INCOME: -0.05,
            MentalAccountType.DEAL_SEEKER: 0.08,
        }
        effective += account_boost.get(self.mental_account, 0.0)

        # 随机噪声
        noise = np.random.normal(0, 0.1)

        # 决策：有效激活值超过阈值则核销
        self.redeemed = effective > (self.decision_threshold + noise)

        if self.redeemed:
            self.total_gtv += self.base_gtv + self.subsidy_amount * 0.5
            self.total_redemptions += 1
        else:
            self.total_gtv += self.base_gtv * 0.3

        # 更新内部状态
        self._update_state()

    def _update_state(self) -> None:
        """更新内部状态（疲劳、参考点、心理账户迁移）"""
        # 疲劳更新
        self.fatigue = fatigue_update(
            fatigue=self.fatigue,
            was_subsidized=self.subsidized,
            account_type=self.mental_account,
        )

        # 参考点更新
        if self.subsidized:
            outcome = self.reference_point * 0.8 + 0.2 * (1.0 if self.redeemed else 0.0)
            self.reference_point = update_reference_point(
                ref=self.reference_point,
                outcome=outcome,
                account_type=self.mental_account,
            )

        # 心理账户迁移
        self.mental_account = check_account_transition(
            fatigue=self.fatigue,
            account_type=self.mental_account,
        )

        # 重置轮次状态
        self.subsidized = False
        self.subsidy_amount = 0.0


# ===========================================================================
# SubsidyModel
# ===========================================================================

class SubsidyModel(Model):
    """
    补贴策略仿真模型

    使用Mesa 3.x API：
    - super().__init__() 初始化
    - self.agents.do('step') 替代旧的 schedule.step()
    - self.agents 是 AgentSet，支持向量化操作

    仿真流程：
    1. 策略分配：根据选定策略为Agent分配补贴
    2. Agent决策：每个Agent独立执行step()
    3. 收集结果：汇总全局指标（ROI, ΔGTV, Coverage）
    """

    def __init__(
        self,
        n_agents: int = 500,
        strategy: str = "cognitive",
        budget_ratio: float = 0.3,
        subsidy_amount: float = 10.0,
        seed: int = 42,
        user_profiles: Optional[pd.DataFrame] = None,
    ):
        """
        初始化仿真模型

        参数：
        - n_agents: Agent数量
        - strategy: 补贴策略类型（random/static/dynamic/cognitive）
        - budget_ratio: 预算覆盖比例（受补贴用户占比）
        - subsidy_amount: 基础补贴金额（元）
        - seed: 随机种子
        - user_profiles: 可选的用户画像DataFrame
        """
        super().__init__()
        self.n_agents = n_agents
        self.strategy = StrategyType(strategy)
        self.budget_ratio = budget_ratio
        self.subsidy_amount = subsidy_amount
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.current_round = 0
        self.round_results: list[dict] = []

        # 创建Agent
        self._create_agents(user_profiles)

    def _create_agents(self, user_profiles: Optional[pd.DataFrame] = None) -> None:
        """创建Agent群体"""
        if user_profiles is not None:
            n = min(self.n_agents, len(user_profiles))
            for i in range(n):
                row = user_profiles.iloc[i]
                agent = SubsidyAgent(
                    model=self,
                    agent_id=i,
                    price_sensitivity=row.get("price_sensitivity", 0.5),
                    income_level=int(row.get("income_level", 3)),
                    city_tier=int(row.get("city_tier", 3)),
                )
        else:
            for i in range(self.n_agents):
                ps = self.rng.beta(2, 5)
                income = self.rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.35, 0.25, 0.1])
                city = self.rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.30, 0.20, 0.10])
                agent = SubsidyAgent(
                    model=self,
                    agent_id=i,
                    price_sensitivity=ps,
                    income_level=income,
                    city_tier=city,
                )

    def _allocate_subsidy(self) -> None:
        """
        根据策略分配补贴

        四种策略：
        - RANDOM: 随机选择Agent分配
        - STATIC: 按价格敏感度排序分配
        - DYNAMIC: 综合评分+动态金额
        - COGNITIVE: 前景理论边际价值+心理账户+有限理性
        """
        n_to_subsidize = int(self.n_agents * self.budget_ratio)
        agents_list = list(self.agents)

        if self.strategy == StrategyType.RANDOM:
            indices = self.rng.choice(self.n_agents, n_to_subsidize, replace=False)
            selected_ids = set(indices.tolist())
            for agent in agents_list:
                if agent.agent_id in selected_ids:
                    agent.receive_subsidy(self.subsidy_amount)

        elif self.strategy == StrategyType.STATIC:
            # 按价格敏感度降序排列，补贴最敏感的用户
            agents_list.sort(key=lambda a: a.price_sensitivity, reverse=True)
            for agent in agents_list[:n_to_subsidize]:
                agent.receive_subsidy(self.subsidy_amount)

        elif self.strategy == StrategyType.DYNAMIC:
            # 综合评分 = 价格敏感度 × (1 - 疲劳/5) + 城市等级因子
            def dynamic_score(a):
                city_factor = (5 - a.city_tier) / 4.0 * 0.2
                return a.price_sensitivity * (1.0 - a.fatigue / 5.0) + city_factor
            agents_list.sort(key=dynamic_score, reverse=True)
            for agent in agents_list[:n_to_subsidize]:
                # 动态调整金额
                amount = self.subsidy_amount * (1.0 + 0.2 * (agent.price_sensitivity - 0.3))
                agent.receive_subsidy(max(amount, 5.0))

        elif self.strategy == StrategyType.COGNITIVE:
            # 认知理论驱动：前景理论边际价值 × 有限理性 × 疲劳折扣 + 心理账户加成
            def cognitive_score(a):
                pv = prospect_discount(
                    subsidy=self.subsidy_amount,
                    reference=a.reference_point,
                    alpha=a.alpha,
                    lambda_=a.lambda_,
                )
                br = bounded_rationality_discount(a.cognitive_load)
                fd = np.exp(-0.3 * a.fatigue)
                account_boost = {
                    MentalAccountType.WINDFALL_SPENDER: 0.15,
                    MentalAccountType.PRICE_SENSITIVE: 0.10,
                    MentalAccountType.ROUTINE_INCOME: -0.05,
                    MentalAccountType.DEAL_SEEKER: 0.08,
                }
                return pv * br * fd + account_boost.get(a.mental_account, 0.0)

            agents_list.sort(key=cognitive_score, reverse=True)
            for agent in agents_list[:n_to_subsidize]:
                # 根据心理账户调整补贴金额
                if agent.mental_account == MentalAccountType.WINDFALL_SPENDER:
                    amount = self.subsidy_amount * 1.2  # 横财型需要更多刺激
                elif agent.mental_account == MentalAccountType.PRICE_SENSITIVE:
                    amount = self.subsidy_amount * 0.8  # 价格敏感型少给也行
                else:
                    amount = self.subsidy_amount
                agent.receive_subsidy(amount)

    def step(self) -> None:
        """
        模型每步执行

        流程：
        1. 策略分配 → 为Agent分配补贴
        2. Agent决策 → 执行所有Agent的step()
        3. 收集结果 → 汇总指标
        """
        self.current_round += 1

        # 1. 分配补贴
        self._allocate_subsidy()

        # 2. Agent决策（Mesa 3.x: 使用 agents.do() 替代 schedule.step()）
        self.agents.do("step")

        # 3. 收集指标
        self.collect_results()

    def collect_results(self) -> dict:
        """
        收集当前轮次的结果

        指标：
        - ROI: 核销增量回报
        - ΔGTV: 补贴驱动的总交易增量
        - Coverage: 受补贴用户覆盖率
        - RedemptionRate: 核销率
        """
        agents_list = list(self.agents)

        # 处理组 vs 对照组（用快照数据，避免step后状态重置的影响）
        treated = [a for a in agents_list if a._step_subsidized]
        control = [a for a in agents_list if not a._step_subsidized]

        total_subsidy_step = sum(a._step_subsidy_amount for a in treated)
        treated_gtv = sum(a.total_gtv for a in treated)
        control_gtv = sum(a.total_gtv for a in control) if control else 0.0

        n_treated = len(treated)
        n_redeemed = sum(1 for a in treated if a.redeemed)

        # ΔGTV: 处理组GTV - 对照组GTV的期望值
        delta_gtv = treated_gtv - control_gtv * (n_treated / max(len(control), 1))
        # ROI: (ΔGTV - 补贴成本) / 补贴成本
        roi = (delta_gtv - total_subsidy_step) / total_subsidy_step if total_subsidy_step > 0 else 0.0
        coverage = n_treated / len(agents_list)
        redemption_rate = n_redeemed / n_treated if n_treated > 0 else 0.0

        result = {
            "round": self.current_round,
            "strategy": self.strategy.value,
            "roi": roi,
            "delta_gtv": delta_gtv,
            "coverage": coverage,
            "redemption_rate": redemption_rate,
            "total_subsidy": total_subsidy_step,
            "treated_gtv": treated_gtv,
            "control_gtv": control_gtv,
            "n_treated": n_treated,
            "n_redeemed": n_redeemed,
            "avg_fatigue": np.mean([a.fatigue for a in agents_list]),
            "avg_reference_point": np.mean([a.reference_point for a in agents_list]),
        }

        self.round_results.append(result)
        return result

    def get_summary(self) -> pd.DataFrame:
        """获取所有轮次的结果汇总"""
        return pd.DataFrame(self.round_results)

    def run(self, n_rounds: int = 30) -> SimulationResult:
        """
        运行完整仿真

        参数：
        - n_rounds: 仿真轮数

        返回：
        - SimulationResult
        """
        for _ in range(n_rounds):
            self.step()

        # 计算最终汇总
        summary = self.get_summary()
        final_metrics = {
            "strategy": self.strategy.value,
            "total_rounds": len(self.round_results),
            "avg_roi": float(summary["roi"].mean()) if len(summary) > 0 else 0.0,
            "cumulative_delta_gtv": float(summary["delta_gtv"].sum()) if len(summary) > 0 else 0.0,
            "avg_coverage": float(summary["coverage"].mean()) if len(summary) > 0 else 0.0,
            "avg_redemption_rate": float(summary["redemption_rate"].mean()) if len(summary) > 0 else 0.0,
            "total_subsidy_spent": float(summary["total_subsidy"].sum()) if len(summary) > 0 else 0.0,
            "final_avg_fatigue": float(summary["avg_fatigue"].iloc[-1]) if len(summary) > 0 else 0.0,
            "final_avg_reference_point": float(summary["avg_reference_point"].iloc[-1]) if len(summary) > 0 else 0.0,
        }

        return SimulationResult(
            strategy=self.strategy,
            round_metrics=self.round_results,
            final_metrics=final_metrics,
        )


# ===========================================================================
# MultiWorldModel — 多平行世界仿真
# ===========================================================================

class MultiWorldModel:
    """
    多平行世界仿真模型

    本项目的核心创新点：同一组Agent在多个"假设世界"中并行运行，
    每个世界应用不同的补贴策略，通过比较不同世界的结果来评估策略效果。

    关键价值：
    1. 解耦假设风险与随机噪声：固定Agent画像，仅改变策略变量
    2. 因果推断视角：多世界对比 ≈ 反事实推断
    3. 排除个体异质性干扰：同一组人在不同策略下的行为差异即策略效果

    参考文献：
    - 多世界仿真思想借鉴自量子力学的"多世界诠释"在社会科学中的类比
    - 因果推断框架参考 Imbens & Rubin (2015): Causal Inference for Statistics,
      Social, and Biomedical Sciences
    """

    def __init__(
        self,
        n_agents: int = 500,
        n_rounds: int = 30,
        seed: int = 42,
        user_profiles: Optional[pd.DataFrame] = None,
    ):
        """
        初始化多世界仿真

        参数：
        - n_agents: 每个世界的Agent数量
        - n_rounds: 每个世界的仿真轮数
        - seed: 随机种子（确保所有世界使用相同的随机初始化）
        - user_profiles: 可选的用户画像DataFrame
        """
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.seed = seed
        self.user_profiles = user_profiles
        self.world_results: dict[str, SimulationResult] = {}

    def add_world(
        self,
        world_name: str,
        strategy: str,
        budget_ratio: float = 0.3,
        subsidy_amount: float = 10.0,
    ) -> SimulationResult:
        """
        添加并运行一个平行世界

        参数：
        - world_name: 世界名称
        - strategy: 策略类型
        - budget_ratio: 预算覆盖比例
        - subsidy_amount: 基础补贴金额

        返回：
        - SimulationResult
        """
        model = SubsidyModel(
            n_agents=self.n_agents,
            strategy=strategy,
            budget_ratio=budget_ratio,
            subsidy_amount=subsidy_amount,
            seed=self.seed,
            user_profiles=self.user_profiles,
        )
        result = model.run(n_rounds=self.n_rounds)
        self.world_results[world_name] = result
        return result

    def run_all_strategies(
        self,
        budget_ratio: float = 0.3,
        subsidy_amount: float = 10.0,
    ) -> dict[str, SimulationResult]:
        """
        运行所有4种策略对比

        返回：
        - 各策略的SimulationResult
        """
        strategies = {
            "random": "random",
            "static": "static",
            "dynamic": "dynamic",
            "cognitive": "cognitive",
        }

        for name, strategy in strategies.items():
            self.add_world(name, strategy, budget_ratio, subsidy_amount)

        return self.world_results

    def compare_worlds(self) -> pd.DataFrame:
        """
        多世界对比分析

        返回DataFrame对比各策略在相同Agent群体下的效果差异
        """
        rows = []
        for name, result in self.world_results.items():
            fm = result.final_metrics
            rows.append({
                "world": name,
                "strategy": fm.get("strategy", name),
                "avg_roi": fm.get("avg_roi", 0),
                "cumulative_delta_gtv": fm.get("cumulative_delta_gtv", 0),
                "avg_coverage": fm.get("avg_coverage", 0),
                "avg_redemption_rate": fm.get("avg_redemption_rate", 0),
                "total_subsidy_spent": fm.get("total_subsidy_spent", 0),
                "final_avg_fatigue": fm.get("final_avg_fatigue", 0),
                "final_avg_reference_point": fm.get("final_avg_reference_point", 0),
            })

        return pd.DataFrame(rows).sort_values("avg_roi", ascending=False)

    def get_step_comparison(self) -> pd.DataFrame:
        """
        获取各策略每轮指标对比的长格式DataFrame
        """
        rows = []
        for name, result in self.world_results.items():
            for m in result.round_metrics:
                rows.append({
                    "world": name,
                    "strategy": m.get("strategy", name),
                    "round": m["round"],
                    "roi": m["roi"],
                    "delta_gtv": m["delta_gtv"],
                    "coverage": m["coverage"],
                    "redemption_rate": m["redemption_rate"],
                    "avg_fatigue": m.get("avg_fatigue", 0),
                    "avg_reference_point": m.get("avg_reference_point", 0),
                })
        return pd.DataFrame(rows)

    def robustness_analysis(self) -> dict[str, Any]:
        """
        稳健性分析

        评估各策略在多世界对比中的稳健性：
        1. ROI排名稳定性
        2. 策略间效应量差异
        3. 变异系数（CV）
        """
        comparison = self.compare_worlds()

        if len(comparison) == 0:
            return {}

        rois = comparison["avg_roi"].values
        delta_gtvs = comparison["cumulative_delta_gtv"].values

        # 最优策略
        best_strategy = comparison.iloc[0]["world"]

        # ROI变异系数
        roi_mean = np.mean(rois)
        roi_cv = np.std(rois) / max(abs(roi_mean), 1e-6)

        # 最优vs最差差异
        roi_range = max(rois) - min(rois)

        return {
            "best_strategy": best_strategy,
            "roi_cv": float(roi_cv),
            "roi_range": float(roi_range),
            "n_worlds": len(comparison),
            "strategies_ranked": comparison[["world", "avg_roi"]].to_dict("records"),
        }


# ===========================================================================
# 运行入口
# ===========================================================================

def run_mesa_simulation(
    n_agents: int = 500,
    n_rounds: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """
    运行Mesa仿真（命令行入口）

    返回：
    - 仿真结果字典
    """
    print("=" * 60)
    print("Mesa ABM 仿真系统")
    print("=" * 60)

    # 多世界仿真
    mw = MultiWorldModel(n_agents=n_agents, n_rounds=n_rounds, seed=seed)
    results = mw.run_all_strategies()

    # 打印每轮结果
    for name, result in results.items():
        print(f"\n--- {name.upper()} Strategy ---")
        for m in result.round_metrics:
            print(f"  Round {m['round']}: "
                  f"ROI={m['roi']:.2f}, "
                  f"ΔGTV={m['delta_gtv']:.1f}, "
                  f"Coverage={m['coverage']:.2%}, "
                  f"RedemptionRate={m['redemption_rate']:.2%}")

    # 对比结果
    comparison = mw.compare_worlds()
    print(f"\n--- Strategy Comparison ---")
    print(comparison.to_string(index=False))

    # 稳健性分析
    robustness = mw.robustness_analysis()
    print(f"\n--- Robustness Analysis ---")
    for k, v in robustness.items():
        print(f"  {k}: {v}")

    return {
        "results": {name: r.to_dict() for name, r in results.items()},
        "comparison": comparison,
        "robustness": robustness,
    }


if __name__ == "__main__":
    run_mesa_simulation()
