"""
DoWhy因果图模块
基于DoWhy 0.14 API，构建补贴场景的完整因果图并进行反驳验证

功能：
1. 构建补贴→核销的因果图（DAG）
2. 估计ATE（平均处理效应），支持多种方法
3. 反驳验证（Bootstrap、Placebo、Random Common Cause）
4. 多方法对比（PSM、IPW、Regression）

参考文献：
- DoWhy (Microsoft): https://github.com/py-why/dowhy
- Pearl (2009): Causality: Models, Reasoning, and Inference
- DoWhy 0.14 API参考（已验证可用）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

# DoWhy导入
from dowhy import CausalModel


@dataclass
class DoWhyConfig:
    """DoWhy因果分析配置"""
    # 默认估计方法
    method_name: str = "backdoor.propensity_score_matching"
    # 反驳验证方法列表
    refutation_methods: List[str] = field(default_factory=lambda: [
        "bootstrap_refuter",
        "placebo_treatment_refuter",
        "random_common_cause"
    ])
    # 随机种子
    random_state: int = 42
    # Bootstrap样本数
    n_bootstrap_samples: int = 100
    # 是否在不可识别时继续
    proceed_when_unidentifiable: bool = True


class SubsidyCausalGraph:
    """
    补贴因果图分析器

    基于DoWhy框架，构建补贴策略的因果图（DAG），
    估计补贴对核销率的因果效应，并进行反驳验证。

    因果图结构：
    - user_profile（用户画像）→ subsidy（补贴）
    - user_profile（用户画像）→ redemption（核销）
    - context（上下文）→ subsidy（补贴）
    - context（上下文）→ redemption（核销）
    - subsidy（补贴）→ redemption（核销）
    - U（未观测混杂）→ subsidy、redemption
    """

    def __init__(self, config: Optional[DoWhyConfig] = None):
        """
        初始化因果图分析器

        参数：
        - config: DoWhy配置对象，若None则使用默认配置
        """
        self.config = config or DoWhyConfig()
        self.model: Optional[CausalModel] = None
        self.identified_estimand = None  # 保存识别结果，避免重复调用identify_effect
        self.estimate = None
        self.refutation_results: Optional[Dict[str, Any]] = None

    def build_causal_graph(self, include_unobserved: bool = True) -> str:
        """
        构建通用因果图（DOT格式字符串）

        节点：
        - user_profile: 用户画像（年龄、性别、城市等级等）
        - context: 上下文（时间、POI类型、天气等）
        - subsidy: 补贴策略（券面额、门槛、折扣率等）
        - redemption: 核销行为（结果变量，是否核销/核销金额）
        - U: 未观测混杂因子（用户偏好、消费习惯等）

        参数：
        - include_unobserved: 是否包含未观测混杂因子

        返回：
        - DOT格式的因果图字符串
        """
        if include_unobserved:
            graph = """
            digraph {
                user_profile -> subsidy;
                user_profile -> redemption;
                user_profile -> context;
                context -> subsidy;
                context -> redemption;
                subsidy -> redemption;
                U[label="未观测混杂"];
                U -> subsidy;
                U -> redemption;
            }
            """
        else:
            graph = """
            digraph {
                user_profile -> subsidy;
                user_profile -> redemption;
                context -> subsidy;
                context -> redemption;
                subsidy -> redemption;
            }
            """
        return graph

    def build_subsidy_dag(self) -> str:
        """
        构建补贴场景的完整因果图（DOT格式）

        相比build_causal_graph，此方法构建更细粒度的补贴场景因果图，
        包含具体的特征节点和因果路径：

        节点：
        - age, gender, city_tier: 用户画像特征
        - poi_type, time_period, weather: 上下文特征
        - coupon_amount, threshold, discount_rate: 补贴策略参数
        - price_sensitivity, brand_loyalty: 未观测混杂因子
        - redemption: 核销行为（结果变量）

        因果路径：
        - 用户画像 → 补贴策略（画像影响补贴配置）
        - 用户画像 → 核销（画像直接影响核销意愿）
        - 上下文 → 补贴策略（场景影响补贴策略）
        - 上下文 → 核销（场景影响核销概率）
        - 补贴策略 → 核销（核心因果路径）
        - 未观测混杂 → 补贴、核销（价格敏感度、品牌忠诚度等）

        返回：
        - DOT格式的完整补贴因果图
        """
        dag = """
        digraph 补贴场景因果图 {
            // 图属性
            rankdir=TB;
            label="补贴策略因果图 (Subsidy Causal DAG)";
            labelloc=t;
            fontsize=16;

            // 用户画像层
            subgraph cluster_profile {
                label="用户画像";
                style=filled;
                fillcolor="#E8F5E9";
                age [label="年龄"];
                gender [label="性别"];
                city_tier [label="城市等级"];
            }

            // 上下文层
            subgraph cluster_context {
                label="上下文特征";
                style=filled;
                fillcolor="#E3F2FD";
                poi_type [label="POI类型"];
                time_period [label="时间段"];
                weather [label="天气"];
            }

            // 补贴策略层（处理变量）
            subgraph cluster_subsidy {
                label="补贴策略 (处理变量)";
                style=filled;
                fillcolor="#FFF3E0";
                coupon_amount [label="券面额"];
                threshold [label="使用门槛"];
                discount_rate [label="折扣率"];
            }

            // 未观测混杂层
            subgraph cluster_unobserved {
                label="未观测混杂因子";
                style=filled;
                fillcolor="#FCE4EC";
                price_sensitivity [label="价格敏感度", color=red, style=dashed];
                brand_loyalty [label="品牌忠诚度", color=red, style=dashed];
            }

            // 结果变量
            redemption [label="核销行为\n(结果变量)", shape=box, style=filled, fillcolor="#FFF9C4"];

            // 用户画像 → 补贴策略
            age -> coupon_amount;
            age -> threshold;
            gender -> coupon_amount;
            city_tier -> coupon_amount;
            city_tier -> threshold;

            // 用户画像 → 核销
            age -> redemption;
            gender -> redemption;
            city_tier -> redemption;

            // 用户画像 → 上下文
            age -> poi_type;
            gender -> poi_type;
            city_tier -> time_period;

            // 上下文 → 补贴策略
            poi_type -> coupon_amount;
            poi_type -> threshold;
            time_period -> coupon_amount;
            weather -> discount_rate;

            // 上下文 → 核销
            poi_type -> redemption;
            time_period -> redemption;
            weather -> redemption;

            // 补贴策略 → 核销（核心因果路径）
            coupon_amount -> redemption;
            threshold -> redemption;
            discount_rate -> redemption;

            // 未观测混杂 → 补贴策略
            price_sensitivity -> coupon_amount;
            price_sensitivity -> threshold;
            price_sensitivity -> discount_rate;
            brand_loyalty -> coupon_amount;

            // 未观测混杂 → 核销
            price_sensitivity -> redemption;
            brand_loyalty -> redemption;
        }
        """
        return dag

    def estimate_ate(
        self,
        df: pd.DataFrame,
        treatment_col: str = "subsidy",
        outcome_col: str = "redemption",
        common_causes: Optional[List[str]] = None,
        graph: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        估计ATE（平均处理效应）

        使用DoWhy 0.14 API，通过backdoor准则识别并估计因果效应。
        优先使用common_causes方式构建模型（更稳定），
        graph仅作为可选的可视化参考。

        参数：
        - df: 数据DataFrame
        - treatment_col: 处理变量列名
        - outcome_col: 结果变量列名
        - common_causes: 混杂变量列表，若None则自动识别
        - graph: 因果图（DOT格式），若None则自动构建

        返回：
        - 包含ATE、置信区间、估计方法等信息的字典
        """
        # 自动识别混杂变量
        if common_causes is None:
            common_causes = self._identify_common_causes(df, treatment_col, outcome_col)

        # 使用DoWhy 0.14 API创建因果模型
        # 注意：graph DOT字符串中的节点名需要与DataFrame列名匹配，
        # 否则DoWhy会将不匹配的节点视为未观测变量，导致无法识别backdoor估计量。
        # 因此这里仅使用common_causes参数（更稳定），graph仅用于可视化。
        self.model = CausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=common_causes
        )

        # 识别因果效应（保存结果，供反驳测试复用）
        self.identified_estimand = self.model.identify_effect(
            proceed_when_unidentifiable=self.config.proceed_when_unidentifiable
        )

        # 估计因果效应
        self.estimate = self.model.estimate_effect(
            self.identified_estimand,
            method_name=self.config.method_name
        )

        # 安全获取ATE值和置信区间
        # DoWhy 0.14的estimate.value可能是numpy类型，转换为Python float
        ate_value = self.estimate.value
        if ate_value is not None:
            ate_value = float(ate_value)

        ate_lower_ci = None
        ate_upper_ci = None

        # 尝试从估计结果获取置信区间
        try:
            ci = self.estimate.get_confidence_intervals()
            if ci is not None:
                ate_lower_ci = float(ci[0])
                ate_upper_ci = float(ci[1])
        except (AttributeError, TypeError, IndexError, ValueError):
            # 使用Bootstrap手动计算置信区间作为fallback
            ate_lower_ci, ate_upper_ci = self._bootstrap_confidence_interval(
                df, treatment_col, outcome_col
            )

        # 如果ATE为None（识别失败），使用Bootstrap估计
        if ate_value is None:
            warnings.warn(
                f"estimate.value 返回 None，使用Bootstrap均值差作为ATE估计"
            )
            # 计算简单均值差
            treated_mean = float(df[df[treatment_col] == 1][outcome_col].mean())
            control_mean = float(df[df[treatment_col] == 0][outcome_col].mean())
            ate_value = treated_mean - control_mean
            if ate_lower_ci is None:
                ate_lower_ci, ate_upper_ci = self._bootstrap_confidence_interval(
                    df, treatment_col, outcome_col
                )

        return {
            "ate": ate_value,
            "ate_lower_ci": ate_lower_ci,
            "ate_upper_ci": ate_upper_ci,
            "estimand": str(self.identified_estimand),
            "method": self.config.method_name,
            "common_causes": common_causes
        }

    def _bootstrap_confidence_interval(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        n_bootstrap: int = 200,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Bootstrap方法估算ATE置信区间（fallback）

        参数：
        - df: 数据
        - treatment_col: 处理列名
        - outcome_col: 结果列名
        - n_bootstrap: Bootstrap重复次数
        - alpha: 显著性水平

        返回：
        - (lower_ci, upper_ci) 置信区间
        """
        np.random.seed(self.config.random_state)
        n = len(df)
        ate_bootstrap = []

        for _ in range(n_bootstrap):
            # 有放回抽样
            indices = np.random.choice(n, size=n, replace=True)
            df_boot = df.iloc[indices]

            # 简单均值差估计（假设随机化后无混杂）
            treated_mean = df_boot[df_boot[treatment_col] == 1][outcome_col].mean()
            control_mean = df_boot[df_boot[treatment_col] == 0][outcome_col].mean()
            ate_bootstrap.append(treated_mean - control_mean)

        ate_bootstrap = np.array(ate_bootstrap)
        lower = np.percentile(ate_bootstrap, 100 * alpha / 2)
        upper = np.percentile(ate_bootstrap, 100 * (1 - alpha / 2))

        return float(lower), float(upper)

    def _identify_common_causes(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str
    ) -> List[str]:
        """
        自动识别混杂变量

        使用所有非处理、非结果的数值型列作为候选混杂变量。

        参数：
        - df: 数据DataFrame
        - treatment_col: 处理变量列名
        - outcome_col: 结果变量列名

        返回：
        - 混杂变量列名列表
        """
        all_cols = df.columns.tolist()
        common_causes = [
            col for col in all_cols
            if col not in [treatment_col, outcome_col]
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        return common_causes

    def refutation_test(self, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        反驳验证（Refutation Test）

        测试因果估计的鲁棒性。使用保存的identified_estimand，
        避免重复调用identify_effect。

        支持的反驳方法：
        - bootstrap_refuter: Bootstrap重抽样验证
        - placebo_treatment_refuter: 安慰剂处理验证
        - random_common_cause: 随机混杂因子验证
        - data_subset_refuter: 数据子集验证

        参数：
        - methods: 反驳方法列表，若None则使用配置中的方法

        返回：
        - 各反驳方法的验证结果字典

        异常：
        - ValueError: 若未先运行estimate_ate()
        """
        if self.model is None or self.estimate is None:
            raise ValueError("必须先运行 estimate_ate()")

        if self.identified_estimand is None:
            raise ValueError("identified_estimand 未保存，请重新运行 estimate_ate()")

        if methods is None:
            methods = self.config.refutation_methods

        self.refutation_results = {}

        for method in methods:
            try:
                # 根据方法名构建参数
                method_kwargs = {}
                if method == "bootstrap_refuter":
                    method_kwargs["num_simulations"] = self.config.n_bootstrap_samples
                elif method == "data_subset_refuter":
                    method_kwargs["subset_fraction"] = 0.8

                # 使用保存的identified_estimand，避免重复调用identify_effect
                result = self.model.refute_estimate(
                    self.identified_estimand,
                    self.estimate,
                    method_name=method,
                    **method_kwargs
                )

                self.refutation_results[method] = {
                    "passed": result.refutation_result,
                    "new_effect": getattr(result, "new_effect", None),
                    "p_value": getattr(result, "p_value", None),
                    "details": str(result)
                }

            except Exception as e:
                self.refutation_results[method] = {
                    "passed": False,
                    "error": str(e)
                }

        return self.refutation_results

    def compare_methods(
        self,
        df: pd.DataFrame,
        treatment_col: str = "subsidy",
        outcome_col: str = "redemption",
        common_causes: Optional[List[str]] = None,
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        对比多种因果效应估计方法

        使用同一因果模型，对比不同估计方法的ATE结果。
        支持的方法包括PSM、IPW、回归等。

        参数：
        - df: 数据DataFrame
        - treatment_col: 处理变量列名
        - outcome_col: 结果变量列名
        - common_causes: 混杂变量列表
        - methods: 要对比的方法列表，默认包含PSM、IPW、回归

        返回：
        - 各方法的ATE估计结果DataFrame
        """
        if common_causes is None:
            common_causes = self._identify_common_causes(df, treatment_col, outcome_col)

        if methods is None:
            methods = [
                "backdoor.propensity_score_matching",
                "backdoor.propensity_score_weighting",
                "backdoor.linear_regression",
                "backdoor.propensity_score_stratification"
            ]

        # 构建因果模型（仅构建一次）
        self.model = CausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=common_causes
        )

        # 识别因果效应（仅识别一次）
        self.identified_estimand = self.model.identify_effect(
            proceed_when_unidentifiable=self.config.proceed_when_unidentifiable
        )

        # 对比各方法
        results = []
        for method in methods:
            try:
                estimate = self.model.estimate_effect(
                    self.identified_estimand,
                    method_name=method
                )

                # 安全获取置信区间
                lower_ci = None
                upper_ci = None
                try:
                    ci = estimate.get_confidence_intervals()
                    if ci is not None:
                        lower_ci = ci[0]
                        upper_ci = ci[1]
                except (AttributeError, TypeError, IndexError):
                    pass

                results.append({
                    "method": method,
                    "ate": float(estimate.value) if estimate.value is not None else None,
                    "ate_lower_ci": lower_ci,
                    "ate_upper_ci": upper_ci,
                })

            except Exception as e:
                results.append({
                    "method": method,
                    "ate": None,
                    "ate_lower_ci": None,
                    "ate_upper_ci": None,
                    "error": str(e)
                })

        # 返回结果DataFrame
        result_df = pd.DataFrame(results)

        # 将最后一次估计作为默认estimate
        if results and results[-1]["ate"] is not None:
            self.estimate = self.model.estimate_effect(
                self.identified_estimand,
                method_name=methods[0]
            )

        return result_df

    def visualize_causal_graph(self, save_path: Optional[str] = None):
        """
        可视化因果图

        使用graphviz渲染DOT格式的因果图。
        需要安装: pip install graphviz

        参数：
        - save_path: 保存路径（不含扩展名），若None则返回graphviz对象

        返回：
        - graphviz.Source对象，或None（若graphviz未安装）
        """
        try:
            import graphviz

            dot_str = self.build_causal_graph()
            graph = graphviz.Source(dot_str)

            if save_path:
                graph.render(save_path, format="png", cleanup=True)
                print(f"因果图已保存至 {save_path}.png")
            else:
                return graph

        except ImportError:
            print("graphviz未安装。安装命令: pip install graphviz")
            return None

    def compare_with_ipw(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        ipw_ate: float
    ) -> Dict[str, Any]:
        """
        与IPW（逆概率加权）方法结果对比

        参数：
        - df: 数据DataFrame
        - treatment_col: 处理变量列名
        - outcome_col: 结果变量列名
        - ipw_ate: IPW方法估计的ATE

        返回：
        - 对比结果字典，包含DoWhy ATE、IPW ATE及差异
        """
        dowhy_result = self.estimate_ate(df, treatment_col, outcome_col)
        dowhy_ate = dowhy_result["ate"]

        return {
            "dowhy_ate": dowhy_ate,
            "ipw_ate": ipw_ate,
            "difference": dowhy_ate - ipw_ate,
            "relative_difference": (
                (dowhy_ate - ipw_ate) / abs(ipw_ate) if ipw_ate != 0 else None
            )
        }

    def summary(self) -> str:
        """
        生成因果分析摘要

        返回：
        - 格式化的摘要字符串
        """
        lines = ["=" * 60]
        lines.append("补贴因果分析摘要 (Subsidy Causal Analysis Summary)")
        lines.append("=" * 60)

        if self.estimate is not None:
            lines.append(f"\n估计方法: {self.config.method_name}")
            lines.append(f"ATE (平均处理效应): {self.estimate.value:.6f}")

            # 尝试获取置信区间
            try:
                ci = self.estimate.get_confidence_intervals()
                if ci is not None:
                    lines.append(f"95% 置信区间: [{ci[0]:.6f}, {ci[1]:.6f}]")
            except (AttributeError, TypeError, IndexError):
                lines.append("置信区间: 不可用（估计方法不支持）")

        if self.refutation_results:
            lines.append(f"\n反驳验证结果:")
            for method, result in self.refutation_results.items():
                status = "通过" if result.get("passed") else "未通过"
                error = result.get("error", "")
                error_str = f" (错误: {error})" if error else ""
                lines.append(f"  - {method}: {status}{error_str}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def run_dowhy_example():
    """
    运行DoWhy因果分析完整示例

    生成合成数据，演示补贴因果图的完整分析流程：
    1. 构建因果图
    2. 估计ATE
    3. 反驳验证
    4. 多方法对比
    5. 输出摘要
    """
    print("=" * 60)
    print("DoWhy 补贴因果分析示例")
    print("基于 DoWhy 0.14 API")
    print("=" * 60)

    # 生成合成数据
    np.random.seed(42)
    n_samples = 5000

    # 用户画像特征（混杂因子）
    user_feat_1 = np.random.randn(n_samples)
    user_feat_2 = np.random.randn(n_samples)
    user_feat_3 = np.random.randn(n_samples)

    # 上下文特征（混杂因子）
    context_1 = np.random.randn(n_samples)
    context_2 = np.random.randn(n_samples)

    # 补贴（处理变量）：受用户画像和上下文影响
    propensity = 1 / (1 + np.exp(-(0.3 * user_feat_1 + 0.2 * context_1)))
    subsidy = np.random.binomial(1, propensity, n_samples)

    # 核销（结果变量）：真实处理效应 = 2.0
    redemption = (
        user_feat_1 * 0.5
        + user_feat_2 * 0.3
        + context_1 * 0.4
        + subsidy * 2.0  # 真实ATE = 2.0
        + np.random.randn(n_samples) * 0.5
    )

    # 构建DataFrame
    df = pd.DataFrame({
        "user_feat_1": user_feat_1,
        "user_feat_2": user_feat_2,
        "user_feat_3": user_feat_3,
        "context_1": context_1,
        "context_2": context_2,
        "subsidy": subsidy,
        "redemption": redemption
    })

    print(f"\n数据集: {n_samples} 样本, {len(df.columns)} 列")
    print(f"补贴发放比例: {subsidy.mean():.2%}")
    print(f"真实ATE: 2.0")

    # 初始化因果图分析器
    config = DoWhyConfig(method_name="backdoor.propensity_score_matching")
    causal_graph = SubsidyCausalGraph(config)

    # 步骤1: 估计ATE
    print("\n" + "-" * 40)
    print("步骤1: 估计ATE (PSM方法)")
    print("-" * 40)
    ate_result = causal_graph.estimate_ate(df, "subsidy", "redemption")
    print(f"  ATE = {ate_result['ate']:.4f}")
    if ate_result['ate_lower_ci'] is not None:
        print(f"  95% CI: [{ate_result['ate_lower_ci']:.4f}, {ate_result['ate_upper_ci']:.4f}]")
    print(f"  方法: {ate_result['method']}")

    # 步骤2: 反驳验证
    print("\n" + "-" * 40)
    print("步骤2: 反驳验证")
    print("-" * 40)
    refutation_results = causal_graph.refutation_test()
    for method, result in refutation_results.items():
        status = "通过" if result.get("passed") else "未通过"
        if "error" in result:
            print(f"  {method}: {status} (错误: {result['error']})")
        else:
            print(f"  {method}: {status}")

    # 步骤3: 多方法对比
    print("\n" + "-" * 40)
    print("步骤3: 多方法对比")
    print("-" * 40)
    comparison = causal_graph.compare_methods(df, "subsidy", "redemption")
    for _, row in comparison.iterrows():
        ate_val = row['ate']
        if ate_val is not None:
            print(f"  {row['method']}: ATE = {ate_val:.4f}")
        else:
            print(f"  {row['method']}: 失败 ({row.get('error', '未知错误')})")

    # 步骤4: 输出摘要
    print("\n" + causal_graph.summary())

    return ate_result, refutation_results, comparison


if __name__ == "__main__":
    run_dowhy_example()
