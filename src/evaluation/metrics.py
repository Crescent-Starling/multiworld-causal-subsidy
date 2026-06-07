"""
评估指标模块
===========
提供因果推断和仿真策略的评估指标。

参考文献:
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife.
  The Annals of Statistics, 7(1), 1-26.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity Analysis in Observational Research:
  Introducing the E-Value. Annals of Internal Medicine, 167(4), 268-274.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Tuple
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap置信区间 (Efron, 1979)

    参数:
        data: 样本数据
        statistic: 统计量函数（默认均值）
        n_bootstrap: Bootstrap重采样次数
        ci: 置信水平（默认0.95）
        random_state: 随机种子

    返回:
        (lower, upper) 置信区间
    """
    rng = np.random.RandomState(random_state)
    n = len(data)
    boot_stats = []

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_roi(subsidy_cost: float, incremental_gtv: float) -> float:
    """
    计算ROI（投资回报率）

    ROI = (增量GTV - 补贴成本) / 补贴成本

    参数:
        subsidy_cost: 补贴总成本
        incremental_gtv: 增量GTV

    返回:
        ROI值
    """
    if subsidy_cost <= 0:
        return 0.0
    return (incremental_gtv - subsidy_cost) / subsidy_cost


def compute_delta_gtv(treated_gtv: float, control_gtv: float) -> float:
    """
    计算ΔGTV（增量交易总额）

    ΔGTV = 处理组GTV - 对照组GTV

    参数:
        treated_gtv: 处理组GTV
        control_gtv: 对照组GTV

    返回:
        ΔGTV值
    """
    return treated_gtv - control_gtv


def e_value(rr: float) -> float:
    """
    E-value计算 (VanderWeele & Ding, 2017)

    E-value衡量未观测混杂因子需要多强才能完全解释掉观察到的效应。
    E-value越大，结果越稳健。

    参数:
        rr: 风险比（Risk Ratio）

    返回:
        E-value
    """
    if rr < 1:
        rr = 1.0 / rr

    # E-value公式
    ev = rr + np.sqrt(rr * (rr - 1))
    return float(ev)


def multi_world_robustness(world_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    多世界稳健性评估

    在多个平行世界中运行相同策略，评估结果的稳健性。
    如果不同世界的结果差异较小，说明策略对假设变化不敏感。

    参数:
        world_results: 各世界的结果字典
            {"world_1": {"roi": 1.5, "delta_gtv": 100}, ...}

    返回:
        稳健性评估结果
    """
    rois = [v["roi"] for v in world_results.values() if "roi" in v]
    delta_gtvs = [v["delta_gtv"] for v in world_results.values() if "delta_gtv" in v]

    result = {}

    if rois:
        result["roi_mean"] = float(np.mean(rois))
        result["roi_std"] = float(np.std(rois))
        result["roi_cv"] = float(np.std(rois) / abs(np.mean(rois))) if np.mean(rois) != 0 else float("inf")
        result["roi_ci"] = bootstrap_ci(np.array(rois))

    if delta_gtvs:
        result["delta_gtv_mean"] = float(np.mean(delta_gtvs))
        result["delta_gtv_std"] = float(np.std(delta_gtvs))

    # 稳健性判断：CV < 0.3 为稳健
    if "roi_cv" in result:
        result["robust"] = result["roi_cv"] < 0.3

    return result


def compare_strategies(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    策略对比

    参数:
        results_dict: 各策略的结果
            {"strategy_a": {"roi": 1.5, "delta_gtv": 100, "coverage": 0.3}, ...}

    返回:
        对比表格
    """
    rows = []
    for name, metrics in results_dict.items():
        row = {"strategy": name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("strategy")

    # 排序：按ROI降序
    if "roi" in df.columns:
        df = df.sort_values("roi", ascending=False)

    return df


def smd(treated: np.ndarray, control: np.ndarray) -> float:
    """
    标准化均值差（Standardized Mean Difference）

    SMD < 0.1 表示平衡良好

    参数:
        treated: 处理组特征值
        control: 对照组特征值

    返回:
        SMD值
    """
    mean_t = np.mean(treated)
    mean_c = np.mean(control)
    var_t = np.var(treated, ddof=1)
    var_c = np.var(control, ddof=1)
    pooled_std = np.sqrt((var_t + var_c) / 2.0)

    if pooled_std < 1e-10:
        return 0.0
    return float((mean_t - mean_c) / pooled_std)


def ate_ci(
    treated_outcome: np.ndarray,
    control_outcome: np.ndarray,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    ATE置信区间（基于正态近似）

    参数:
        treated_outcome: 处理组结果
        control_outcome: 对照组结果
        ci: 置信水平

    返回:
        {"ate": float, "se": float, "lower": float, "upper": float}
    """
    ate = np.mean(treated_outcome) - np.mean(control_outcome)
    se = np.sqrt(np.var(treated_outcome) / len(treated_outcome) +
                 np.var(control_outcome) / len(control_outcome))
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    lower = ate - z * se
    upper = ate + z * se

    return {
        "ate": float(ate),
        "se": float(se),
        "lower": float(lower),
        "upper": float(upper),
    }


def policy_value_estimate(
    cate_estimates: np.ndarray,
    subsidy_cost: np.ndarray,
    gtv_per_user: np.ndarray,
    budget: float,
) -> Dict[str, float]:
    """
    策略价值估计

    基于CATE估计值选择补贴目标，计算策略价值。

    参数:
        cate_estimates: CATE估计值数组
        subsidy_cost: 每个用户的补贴成本
        gtv_per_user: 每个用户的GTV
        budget: 总预算

    返回:
        {"roi": float, "delta_gtv": float, "coverage": float, "total_cost": float}
    """
    # 按CATE降序排列
    sorted_idx = np.argsort(cate_estimates)[::-1]

    total_cost = 0.0
    total_delta_gtv = 0.0
    n_treated = 0

    for idx in sorted_idx:
        cost = subsidy_cost[idx]
        if total_cost + cost > budget:
            break
        total_cost += cost
        total_delta_gtv += cate_estimates[idx]
        n_treated += 1

    roi = compute_roi(total_cost, total_delta_gtv)
    coverage = n_treated / len(cate_estimates)

    return {
        "roi": float(roi),
        "delta_gtv": float(total_delta_gtv),
        "coverage": float(coverage),
        "total_cost": float(total_cost),
        "n_treated": int(n_treated),
    }


if __name__ == "__main__":
    # 演示
    np.random.seed(42)

    # Bootstrap CI
    data = np.random.randn(1000) + 2.0
    ci = bootstrap_ci(data)
    print(f"Bootstrap CI for mean: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # ROI
    roi = compute_roi(100, 250)
    print(f"ROI: {roi:.2f}")

    # E-value
    ev = e_value(1.5)
    print(f"E-value for RR=1.5: {ev:.4f}")

    # Multi-world robustness
    worlds = {
        "world_1": {"roi": 1.8, "delta_gtv": 120},
        "world_2": {"roi": 2.1, "delta_gtv": 150},
        "world_3": {"roi": 1.9, "delta_gtv": 130},
    }
    robustness = multi_world_robustness(worlds)
    print(f"Multi-world robustness: {robustness}")

    # Compare strategies
    strategies = {
        "static": {"roi": 1.9, "delta_gtv": 100, "coverage": 0.18},
        "dynamic": {"roi": 2.3, "delta_gtv": 150, "coverage": 0.22},
        "cognitive": {"roi": 2.7, "delta_gtv": 200, "coverage": 0.25},
        "theoretical": {"roi": 3.7, "delta_gtv": 360, "coverage": 0.18},
    }
    comparison = compare_strategies(strategies)
    print(f"\nStrategy comparison:\n{comparison}")
