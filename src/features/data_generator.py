"""
合成数据生成模块
生成模拟的用户画像、订单、优惠券等数据，用于框架演示和测试

数据来源说明：
- 合成数据基于美团脱敏数据的统计特征生成
- 不包含任何真实用户信息
- 仅用于研究和方法验证
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class SyntheticDataConfig:
    """合成数据配置"""
    n_users: int = 10000
    n_orders: int = 50000
    n_coupons: int = 100000
    n_features: int = 20
    treatment_ratio: float = 0.3
    true_ate: float = 2.0
    heterogeneous_effect: bool = True
    random_state: int = 42


def generate_user_profiles(config: Optional[SyntheticDataConfig] = None) -> pd.DataFrame:
    """
    生成用户画像数据

    特征包括：
    - price_sensitivity: 价格敏感度 [0, 1]
    - income_level: 收入水平 [1, 5]
    - city_tier: 城市线级 [1, 5]
    - age_group: 年龄段 [18, 60]
    - consumption_frequency: 消费频次
    - mental_account_type: 心理账户类型
    """
    if config is None:
        config = SyntheticDataConfig()

    rng = np.random.RandomState(config.random_state)

    n = config.n_users

    # 基础特征
    price_sensitivity = rng.beta(2, 5, n)  # 多数用户价格敏感度偏低
    income_level = rng.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.2, 0.35, 0.25, 0.1])
    city_tier = rng.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    age = rng.normal(32, 8, n).clip(18, 60).astype(int)
    consumption_freq = rng.poisson(5, n).clip(0, 30)  # 月均消费次数

    # 心理账户分类
    mental_account = []
    for i in range(n):
        if price_sensitivity[i] > 0.5 and income_level[i] <= 2:
            mental_account.append("windfall_spender")
        elif price_sensitivity[i] > 0.3:
            mental_account.append("price_sensitive")
        elif income_level[i] >= 4:
            mental_account.append("routine_income")
        else:
            mental_account.append("deal_seeker")

    df = pd.DataFrame({
        "user_id": [f"U{i:06d}" for i in range(n)],
        "price_sensitivity": price_sensitivity.round(4),
        "income_level": income_level,
        "city_tier": city_tier,
        "age": age,
        "consumption_frequency": consumption_freq,
        "mental_account": mental_account,
    })

    # 添加额外特征
    for j in range(config.n_features - 6):
        df[f"feature_{j}"] = rng.randn(n).round(4)

    return df


def generate_order_data(
    user_profiles: pd.DataFrame,
    config: Optional[SyntheticDataConfig] = None
) -> pd.DataFrame:
    """
    生成订单数据

    每个用户生成多条订单记录
    """
    if config is None:
        config = SyntheticDataConfig()

    rng = np.random.RandomState(config.random_state + 1)

    records = []
    for _, user in user_profiles.iterrows():
        n_orders = rng.poisson(user["consumption_frequency"] // 2) + 1
        for _ in range(min(n_orders, 10)):
            # 基础消费金额
            base_amount = rng.lognormal(3.5, 0.8)
            # 是否获得补贴
            treated = rng.random() < config.treatment_ratio
            # 处理效应（异质性）
            if config.heterogeneous_effect:
                cate = config.true_ate * user["price_sensitivity"]
            else:
                cate = config.true_ate

            # 消费金额
            gtv = base_amount + (cate if treated else 0) + rng.randn() * 5
            gtv = max(gtv, 0)

            records.append({
                "user_id": user["user_id"],
                "order_id": f"O{len(records):08d}",
                "treatment": int(treated),
                "subsidy_amount": rng.choice([5, 10, 15, 20]) if treated else 0,
                "gtv": round(gtv, 2),
                "redeemed": int(gtv > base_amount * 0.8 and treated and rng.random() < user["price_sensitivity"]),
                "poi_category": rng.choice(["美发", "餐饮", "电影", "KTV", "零售"]),
            })

    return pd.DataFrame(records)


def generate_causal_inference_data(
    config: Optional[SyntheticDataConfig] = None
) -> pd.DataFrame:
    """
    生成因果推断专用数据

    包含明确的混杂因子、处理变量、结果变量
    """
    if config is None:
        config = SyntheticDataConfig()

    rng = np.random.RandomState(config.random_state)

    n = config.n_users

    # 混杂因子
    Z1 = rng.randn(n)  # 用户偏好
    Z2 = rng.binomial(1, 0.5, n)  # 性别
    Z3 = rng.randn(n)  # 消费能力
    Z4 = rng.choice([1, 2, 3, 4, 5], n)  # 城市线级

    # 处理分配（与混杂因子相关，模拟选择偏差）
    propensity = 1 / (1 + np.exp(-0.5 * Z1 - 0.3 * Z3 + 0.2 * (Z4 - 3)))
    treatment = rng.binomial(1, propensity)

    # 异质性处理效应
    if config.heterogeneous_effect:
        cate = config.true_ate * (1 + 0.5 * Z1 + 0.3 * Z3)
    else:
        cate = config.true_ate

    # 结果变量
    outcome = Z1 + 0.5 * Z2 + Z3 + cate * treatment + rng.randn(n)

    feature_cols = [f"Z{i+1}" for i in range(4)]
    df = pd.DataFrame({
        "Z1": Z1, "Z2": Z2, "Z3": Z3, "Z4": Z4,
        "treatment": treatment,
        "outcome": outcome,
    })

    # 添加额外特征
    for j in range(config.n_features - 4):
        df[f"feature_{j}"] = rng.randn(n)

    df["true_cate"] = cate

    return df


def generate_all_data(config: Optional[SyntheticDataConfig] = None) -> Dict[str, pd.DataFrame]:
    """
    生成所有合成数据

    返回：
    - user_profiles: 用户画像
    - orders: 订单数据
    - causal_data: 因果推断数据
    """
    if config is None:
        config = SyntheticDataConfig()

    user_profiles = generate_user_profiles(config)
    orders = generate_order_data(user_profiles, config)
    causal_data = generate_causal_inference_data(config)

    return {
        "user_profiles": user_profiles,
        "orders": orders,
        "causal_data": causal_data,
    }


if __name__ == "__main__":
    import os

    config = SyntheticDataConfig()
    data = generate_all_data(config)

    os.makedirs("data/synthetic", exist_ok=True)

    for name, df in data.items():
        path = f"data/synthetic/{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols -> {path}")
