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
from typing import Optional, Dict, List
from dataclasses import dataclass, field


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

    # ---- 行为链扩展配置 ----
    enable_behavior_chain: bool = True
    behavior_chain_steps: List[str] = field(default_factory=lambda: [
        "browsed", "clicked", "carted", "paid", "redeemed"
    ])
    # 各步基础转换率（从美团数据校准）：
    #   P(browsed)        = 1.0 （发券即视为有曝光）
    #   P(clicked|browse) ~ 0.55  (美团 CTR 实测约 0.50-0.60)
    #   P(carted|click)   ~ 0.33  (用户行为序列数据 cart/click ≈ 33%)
    #   P(paid|cart)      ~ 0.42  (行为序列 order/cart ≈ 42%)
    #   P(redeemed|paid)  ~ 0.64  (神券订单数据 补贴订单/总订单 ≈ 64% 作为核销代理)
    base_conversion_rates: Dict[str, float] = field(default_factory=lambda: {
        "browsed": 1.0,
        "clicked": 0.55,
        "carted":  0.33,
        "paid":    0.42,
        "redeemed": 0.64,
    })
    # 补贴对各步的增量提升权重（合计 1.0，分配处理效应）
    # 补贴主要影响点击与核销，中间步骤提升较小
    cate_effect_weights: Dict[str, float] = field(default_factory=lambda: {
        "clicked":  0.35,
        "carted":   0.15,
        "paid":     0.15,
        "redeemed": 0.35,
    })
    # 品类列表（与美团数据对齐）
    poi_categories: List[str] = field(default_factory=lambda: [
        "快餐简餐", "中式正餐", "饮品", "健身中心", "超市便利"
    ])


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


# ---------------------------------------------------------------------------
# 行为链数据生成
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid 函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def generate_behavior_chain_data(
    user_profiles: pd.DataFrame,
    config: Optional[SyntheticDataConfig] = None,
    n_records_per_user: int = 3,
) -> pd.DataFrame:
    """
    生成行为链合成数据（漏斗形式）

    每条记录代表一次"发券曝光"对应的完整行为链决策：
        browsed → clicked → carted → paid → redeemed

    漏斗约束：前一步为 False 时，后续步自动为 False。

    参数校准来源：
    - P(clicked)  : 美团用户行为序列 CTR 实测约 0.55
    - P(carted)   : 美团用户行为序列 cart/click ≈ 0.33
    - P(paid)     : 美团用户行为序列 order/cart ≈ 0.42
    - P(redeemed) : 美团神券订单数据 补贴订单占比 ≈ 0.64

    文献：
    - Li & Kannan (2016) 多触点归因 + 因果图
    - Imai et al. (2010) 中介分析框架
    """
    if config is None:
        config = SyntheticDataConfig()

    rng = np.random.RandomState(config.random_state + 10)
    rates = config.base_conversion_rates
    weights = config.cate_effect_weights

    records = []
    n_users = len(user_profiles)

    # 向量化生成，每个用户生成 n_records_per_user 条曝光记录
    total = n_users * n_records_per_user
    user_idx = np.tile(np.arange(n_users), n_records_per_user)
    user_idx = np.sort(user_idx)  # 按用户聚合

    # 用户特征向量
    ps = user_profiles["price_sensitivity"].values[user_idx]          # [0,1]
    income = user_profiles["income_level"].values[user_idx].astype(float)  # 1-5
    mental = user_profiles["mental_account"].values[user_idx]

    # 处理分配（模拟倾向评分驱动）
    propensity = _sigmoid(0.5 * (ps - 0.3) + 0.2 * (income - 3) / 4)
    treatment = rng.binomial(1, propensity)

    # 补贴金额（处理组）
    subsidy_amounts = np.where(
        treatment == 1,
        rng.choice([3, 5, 7, 10, 12, 15], size=total),
        0.0
    ).astype(float)

    # 异质性 CATE（每步的增量效应）
    if config.heterogeneous_effect:
        cate_base = config.true_ate * ps * (income / 3.0)
    else:
        cate_base = np.full(total, config.true_ate)

    # 心理账户增益（windfall_spender 反应更强）
    account_boost = np.where(mental == "windfall_spender", 0.15,
                    np.where(mental == "deal_seeker", 0.10,
                    np.where(mental == "price_sensitive", 0.05, 0.0)))

    # ---- 各步真实 CATE（供多结果因果推断使用）----
    true_cate_clicked  = cate_base * weights["clicked"]
    true_cate_carted   = cate_base * weights["carted"]
    true_cate_paid     = cate_base * weights["paid"]
    true_cate_redeemed = cate_base * weights["redeemed"]

    # ---- 各步决策概率（logistic 形式）----
    # P(step | prev_step=1, treatment, user_features)
    noise_scale = 0.05  # 小量噪声保持可复现性

    def step_prob(base_rate: float, cate_effect: np.ndarray, boost: np.ndarray) -> np.ndarray:
        """将基础转换率 + CATE 效应 + 账户增益转化为概率"""
        logit_base = np.log(base_rate / max(1 - base_rate, 1e-9))
        logit = logit_base + cate_effect * treatment + boost * treatment
        noise = rng.normal(0, noise_scale, size=total)
        return _sigmoid(logit + noise)

    p_clicked  = step_prob(rates["clicked"],  true_cate_clicked * 0.5, account_boost)
    p_carted   = step_prob(rates["carted"],   true_cate_carted * 0.5,  account_boost * 0.8)
    p_paid     = step_prob(rates["paid"],     true_cate_paid * 0.5,    account_boost * 0.6)
    p_redeemed = step_prob(rates["redeemed"], true_cate_redeemed * 0.5, account_boost)

    # ---- 序贯采样（漏斗约束）----
    browsed  = np.ones(total, dtype=int)                                   # 曝光即浏览
    clicked  = (rng.random(total) < p_clicked).astype(int)
    carted   = (rng.random(total) < p_carted).astype(int) * clicked        # 漏斗约束
    paid     = (rng.random(total) < p_paid).astype(int) * carted           # 漏斗约束
    redeemed = (rng.random(total) < p_redeemed).astype(int) * paid         # 漏斗约束

    # ---- GTV 归因（支付时产生）----
    base_gtv = rng.lognormal(3.5, 0.8, size=total)
    gtv = base_gtv * paid + subsidy_amounts * 0.5 * redeemed

    # ---- 品类偏好（从美团校准）----
    categories = rng.choice(config.poi_categories, size=total)

    df = pd.DataFrame({
        "record_id":         np.arange(total),
        "user_id":           user_profiles["user_id"].values[user_idx],
        "treatment":         treatment,
        "subsidy_amount":    subsidy_amounts.round(1),
        "poi_category":      categories,
        # 行为链步骤（二元）
        "browsed":           browsed,
        "clicked":           clicked,
        "carted":            carted,
        "paid":              paid,
        "redeemed":          redeemed,
        # GTV
        "gtv":               gtv.round(2),
        # 真实 CATE（仅供因果评估使用，实际推断时不可见）
        "true_cate_clicked":  true_cate_clicked.round(4),
        "true_cate_carted":   true_cate_carted.round(4),
        "true_cate_paid":     true_cate_paid.round(4),
        "true_cate_redeemed": true_cate_redeemed.round(4),
        # 用户特征（方便直接用于因果推断）
        "price_sensitivity":  ps.round(4),
        "income_level":       income.astype(int),
        "mental_account":     mental,
    })

    return df


def calibrate_from_meituan(
    order_path: str = "data/神券订单数据样例.xlsx",
    behavior_path: str = "data/用户行为序列.xlsx",
) -> Dict[str, float]:
    """
    从美团真实数据校准行为链各步转换率

    返回：{step: base_rate}（可直接赋给 SyntheticDataConfig.base_conversion_rates）

    注：行为序列数据不包含完整漏斗，用如下代理：
    - P(click|browse) : search(浏览代理) → click 比率
    - P(cart|click)   : cart/click 比率
    - P(order|cart)   : order/cart 比率
    - P(redeem|order) : 补贴订单/总订单（作为核销代理）
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        seq = pd.read_excel(behavior_path)
        click_cnt  = seq["行为类型"].str.contains("click|点击", case=False, na=False).sum()
        cart_cnt   = seq["行为类型"].str.contains("cart|加购|加入购物车", case=False, na=False).sum()
        order_cnt  = seq["行为类型"].str.contains("order|下单", case=False, na=False).sum()
        search_cnt = seq["行为类型"].str.contains("search|搜索", case=False, na=False).sum()

        orders = pd.read_excel(order_path)
        redeem_rate = (orders["美补金额"] > 0).sum() / max(len(orders), 1)

        # 条件转换率
        browse_proxy = max(search_cnt, 1)
        p_click  = min(click_cnt / max(browse_proxy + click_cnt, 1), 0.90)
        p_cart   = min(cart_cnt  / max(click_cnt, 1), 0.90)
        p_order  = min(order_cnt / max(cart_cnt, 1), 0.90)

        return {
            "browsed":  1.0,
            "clicked":  round(p_click, 3),
            "carted":   round(p_cart, 3),
            "paid":     round(p_order, 3),
            "redeemed": round(redeem_rate, 3),
        }
    except Exception as e:
        print(f"[calibrate] 无法读取美团数据（{e}），使用内置默认值")
        return {
            "browsed": 1.0,
            "clicked": 0.55,
            "carted":  0.33,
            "paid":    0.42,
            "redeemed": 0.64,
        }


def generate_all_data(config: Optional[SyntheticDataConfig] = None) -> Dict[str, pd.DataFrame]:
    """
    生成所有合成数据

    返回：
    - user_profiles:   用户画像
    - orders:          订单数据
    - causal_data:     因果推断数据（单结果，向后兼容）
    - behavior_chain:  行为链数据（多步骤，当 config.enable_behavior_chain=True）
    """
    if config is None:
        config = SyntheticDataConfig()

    user_profiles = generate_user_profiles(config)
    orders = generate_order_data(user_profiles, config)
    causal_data = generate_causal_inference_data(config)

    result = {
        "user_profiles": user_profiles,
        "orders": orders,
        "causal_data": causal_data,
    }

    if config.enable_behavior_chain:
        behavior_chain = generate_behavior_chain_data(user_profiles, config)
        result["behavior_chain"] = behavior_chain

    return result


if __name__ == "__main__":
    import os

    config = SyntheticDataConfig()
    data = generate_all_data(config)

    os.makedirs("data/synthetic", exist_ok=True)

    for name, df in data.items():
        path = f"data/synthetic/{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols -> {path}")
