"""
认知Agent理论模块
================
基于行为经济学的认知决策Agent，整合前景理论、心理账户、有限理性和疲劳脱敏等理论。

参考文献:
- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk.
  Econometrica, 47(2), 263-291.
- Thaler, R. H. (1985). Mental Accounting and Consumer Choice. Marketing Science, 4(3), 199-214.
- Simon, H. A. (1955). A Behavioral Model of Rational Choice. The Quarterly Journal of Economics,
  69(1), 99-118.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity Analysis in Observational Research:
  Introducing the E-Value. Annals of Internal Medicine, 167(4), 268-274.
"""

import numpy as np
from enum import Enum
from typing import Dict, Optional, Any


# ============================================================================
# 1. 前景理论价值函数 (Kahneman & Tversky, 1979)
# ============================================================================

def prospect_value(x: float, alpha: float = 0.88, lambda_: float = 2.25) -> float:
    """
    前景理论价值函数 V(x)。

    对收益（x >= 0）呈凹函数，对损失（x < 0）呈凸函数，损失厌恶系数 lambda_ > 1。

    V(x) = x^alpha           (x >= 0)
    V(x) = -lambda_ * (-x)^alpha  (x < 0)

    参数:
        x: 收益或损失值
        alpha: 敏感度递减指数，典型值 0.88 (Tversky & Kahneman, 1992)
        lambda_: 损失厌恶系数，典型值 2.25

    返回:
        主观价值
    """
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_ * ((-x) ** alpha)


def prospect_discount(
    subsidy: float,
    reference: float,
    alpha: float = 0.88,
    lambda_: float = 2.25,
) -> float:
    """
    补贴相对于参考点的前景理论折扣价值。

    当补贴高于参考点时，边际价值递减（敏感性递减）；
    当补贴低于参考点时，感知损失被放大（损失厌恶）。

    参数:
        subsidy: 补贴金额
        reference: 参考点（用户对"正常补贴"的心理预期）
        alpha: 敏感度递减指数
        lambda_: 损失厌恶系数

    返回:
        补贴的主观折扣价值
    """
    delta = subsidy - reference
    return prospect_value(delta, alpha=alpha, lambda_=lambda_)


# ============================================================================
# 2. 心理账户 (Thaler, 1985)
# ============================================================================

class MentalAccountType(Enum):
    """
    心理账户类型枚举。

    - WINDFALL_SPENDER: 横财型——意外获得的补贴，消费倾向高，参考点更新慢
    - PRICE_SENSITIVE: 价格敏感型——对价格高度敏感，参考点更新快
    - ROUTINE_INCOME: 常规收入型——将补贴视为收入的一部分，消费倾向中等
    - DEAL_SEEKER: 捡漏型——追求优惠，参考点更新中等
    """
    WINDFALL_SPENDER = "windfall_spender"
    PRICE_SENSITIVE = "price_sensitive"
    ROUTINE_INCOME = "routine_income"
    DEAL_SEEKER = "deal_seeker"


# 心理账户类型对应的参考点更新速率 eta
# windfall: 意外之财，参考点更新慢 (η=0.10)
# income: 常规收入，参考点更新中等 (η=0.35)
# deal_seeker: 捡漏型，参考点更新中等偏低 (η=0.20)
# price_sensitive: 价格敏感，参考点更新快 (η=0.50)
DEFAULT_ETA_DICT: Dict[MentalAccountType, float] = {
    MentalAccountType.WINDFALL_SPENDER: 0.10,
    MentalAccountType.ROUTINE_INCOME: 0.35,
    MentalAccountType.DEAL_SEEKER: 0.20,
    MentalAccountType.PRICE_SENSITIVE: 0.50,
}


def update_reference_point(
    ref: float,
    outcome: float,
    account_type: MentalAccountType,
    eta_dict: Optional[Dict[MentalAccountType, float]] = None,
) -> float:
    """
    参考点自适应更新（Thaler, 1985）。

    ref_new = ref + eta * (outcome - ref)

    横财型账户更新慢（η小），价格敏感型更新快（η大）。

    参数:
        ref: 当前参考点
        outcome: 实际获得的补贴/收益
        account_type: 心理账户类型
        eta_dict: 各账户类型的更新速率字典，默认使用 DEFAULT_ETA_DICT

    返回:
        更新后的参考点
    """
    if eta_dict is None:
        eta_dict = DEFAULT_ETA_DICT
    eta = eta_dict[account_type]
    return ref + eta * (outcome - ref)


def classify_mental_account(row: Dict[str, Any]) -> MentalAccountType:
    """
    基于行为特征分类心理账户类型。

    分类规则:
    - 使用频率低 + 单次补贴高 → WINDFALL_SPENDER（横财型）
    - 价格弹性高 + 搜索比价 → PRICE_SENSITIVE（价格敏感型）
    - 使用频率高 + 补贴占比低 → ROUTINE_INCOME（常规收入型）
    - 优惠券使用率高 + 多平台比价 → DEAL_SEEKER（捡漏型）

    参数:
        row: 包含行为特征的字典，需包含以下键:
            - usage_freq: 使用频率 (0-1)
            - avg_subsidy_ratio: 平均补贴占比 (0-1)
            - price_elasticity: 价格弹性 (0-1)
            - coupon_usage_rate: 优惠券使用率 (0-1)
            - search_compare_rate: 搜索比价率 (0-1)

    返回:
        心理账户类型
    """
    usage_freq = row.get("usage_freq", 0.5)
    avg_subsidy_ratio = row.get("avg_subsidy_ratio", 0.5)
    price_elasticity = row.get("price_elasticity", 0.5)
    coupon_usage_rate = row.get("coupon_usage_rate", 0.5)
    search_compare_rate = row.get("search_compare_rate", 0.5)

    # 评分法分类
    scores = {
        MentalAccountType.WINDFALL_SPENDER: (
            (1.0 - usage_freq) * 0.4           # 使用频率低
            + avg_subsidy_ratio * 0.3           # 补贴占比高
            + (1.0 - coupon_usage_rate) * 0.3   # 优惠券使用率低
        ),
        MentalAccountType.PRICE_SENSITIVE: (
            price_elasticity * 0.4              # 价格弹性高
            + search_compare_rate * 0.35        # 搜索比价多
            + (1.0 - avg_subsidy_ratio) * 0.25  # 补贴占比低（对价格敏感）
        ),
        MentalAccountType.ROUTINE_INCOME: (
            usage_freq * 0.45                   # 使用频率高
            + (1.0 - avg_subsidy_ratio) * 0.35  # 补贴占比低
            + (1.0 - price_elasticity) * 0.2    # 价格弹性低
        ),
        MentalAccountType.DEAL_SEEKER: (
            coupon_usage_rate * 0.4             # 优惠券使用率高
            + search_compare_rate * 0.35        # 搜索比价多
            + avg_subsidy_ratio * 0.25          # 补贴占比高
        ),
    }

    return max(scores, key=scores.get)


# ============================================================================
# 3. 有限理性 (Simon, 1955)
# ============================================================================

def bounded_rationality_discount(
    cognitive_load: float,
    max_load: float = 10.0,
) -> float:
    """
    有限理性折扣函数（Simon, 1955）。

    认知负荷越高，决策质量越低，折扣越大。
    使用 Sigmoid 形式的衰减，模拟认知资源有限时的决策质量下降。

    discount = 1 / (1 + exp(k * (cognitive_load - max_load/2)))

    参数:
        cognitive_load: 当前认知负荷 (0 ~ max_load)
        max_load: 认知负荷上限

    返回:
        折扣因子 (0, 1)，值越低表示决策质量越差
    """
    # Sigmoid 衰减，k 控制衰减速率
    k = 1.5
    mid = max_load / 2.0
    return 1.0 / (1.0 + np.exp(k * (cognitive_load - mid)))


# ============================================================================
# 4. 疲劳脱敏
# ============================================================================

# 各心理账户类型的脱敏速率
DEFAULT_DESENS_RATES: Dict[MentalAccountType, float] = {
    MentalAccountType.WINDFALL_SPENDER: 0.15,   # 横财型脱敏快
    MentalAccountType.PRICE_SENSITIVE: 0.08,     # 价格敏感型脱敏慢
    MentalAccountType.ROUTINE_INCOME: 0.12,      # 常规收入型脱敏中等
    MentalAccountType.DEAL_SEEKER: 0.10,         # 捡漏型脱敏中等
}

# 账户类型迁移映射: windfall → income（横财效应衰减）
ACCOUNT_TRANSITION: Dict[MentalAccountType, MentalAccountType] = {
    MentalAccountType.WINDFALL_SPENDER: MentalAccountType.ROUTINE_INCOME,
}


def fatigue_update(
    fatigue: float,
    was_subsidized: bool,
    account_type: MentalAccountType,
    desens_rates: Optional[Dict[MentalAccountType, float]] = None,
) -> float:
    """
    非线性疲劳脱敏更新。

    当用户持续获得补贴时，对补贴的敏感度下降（脱敏），疲劳值增加。
    使用对数增长形式，避免疲劳值无限增长。

    fatigue_new = fatigue + desens_rate * log(1 + fatigue)  (was_subsidized)
    fatigue_new = fatigue * decay                            (!was_subsidized)

    当疲劳值超过阈值时，windfall 类型迁移为 income 类型（横财效应衰减）。

    参数:
        fatigue: 当前疲劳值 (>= 0)
        was_subsidized: 是否获得补贴
        account_type: 心理账户类型
        desens_rates: 各账户类型的脱敏速率

    返回:
        更新后的疲劳值
    """
    if desens_rates is None:
        desens_rates = DEFAULT_DESENS_RATES

    rate = desens_rates[account_type]

    if was_subsidized:
        # 对数增长：疲劳积累逐渐放缓
        fatigue_new = fatigue + rate * np.log1p(fatigue)
    else:
        # 无补贴时疲劳自然衰减
        decay = 0.85
        fatigue_new = fatigue * decay

    return max(fatigue_new, 0.0)


def check_account_transition(
    fatigue: float,
    account_type: MentalAccountType,
    threshold: float = 3.0,
) -> MentalAccountType:
    """
    检查是否触发心理账户类型迁移。

    当疲劳值超过阈值时，windfall_spender 迁移为 routine_income，
    模拟"横财效应"随时间衰减。

    参数:
        fatigue: 当前疲劳值
        account_type: 当前心理账户类型
        threshold: 迁移阈值

    返回:
        可能迁移后的心理账户类型
    """
    if fatigue >= threshold and account_type in ACCOUNT_TRANSITION:
        return ACCOUNT_TRANSITION[account_type]
    return account_type


# ============================================================================
# 5. TheoreticalCognitiveAgent 类
# ============================================================================

class TheoreticalCognitiveAgent:
    """
    理论驱动的认知决策Agent。

    整合前景理论、心理账户、有限理性和疲劳脱敏，模拟用户在补贴场景下的
    核销决策过程。

    决策流程:
    1. 计算补贴相对于参考点的前景价值
    2. 应用有限理性折扣
    3. 应用疲劳脱敏折扣
    4. 与随机阈值比较，决定是否核销

    参数:
        agent_id: Agent唯一标识
        reference_point: 初始参考点（默认0）
        fatigue: 初始疲劳值（默认0）
        cognitive_load: 认知负荷（默认3.0）
        mental_account: 心理账户类型（默认WINDFALL_SPENDER）
        alpha: 前景理论敏感度指数
        lambda_: 损失厌恶系数
        eta_dict: 参考点更新速率字典
        desens_rates: 脱敏速率字典
        decision_threshold: 决策阈值（默认0.3）
        rng: 随机数生成器
    """

    def __init__(
        self,
        agent_id: str = "agent_0",
        reference_point: float = 0.0,
        fatigue: float = 0.0,
        cognitive_load: float = 3.0,
        mental_account: MentalAccountType = MentalAccountType.WINDFALL_SPENDER,
        alpha: float = 0.88,
        lambda_: float = 2.25,
        eta_dict: Optional[Dict[MentalAccountType, float]] = None,
        desens_rates: Optional[Dict[MentalAccountType, float]] = None,
        decision_threshold: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.agent_id = agent_id
        self.reference_point = reference_point
        self.fatigue = fatigue
        self.cognitive_load = cognitive_load
        self.mental_account = mental_account
        self.alpha = alpha
        self.lambda_ = lambda_
        self.eta_dict = eta_dict if eta_dict is not None else DEFAULT_ETA_DICT.copy()
        self.desens_rates = desens_rates if desens_rates is not None else DEFAULT_DESENS_RATES.copy()
        self.decision_threshold = decision_threshold
        self.rng = rng if rng is not None else np.random.default_rng()

        # 记录历史决策
        self.history: list = []

    def decide(self, subsidy_amount: float, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        决定是否核销补贴。

        决策流程:
        1. 计算补贴的前景理论折扣价值（相对于参考点）
        2. 归一化为激活值
        3. 应用有限理性折扣
        4. 应用疲劳脱敏折扣
        5. 与决策阈值 + 随机噪声比较

        参数:
            subsidy_amount: 补贴金额
            context: 额外上下文信息（可选），可包含:
                - base_gtv: 基础GTV，用于归一化
                - noise_scale: 随机噪声强度

        返回:
            True 表示核销，False 表示不核销
        """
        context = context or {}
        base_gtv = context.get("base_gtv", subsidy_amount + 1.0)
        noise_scale = context.get("noise_scale", 0.1)

        # 步骤1: 前景理论折扣价值
        pv = prospect_discount(
            subsidy=subsidy_amount,
            reference=self.reference_point,
            alpha=self.alpha,
            lambda_=self.lambda_,
        )

        # 步骤2: 归一化激活值（将前景价值映射到 [0, 1]）
        # 使用 sigmoid 归一化
        activation = 1.0 / (1.0 + np.exp(-pv / max(base_gtv, 1.0)))

        # 步骤3: 有限理性折扣
        br_discount = bounded_rationality_discount(self.cognitive_load)

        # 步骤4: 疲劳脱敏折扣
        # 疲劳值越高，折扣越大；使用指数衰减
        fatigue_discount = np.exp(-0.3 * self.fatigue)

        # 步骤5: 综合激活值
        effective_activation = activation * br_discount * fatigue_discount

        # 加入随机噪声（模拟不可观测因素）
        noise = self.rng.normal(0, noise_scale)
        effective_threshold = self.decision_threshold + noise

        # 决策
        redeemed = effective_activation > effective_threshold

        return bool(redeemed)

    def update_state(self, was_subsidized: bool, redeemed: bool) -> None:
        """
        更新Agent内部状态。

        在每次决策后调用，更新：
        - 疲劳值（基于是否获得补贴）
        - 参考点（基于实际获得的补贴）
        - 心理账户类型（疲劳超过阈值时可能迁移）

        参数:
            was_subsidized: 是否获得补贴
            redeemed: 是否核销
        """
        # 更新疲劳值
        self.fatigue = fatigue_update(
            fatigue=self.fatigue,
            was_subsidized=was_subsidized,
            account_type=self.mental_account,
            desens_rates=self.desens_rates,
        )

        # 更新参考点（仅在有补贴时更新）
        if was_subsidized:
            # 使用补贴金额作为 outcome 的近似
            # 实际场景中应传入具体补贴金额，此处用参考点偏移模拟
            outcome = self.reference_point * 0.8 + 0.2 * (1.0 if redeemed else 0.0)
            self.reference_point = update_reference_point(
                ref=self.reference_point,
                outcome=outcome,
                account_type=self.mental_account,
                eta_dict=self.eta_dict,
            )

        # 检查心理账户类型迁移
        self.mental_account = check_account_transition(
            fatigue=self.fatigue,
            account_type=self.mental_account,
        )

        # 记录历史
        self.history.append({
            "was_subsidized": was_subsidized,
            "redeemed": redeemed,
            "fatigue": self.fatigue,
            "reference_point": self.reference_point,
            "mental_account": self.mental_account.value,
        })

    def reset(self) -> None:
        """重置Agent状态到初始值。"""
        self.reference_point = 0.0
        self.fatigue = 0.0
        self.mental_account = MentalAccountType.WINDFALL_SPENDER
        self.history = []

    def __repr__(self) -> str:
        return (
            f"TheoreticalCognitiveAgent(id={self.agent_id}, "
            f"ref={self.reference_point:.3f}, "
            f"fatigue={self.fatigue:.3f}, "
            f"account={self.mental_account.value}, "
            f"load={self.cognitive_load:.1f})"
        )
