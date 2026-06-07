"""
AI Subsidy Simulation System
=============================

因果推断与AI仿真驱动的优惠券补贴策略评估系统。

子模块:
- features: 合成数据生成与特征工程
- modeling: 因果推断模型（CausalML, DoWhy, PSM）
- simulation: AI驱动的ABM仿真（Mesa, 认知Agent, 社会网络）
- evaluation: 策略评估指标与鲁棒性分析
"""

from src.features import (
    SyntheticDataConfig,
    generate_user_profiles,
    generate_order_data,
    generate_causal_inference_data,
    generate_all_data,
)

from src.modeling import (
    CausalMLWrapper, CausalMLConfig,
    SubsidyCausalGraph, DoWhyConfig,
    PSMMatcher, PSMConfig,
)

from src.simulation import (
    prospect_value, prospect_discount, MentalAccountType,
    DEFAULT_ETA_DICT, DEFAULT_DESENS_RATES,
    update_reference_point, classify_mental_account,
    bounded_rationality_discount, fatigue_update,
    check_account_transition, TheoreticalCognitiveAgent,
    SubsidyAgent, SubsidyModel, MultiWorldModel, StrategyType, SimulationResult,
    PromptTemplate, LLMClient, LLMSubsidyAgent, LLMAgentSociety,
)

from src.evaluation import (
    bootstrap_ci, compute_roi, compute_delta_gtv, e_value,
    multi_world_robustness, compare_strategies, smd,
    ate_ci, policy_value_estimate,
)

__all__ = [
    # features
    "SyntheticDataConfig",
    "generate_user_profiles", "generate_order_data",
    "generate_causal_inference_data", "generate_all_data",
    # modeling
    "CausalMLWrapper", "CausalMLConfig",
    "SubsidyCausalGraph", "DoWhyConfig",
    "PSMMatcher", "PSMConfig",
    # simulation - cognitive agent
    "prospect_value", "prospect_discount", "MentalAccountType",
    "DEFAULT_ETA_DICT", "DEFAULT_DESENS_RATES",
    "update_reference_point", "classify_mental_account",
    "bounded_rationality_discount", "fatigue_update",
    "check_account_transition", "TheoreticalCognitiveAgent",
    # simulation - mesa
    "SubsidyAgent", "SubsidyModel", "MultiWorldModel", "StrategyType", "SimulationResult",
    # simulation - llm
    "PromptTemplate", "LLMClient", "LLMSubsidyAgent", "LLMAgentSociety",
    # evaluation
    "bootstrap_ci", "compute_roi", "compute_delta_gtv", "e_value",
    "multi_world_robustness", "compare_strategies", "smd",
    "ate_ci", "policy_value_estimate",
]
