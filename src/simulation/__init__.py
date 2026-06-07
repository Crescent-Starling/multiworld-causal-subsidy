"""
仿真模块

- cognitive_agent_theory: 认知Agent理论（前景理论+心理账户+有限理性）
- mesa_agent_model: Mesa ABM仿真系统（SubsidyAgent, SubsidyModel, MultiWorldModel）
- network_contagion: 社会网络传染模型（SocialNetwork, SocialContagion）
- llm_agent: LLM驱动的用户决策仿真（LLMSubsidyAgent, LLMAgentSociety）
"""

from .cognitive_agent_theory import (
    prospect_value,
    prospect_discount,
    MentalAccountType,
    DEFAULT_ETA_DICT,
    DEFAULT_DESENS_RATES,
    update_reference_point,
    classify_mental_account,
    bounded_rationality_discount,
    fatigue_update,
    check_account_transition,
    TheoreticalCognitiveAgent,
)

from .mesa_agent_model import (
    SubsidyAgent,
    SubsidyModel,
    MultiWorldModel,
    StrategyType,
    SimulationResult,
)

from .llm_agent import (
    PromptTemplate,
    LLMClient,
    LLMSubsidyAgent,
    LLMAgentSociety,
)

__all__ = [
    # 认知Agent理论
    "prospect_value", "prospect_discount", "MentalAccountType",
    "DEFAULT_ETA_DICT", "DEFAULT_DESENS_RATES",
    "update_reference_point", "classify_mental_account",
    "bounded_rationality_discount", "fatigue_update",
    "check_account_transition", "TheoreticalCognitiveAgent",
    # Mesa ABM
    "SubsidyAgent", "SubsidyModel", "MultiWorldModel", "StrategyType", "SimulationResult",
    # LLM Agent
    "PromptTemplate", "LLMClient", "LLMSubsidyAgent", "LLMAgentSociety",
]
