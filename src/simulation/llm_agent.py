"""
LLM Agent仿真模块
================
使用大语言模型（LLM）模拟消费者补贴决策。

支持：
1. OpenAI API（GPT-4等）
2. Anthropic API（Claude等）
3. Mock模式（无需API Key，使用规则回退）

参考文献:
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra
  of Human Behavior. UIST 2023.
- Horton, J. J. (2023). Homo Silicus: Using LLMs to Simulate Economic
  Experiments. NBER Working Paper.
"""

from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List


# ===========================================================================
# 心理账户类型对应的提示词模板
# ===========================================================================

class PromptTemplate:
    """LLM提示词模板"""

    SYSTEM_BASE = """你是一位消费者行为模拟Agent。你需要根据给定的场景信息，模拟一个真实消费者面对优惠券补贴时的决策过程。

请严格按照以下JSON格式输出你的决策：
```json
{{
    "redeemed": true/false,
    "reasoning": "你的决策推理过程",
    "confidence": 0.0-1.0
}}
```

决策因素：
1. 补贴金额相对于你的消费水平是否有吸引力
2. 使用门槛是否合理
3. 你当前的消费需求
4. 过去的优惠券使用体验
5. 你的消费习惯和偏好
"""

    TEMPLATES = {
        "windfall_spender": SYSTEM_BASE + """
你的消费画像：
- 你是一个"横财型"消费者，意外获得的补贴对你吸引力很大
- 你倾向于把意外收入花掉，而不是储蓄
- 你对补贴的使用门槛不太敏感，只要有补贴就倾向于使用
- 你的参考点较低，容易满足
""",

        "price_sensitive": SYSTEM_BASE + """
你的消费画像：
- 你是一个"价格敏感型"消费者，对价格变化非常敏感
- 你会仔细比较不同优惠方案，选择最划算的
- 你对补贴的使用门槛很敏感，门槛过高会放弃
- 你经常搜索和比较价格信息
""",

        "routine_income": SYSTEM_BASE + """
你的消费画像：
- 你是一个"常规收入型"消费者，将补贴视为收入的一部分
- 你对补贴持谨慎态度，不会因为补贴而冲动消费
- 你更关注补贴是否能带来实际价值
- 你的消费决策较为理性，不受情绪驱动
""",

        "deal_seeker": SYSTEM_BASE + """
你的消费画像：
- 你是一个"捡漏型"消费者，热衷于寻找优惠
- 你享受"薅羊毛"的过程，不仅是为了省钱
- 你对补贴信息很关注，但也要看优惠力度是否足够
- 你经常在不同平台之间比较优惠
""",
    }

    USER_TEMPLATE = """
当前场景：
- 补贴类型：优惠券
- 补贴金额：{subsidy_amount}元
- 使用门槛：最低消费{threshold}元
- 你的消费频率：{consumption_freq}次/月
- 过去7天是否收到过补贴：{recent_subsidy}
- 当前疲劳程度：{fatigue_level}

请做出你的决策：
"""

    @classmethod
    def get_system_prompt(cls, mental_account: str) -> str:
        """获取系统提示词"""
        return cls.TEMPLATES.get(mental_account, cls.TEMPLATES["routine_income"])

    @classmethod
    def get_user_prompt(cls, **kwargs) -> str:
        """获取用户提示词"""
        defaults = {
            "subsidy_amount": 10,
            "threshold": 30,
            "consumption_freq": 5,
            "recent_subsidy": "是",
            "fatigue_level": "低",
        }
        defaults.update(kwargs)
        return cls.USER_TEMPLATE.format(**defaults)


# ===========================================================================
# LLM API 客户端
# ===========================================================================

class LLMClient:
    """LLM API客户端，支持多种后端"""

    def __init__(self, backend: str = "mock", api_key: Optional[str] = None):
        self.backend = backend
        self.api_key = api_key
        self.client = None

        if backend == "openai" and api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                print("Warning: openai package not installed, falling back to mock")
                self.backend = "mock"

        elif backend == "anthropic" and api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
            except ImportError:
                print("Warning: anthropic package not installed, falling back to mock")
                self.backend = "mock"

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM"""
        if self.backend == "openai" and self.client:
            return self._call_openai(system_prompt, user_prompt)
        elif self.backend == "anthropic" and self.client:
            return self._call_anthropic(system_prompt, user_prompt)
        else:
            return self._call_mock(system_prompt, user_prompt)

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """调用OpenAI API"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """调用Anthropic API"""
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.content[0].text

    def _call_mock(self, system_prompt: str, user_prompt: str) -> str:
        """
        Mock模式：基于规则生成决策

        当LLM API不可用时，回退到规则Agent
        """
        # 从用户提示词中提取补贴金额
        subsidy_match = re.search(r"补贴金额：(\d+)元", user_prompt)
        subsidy = int(subsidy_match.group(1)) if subsidy_match else 10

        # 从系统提示词中推断心理账户类型
        if "横财型" in system_prompt:
            base_prob = 0.7
        elif "价格敏感" in system_prompt:
            base_prob = 0.6
        elif "捡漏" in system_prompt:
            base_prob = 0.65
        else:
            base_prob = 0.45

        # 疲劳调整
        if "高" in user_prompt:
            base_prob *= 0.7
        elif "中" in user_prompt:
            base_prob *= 0.85

        # 补贴金额调整
        if subsidy >= 15:
            base_prob += 0.1
        elif subsidy <= 5:
            base_prob -= 0.1

        # 随机决策
        redeemed = np.random.random() < base_prob
        confidence = np.random.uniform(0.6, 0.95)

        reasoning = ""
        if redeemed:
            reasoning = f"该{subsidy}元补贴对我有吸引力，决定使用。"
        else:
            reasoning = f"该{subsidy}元补贴吸引力不足，暂不使用。"

        return json.dumps({
            "redeemed": redeemed,
            "reasoning": reasoning,
            "confidence": round(confidence, 2),
        }, ensure_ascii=False)


# ===========================================================================
# LLMSubsidyAgent
# ===========================================================================

class LLMSubsidyAgent:
    """
    LLM驱动的补贴决策Agent

    使用LLM模拟消费者面对补贴时的决策过程，
    包含完整的推理链（Chain-of-Thought）。
    """

    def __init__(
        self,
        agent_id: str,
        mental_account: str = "routine_income",
        price_sensitivity: float = 0.5,
        income_level: int = 3,
        consumption_freq: int = 5,
        llm_client: Optional[LLMClient] = None,
    ):
        self.agent_id = agent_id
        self.mental_account = mental_account
        self.price_sensitivity = price_sensitivity
        self.income_level = income_level
        self.consumption_freq = consumption_freq
        self.llm = llm_client or LLMClient(backend="mock")

        # 内部状态
        self.fatigue = 0.0
        self.reference_point = 0.0
        self.n_subsidized = 0
        self.n_redeemed = 0

        # 决策轨迹
        self.trajectory: List[Dict[str, Any]] = []

    def decide(
        self,
        subsidy_amount: float,
        threshold: float = 30.0,
    ) -> Dict[str, Any]:
        """
        使用LLM决策

        参数:
            subsidy_amount: 补贴金额
            threshold: 使用门槛

        返回:
            {"redeemed": bool, "reasoning": str, "confidence": float}
        """
        # 构建提示词
        system_prompt = PromptTemplate.get_system_prompt(self.mental_account)
        user_prompt = PromptTemplate.get_user_prompt(
            subsidy_amount=int(subsidy_amount),
            threshold=int(threshold),
            consumption_freq=self.consumption_freq,
            recent_subsidy="是" if self.n_subsidized > 0 else "否",
            fatigue_level=self._fatigue_label(),
        )

        # 调用LLM
        response_text = self.llm.call(system_prompt, user_prompt)

        # 解析结果
        result = self._parse_response(response_text)

        # 更新状态
        self.n_subsidized += 1
        if result["redeemed"]:
            self.n_redeemed += 1
        self.fatigue = self.fatigue * 0.85 + 0.15 * (1.0 if self.n_subsidized > 3 else 0.0)

        # 记录轨迹
        self.trajectory.append({
            "agent_id": self.agent_id,
            "subsidy_amount": subsidy_amount,
            "threshold": threshold,
            "redeemed": result["redeemed"],
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0.0),
            "fatigue": self.fatigue,
        })

        return result

    def _fatigue_label(self) -> str:
        """疲劳程度标签"""
        if self.fatigue < 0.3:
            return "低"
        elif self.fatigue < 0.7:
            return "中"
        else:
            return "高"

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试从响应中提取JSON
            json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "redeemed": bool(data.get("redeemed", False)),
                    "reasoning": data.get("reasoning", ""),
                    "confidence": float(data.get("confidence", 0.5)),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # 回退：基于文本判断
        redeemed = "true" in text.lower() or "使用" in text or "核销" in text
        return {
            "redeemed": redeemed,
            "reasoning": text[:200],
            "confidence": 0.5,
        }


# ===========================================================================
# LLMAgentSociety
# ===========================================================================

class LLMAgentSociety:
    """
    LLM Agent社会仿真

    管理一组LLMSubsidyAgent，执行多轮仿真。
    """

    def __init__(
        self,
        n_agents: int = 10,
        use_mock: bool = True,
        backend: str = "mock",
        api_key: Optional[str] = None,
        seed: int = 42,
    ):
        np.random.seed(seed)

        self.n_agents = n_agents
        self.use_mock = use_mock
        self.results: List[Dict[str, Any]] = []

        # 创建LLM客户端
        if not use_mock and api_key:
            self.llm = LLMClient(backend=backend, api_key=api_key)
        else:
            self.llm = LLMClient(backend="mock")

        # 创建Agent
        mental_accounts = ["windfall_spender", "price_sensitive", "routine_income", "deal_seeker"]
        self.agents: List[LLMSubsidyAgent] = []
        for i in range(n_agents):
            ma = mental_accounts[i % len(mental_accounts)]
            ps = np.random.uniform(0.2, 0.8)
            income = np.random.choice([1, 2, 3, 4, 5])
            agent = LLMSubsidyAgent(
                agent_id=f"llm_agent_{i}",
                mental_account=ma,
                price_sensitivity=ps,
                income_level=income,
                consumption_freq=np.random.poisson(5),
                llm_client=self.llm,
            )
            self.agents.append(agent)

    def run_round(self, subsidy_amount: float = 10.0, threshold: float = 30.0) -> Dict[str, Any]:
        """执行一轮仿真"""
        n_redeemed = 0
        round_trajectories = []

        for agent in self.agents:
            result = agent.decide(subsidy_amount, threshold)
            if result["redeemed"]:
                n_redeemed += 1
            round_trajectories.append(result)

        redemption_rate = n_redeemed / self.n_agents

        round_result = {
            "round": len(self.results) + 1,
            "subsidy_amount": subsidy_amount,
            "threshold": threshold,
            "n_redeemed": n_redeemed,
            "redemption_rate": redemption_rate,
            "trajectories": round_trajectories,
        }

        self.results.append(round_result)
        return round_result

    def run_simulation(self, n_rounds: int = 8) -> List[Dict[str, Any]]:
        """执行多轮仿真"""
        # 每轮递增补贴门槛（模拟策略调整）
        for r in range(n_rounds):
            subsidy = 10.0 + r * 2.0  # 逐轮增加补贴
            threshold = 30.0 + r * 5.0  # 逐轮增加门槛
            result = self.run_round(subsidy, threshold)

            mode = "MOCK" if self.use_mock else "LLM"
            print(f"  Round {result['round']} [{mode}]: "
                  f"subsidy=¥{subsidy:.0f}, threshold=¥{threshold:.0f}, "
                  f"redemption_rate={result['redemption_rate']:.2%}")

        return self.results

    def get_trajectory_df(self) -> pd.DataFrame:
        """获取所有Agent的决策轨迹"""
        rows = []
        for agent in self.agents:
            for entry in agent.trajectory:
                rows.append(entry)
        return pd.DataFrame(rows)

    def get_summary(self) -> pd.DataFrame:
        """获取仿真结果汇总"""
        rows = []
        for r in self.results:
            rows.append({
                "round": r["round"],
                "subsidy_amount": r["subsidy_amount"],
                "threshold": r["threshold"],
                "redemption_rate": r["redemption_rate"],
                "n_redeemed": r["n_redeemed"],
            })
        return pd.DataFrame(rows)


# ===========================================================================
# 运行入口
# ===========================================================================

def run_llm_simulation(
    n_agents: int = 10,
    n_rounds: int = 8,
    use_mock: bool = True,
) -> Dict[str, Any]:
    """运行LLM Agent仿真"""
    print("=" * 60)
    print("LLM Agent Simulation")
    mode = "MOCK" if use_mock else "LLM API"
    print(f"Mode: {mode}")
    print("=" * 60)

    society = LLMAgentSociety(n_agents=n_agents, use_mock=use_mock)
    results = society.run_simulation(n_rounds=n_rounds)

    # 汇总
    summary = society.get_summary()
    print(f"\n--- Summary ---")
    print(summary.to_string(index=False))

    # 轨迹
    trajectory = society.get_trajectory_df()
    print(f"\n--- Trajectory (first 10 rows) ---")
    print(trajectory.head(10).to_string(index=False))

    return {
        "results": results,
        "summary": summary,
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    run_llm_simulation(n_agents=10, n_rounds=5, use_mock=True)
