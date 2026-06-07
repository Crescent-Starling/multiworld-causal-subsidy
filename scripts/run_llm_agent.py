"""
LLM Agent 补贴仿真运行脚本

使用合成数据测试LLM Agent仿真模块。
- 如果有API Key（OPENAI_API_KEY / ANTHROPIC_API_KEY），使用真实LLM
- 如果没有API Key，自动回退到Mock模式（随机决策但保留推理格式）

用法：
    python scripts/run_llm_agent.py

环境变量：
    OPENAI_API_KEY: OpenAI API密钥
    ANTHROPIC_API_KEY: Anthropic API密钥
    LLM_MODEL: 指定模型名称（默认 gpt-4o）

参考文献：
- Park et al. (2023): Generative Agents
- AgentSociety (Tsinghua): LLM-based Agent Simulation Framework
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.llm_agent import (
    PromptTemplate,
    LLMSubsidyAgent,
    LLMAgentSociety,
    AgentProfile,
    SubsidyInfo,
    AgentDecision,
    MockLLM,
    create_agent_society,
    HAS_OPENAI,
    HAS_ANTHROPIC,
    HAS_AGENT_SOCIETY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LLM Agent 补贴仿真实验"
    )
    parser.add_argument(
        "--n-agents", type=int, default=20,
        help="Agent数量 (默认: 20)"
    )
    parser.add_argument(
        "--n-rounds", type=int, default=6,
        help="仿真轮数 (默认: 6)"
    )
    parser.add_argument(
        "--model", type=str, default=os.environ.get("LLM_MODEL", "gpt-4o"),
        help="LLM模型名称 (默认: gpt-4o)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="LLM温度参数 (默认: 0.7)"
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["openai", "anthropic", "mock"],
        help="强制指定后端 (默认: 自动选择)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果输出JSON路径 (可选)"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="并行执行决策"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式（减少输出）"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="快速演示模式（5个Agent，2轮）"
    )
    return parser.parse_args()


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    env_status = {
        "OpenAI API Key": "已设置" if HAS_OPENAI else "未设置",
        "Anthropic API Key": "已设置" if HAS_ANTHROPIC else "未设置",
        "AgentSociety (清华)": "可用" if HAS_AGENT_SOCIETY else "不可用",
    }

    for key, value in env_status.items():
        status_icon = "+" if "已设置" in value or "可用" in value else "-"
        print(f"  [{status_icon}] {key}: {value}")

    # 确定运行模式
    if HAS_OPENAI:
        mode = "OpenAI API (真实LLM)"
    elif HAS_ANTHROPIC:
        mode = "Anthropic API (真实LLM)"
    else:
        mode = "Mock模式 (基于规则的随机决策)"

    print(f"\n  运行模式: {mode}")
    print()

    return mode


def create_subsidy_scenarios(n_rounds: int) -> list:
    """
    创建多轮补贴策略场景

    设计不同的A/B测试策略组合：
    - 面值梯度: 10, 15, 20, 25, 30, 35
    - 门槛梯度: 30, 50, 80, 100, 120, 150
    - 品类轮换: 通用, 餐饮, 购物, 出行, 娱乐, 通用
    - 类型组合: 满减, 折扣, 满减, 直减, 满减, 折扣
    """
    scenarios = [
        SubsidyInfo(
            subsidy_id=f"scenario_{i+1}",
            face_value=[10, 15, 20, 25, 30, 35][i],
            threshold=[30, 50, 80, 100, 120, 150][i],
            category=["通用", "餐饮", "购物", "出行", "娱乐", "通用"][i],
            expire_days=[3, 5, 7, 7, 5, 3][i],
            discount_type=["满减", "折扣", "满减", "直减", "满减", "折扣"][i],
            discount_rate=[None, 0.85, None, None, None, 0.9][i],
            subsidy_pool=100000.0
        )
        for i in range(n_rounds)
    ]
    return scenarios


def run_single_agent_test(args):
    """单个Agent的决策测试"""
    print("=" * 60)
    print("单Agent决策测试")
    print("=" * 60)

    # 创建4种心理账户的Agent
    profiles = [
        AgentProfile(
            user_id="test_windfall",
            mental_account="windfall_spender",
            price_sensitivity=0.2,
            monthly_budget=4000.0,
            age_group="26-35",
            income_level="medium",
            category_preference=["餐饮", "购物"]
        ),
        AgentProfile(
            user_id="test_price_sensitive",
            mental_account="price_sensitive",
            price_sensitivity=0.9,
            monthly_budget=2500.0,
            age_group="18-25",
            income_level="low",
            category_preference=["餐饮", "购物"]
        ),
        AgentProfile(
            user_id="test_routine",
            mental_account="routine_income",
            price_sensitivity=0.4,
            monthly_budget=6000.0,
            age_group="36-50",
            income_level="high",
            category_preference=["出行", "教育"]
        ),
        AgentProfile(
            user_id="test_deal_seeker",
            mental_account="deal_seeker",
            price_sensitivity=0.7,
            monthly_budget=3500.0,
            age_group="26-35",
            income_level="medium",
            category_preference=["购物", "娱乐"]
        ),
    ]

    # 测试补贴
    subsidy = SubsidyInfo(
        subsidy_id="test_001",
        face_value=20.0,
        threshold=80.0,
        category="餐饮",
        expire_days=7,
        discount_type="满减",
        subsidy_pool=100000.0
    )

    for profile in profiles:
        agent = LLMSubsidyAgent(
            profile=profile,
            model_name=args.model,
            temperature=args.temperature,
            backend=args.backend
        )

        print(f"\n--- {profile.user_id} ({profile.mental_account}) ---")
        print(f"  系统提示词 (前150字):")
        sys_prompt = agent.build_system_prompt()
        print(f"  {sys_prompt[:150]}...")

        print(f"\n  决策提示词:")
        from src.simulation.llm_agent import PromptTemplate
        decision_prompt = PromptTemplate.build_decision_prompt(subsidy, profile)
        print(f"  {decision_prompt[:200]}...")

        print(f"\n  决策结果:")
        decision = agent.decide(subsidy, round_num=0)
        print(f"    接受: {decision.accept}")
        print(f"    概率: {decision.usage_probability:.2f}")
        print(f"    预期消费: {decision.expected_spend:.0f}元")
        print(f"    净收益: {decision.net_benefit:.0f}元")
        print(f"    推理: {decision.reasoning}")
        print(f"    后端: {decision.api_used} | 耗时: {decision.llm_call_time:.3f}s")


def run_full_simulation(args):
    """完整多Agent多轮仿真"""
    print("=" * 60)
    print("多Agent多轮仿真实验")
    print("=" * 60)

    # 创建Agent社会
    print(f"\n创建 {args.n_agents} 个Agent...")
    start_time = time.time()

    society = create_agent_society(
        n_agents=args.n_agents,
        seed=args.seed,
        model_name=args.model,
        temperature=args.temperature,
        verbose=not args.quiet
    )
    society.parallel = args.parallel

    # 统计Agent分布
    account_counts = {}
    for agent in society.agents:
        acc = agent.profile.mental_account
        account_counts[acc] = account_counts.get(acc, 0) + 1
    print(f"心理账户分布: {account_counts}")

    # 创建补贴策略
    print(f"\n创建 {args.n_rounds} 轮补贴策略...")
    scenarios = create_subsidy_scenarios(args.n_rounds)
    for i, s in enumerate(scenarios):
        print(f"  第{i+1}轮: {s.face_value}元{s.discount_type} | "
              f"满{s.threshold:.0f} | {s.category} | {s.expire_days}天有效")

    # 执行仿真
    print(f"\n开始仿真...")
    sim_start = time.time()

    result = society.run_simulation(
        subsidy_policies=scenarios,
        n_rounds=args.n_rounds
    )

    sim_time = time.time() - sim_start
    total_time = time.time() - start_time

    # 输出结果
    stats = result["overall_stats"]
    print(f"\n{'='*60}")
    print("仿真结果")
    print(f"{'='*60}")
    print(f"  仿真耗时: {sim_time:.2f}s (总计: {total_time:.2f}s)")
    print(f"  Agent数量: {result['agent_count']}")
    print(f"  仿真轮数: {result['n_rounds']}")
    print(f"  总决策次数: {stats['total_decisions']}")
    print(f"  整体接受率: {stats['overall_accept_rate']:.2%}")
    print(f"  总净收益: {stats['total_net_benefit']:.2f}元")
    print(f"  平均净收益/Agent: {stats['avg_net_benefit_per_agent']:.2f}元")
    print(f"  补贴成本: {stats['total_subsidy_cost']:.2f}元")
    print(f"  后端信息: {stats['backend_info']}")

    # 各心理账户接受率
    print(f"\n  各心理账户接受率:")
    for acc, rate in stats.get("accept_rate_by_account", {}).items():
        bar = "█" * int(rate * 30)
        print(f"    {acc:20s}: {rate:.2%} {bar}")

    # 各轮接受率变化
    print(f"\n  各轮接受率变化:")
    for rnd, rate in sorted(stats.get("accept_rate_by_round", {}).items()):
        bar = "█" * int(rate * 30)
        policy = scenarios[rnd]
        print(f"    第{rnd+1}轮 ({policy.face_value}元满{policy.threshold:.0f}): "
              f"{rate:.2%} {bar}")

    # 推理过程采样
    print(f"\n  推理过程采样 (每类1个Agent):")
    sampled = {}
    for user_id, trajectory in result["trajectories"].items():
        agent = None
        for a in society.agents:
            if a.profile.user_id == user_id:
                agent = a
                break
        if agent and agent.profile.mental_account not in sampled:
            sampled[agent.profile.mental_account] = trajectory[-1]
        if len(sampled) >= 4:
            break

    for account, decision in sampled.items():
        print(f"\n    [{account}]")
        print(f"    决策: {'接受' if decision.accept else '拒绝'}")
        print(f"    推理: {decision.reasoning[:120]}...")

    # 导出DataFrame
    df = society.get_trajectory_dataframe()
    print(f"\n  决策轨迹DataFrame: {df.shape[0]}行 x {df.shape[1]}列")

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        os.makedirs("data/results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"data/results/llm_agent_simulation_{timestamp}.json"

    # 序列化结果
    output_data = {
        "config": {
            "n_agents": args.n_agents,
            "n_rounds": args.n_rounds,
            "model": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
            "backend": result["backend_info"]
        },
        "overall_stats": {
            k: v for k, v in stats.items()
            if not isinstance(v, (dict, list)) or k in ["accept_rate_by_account", "accept_rate_by_round"]
        },
        "round_summaries": [
            {
                k: v for k, v in summary.items()
                if k != "subsidy_policy" or isinstance(v, dict)
            }
            for summary in result["round_summaries"]
        ],
        "agent_count": result["agent_count"],
        "n_rounds": result["n_rounds"],
        "simulation_time": sim_time,
        "total_time": total_time
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存到: {output_path}")

    # 同时保存CSV
    csv_path = output_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  决策轨迹CSV已保存到: {csv_path}")

    return result


def run_ab_test(args):
    """
    A/B测试：对比不同补贴策略的效果

    测试维度：
    1. 面值变化: 10, 20, 30元
    2. 门槛变化: 无门槛, 低门槛, 高门槛
    3. 品类变化: 通用, 餐饮, 购物
    """
    print("=" * 60)
    print("A/B测试: 补贴策略对比实验")
    print("=" * 60)

    society = create_agent_society(
        n_agents=min(args.n_agents, 30),
        seed=args.seed,
        model_name=args.model,
        temperature=args.temperature,
        verbose=False
    )
    society.verbose = False

    # 测试策略
    test_policies = {
        "面值10元-无门槛": SubsidyInfo("ab_1", 10, 0, "通用", 7, "满减", subsidy_pool=100000),
        "面值10元-满50": SubsidyInfo("ab_2", 10, 50, "通用", 7, "满减", subsidy_pool=100000),
        "面值20元-无门槛": SubsidyInfo("ab_3", 20, 0, "通用", 7, "满减", subsidy_pool=100000),
        "面值20元-满80": SubsidyInfo("ab_4", 20, 80, "通用", 7, "满减", subsidy_pool=100000),
        "面值30元-无门槛": SubsidyInfo("ab_5", 30, 0, "通用", 7, "满减", subsidy_pool=100000),
        "面值30元-满120": SubsidyInfo("ab_6", 30, 120, "通用", 7, "满减", subsidy_pool=100000),
        "品类-餐饮": SubsidyInfo("ab_7", 20, 50, "餐饮", 7, "满减", subsidy_pool=100000),
        "品类-购物": SubsidyInfo("ab_8", 20, 50, "购物", 7, "满减", subsidy_pool=100000),
    }

    results = {}
    for name, policy in test_policies.items():
        policy.agent_count = len(society.agents)
        policy.per_capita = policy.subsidy_pool / max(len(society.agents), 1)

        decisions = society.run_round(policy, round_num=0, context=None)
        accept_rate = sum(1 for d in decisions.values() if d.accept) / len(decisions)
        avg_net = sum(d.net_benefit for d in decisions.values()) / len(decisions)
        avg_spend = sum(d.expected_spend for d in decisions.values()) / len(decisions)

        results[name] = {
            "accept_rate": accept_rate,
            "avg_net_benefit": avg_net,
            "avg_expected_spend": avg_spend
        }

    # 打印结果
    print(f"\n{'策略':<20s} {'接受率':>8s} {'平均净收益':>10s} {'平均消费':>10s}")
    print("-" * 52)
    for name, metrics in results.items():
        print(f"{name:<20s} {metrics['accept_rate']:>7.2%} "
              f"{metrics['avg_net_benefit']:>9.2f}元 "
              f"{metrics['avg_expected_spend']:>9.2f}元")

    # 分析
    print(f"\n{'='*60}")
    print("A/B测试分析")
    print(f"{'='*60}")

    # 面值效果
    print("\n面值效果分析:")
    for face_value in [10, 20, 30]:
        no_threshold = results.get(f"面值{face_value}元-无门槛", {}).get("accept_rate", 0)
        with_threshold = results.get(f"面值{face_value}元-满{int(face_value*2.5 if face_value==10 else face_value*4)}", {}).get("accept_rate", 0)
        print(f"  {face_value}元: 无门槛 {no_threshold:.2%} | 有门槛 {with_threshold:.2%}")

    # 品类效果
    print("\n品类效果分析:")
    for cat_name in ["品类-餐饮", "品类-购物"]:
        rate = results.get(cat_name, {}).get("accept_rate", 0)
        print(f"  {cat_name}: {rate:.2%}")

    return results


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("LLM Agent 补贴仿真系统")
    print("基于大语言模型的消费者行为仿真")
    print("=" * 60)
    print()

    # 演示模式
    if args.demo:
        from src.simulation.llm_agent import demo
        demo()
        return

    # 环境检查
    mode = check_environment()

    # 单Agent测试
    run_single_agent_test(args)

    print("\n")

    # 完整仿真
    run_full_simulation(args)

    print("\n")

    # A/B测试
    run_ab_test(args)

    print(f"\n{'='*60}")
    print("所有实验完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
