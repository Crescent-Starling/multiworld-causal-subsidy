"""
LLM Agent 补贴仿真运行脚本

使用 LLMSubsidyAgent / LLMAgentSociety 进行仿真。
- 如果环境变量设置了 OPENAI_API_KEY 或 ANTHROPIC_API_KEY，可接入真实 LLM
- 默认回退到 Mock 模式（基于规则的决策，保留推理格式）

用法：
    python scripts/run_llm_agent.py
    python scripts/run_llm_agent.py --n-agents 50 --n-rounds 10
    python scripts/run_llm_agent.py --backend openai --model gpt-4o-mini

环境变量：
    OPENAI_API_KEY:    OpenAI API 密钥
    DEEPSEEK_API_KEY: DeepSeek API 密钥（OpenAI 兼容格式）
    ANTHROPIC_API_KEY: Anthropic API 密钥
    LLM_BACKEND:       强制指定后端 (openai / deepseek / anthropic / mock)

参考文献：
- Park et al. (2023): Generative Agents
- Horton (2023): Homo Silicus, NBER Working Paper
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.llm_agent import (
    PromptTemplate,
    LLMClient,
    LLMSubsidyAgent,
    LLMAgentSociety,
    run_llm_simulation,
)


# ===========================================================================
# 环境检查
# ===========================================================================

def check_environment() -> str:
    """检查运行环境，返回运行模式描述"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    has_deepseek_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    backend_env = os.environ.get("LLM_BACKEND", "").strip()

    env_status = {
        "OPENAI_API_KEY": "已设置" if has_openai_key else "未设置",
        "DEEPSEEK_API_KEY": "已设置" if has_deepseek_key else "未设置",
        "ANTHROPIC_API_KEY": "已设置" if has_anthropic_key else "未设置",
        "LLM_BACKEND (env override)": backend_env if backend_env else "未设置 (自动选择)",
    }

    for key, value in env_status.items():
        icon = "+" if ("已设置" in value or value in ("openai", "deepseek", "anthropic")) else "-"
        print(f"  [{icon}] {key}: {value}")

    # 确定运行模式
    if backend_env == "openai" or (not backend_env and has_openai_key):
        mode = "OpenAI API (真实 LLM)"
    elif backend_env == "deepseek" or (not backend_env and has_deepseek_key):
        mode = "DeepSeek API (真实 LLM)"
    elif backend_env == "anthropic" or (not backend_env and has_anthropic_key):
        mode = "Anthropic API (真实 LLM)"
    else:
        mode = "Mock 模式 (基于规则的决策回退)"

    print(f"\n  运行模式: {mode}")
    print()
    return mode


# ===========================================================================
# 单 Agent 决策测试
# ===========================================================================

def run_single_agent_test(args, llm_client: LLMClient):
    """测试单个 Agent 的决策输出格式"""
    print("=" * 60)
    print("单 Agent 决策测试")
    print("=" * 60)

    # 创建 4 种心理账户的 Agent（仅测试决策格式，不依赖真实 LLM）
    test_profiles = [
        {"agent_id": "test_windfall", "mental_account": "windfall_spender", "price_sensitivity": 0.2},
        {"agent_id": "test_price_sensitive", "mental_account": "price_sensitive", "price_sensitivity": 0.9},
        {"agent_id": "test_routine", "mental_account": "routine_income", "price_sensitivity": 0.4},
        {"agent_id": "test_deal_seeker", "mental_account": "deal_seeker", "price_sensitivity": 0.7},
    ]

    agents = []
    for p in test_profiles:
        agent = LLMSubsidyAgent(
            agent_id=p["agent_id"],
            mental_account=p["mental_account"],
            price_sensitivity=p["price_sensitivity"],
            income_level=3,
            consumption_freq=5,
            llm_client=llm_client,
        )
        agents.append(agent)

    # 显示系统提示词（截取前 200 字符）
    print("\n系统提示词模板预览（windfall_spender）:")
    sys_prompt = PromptTemplate.get_system_prompt("windfall_spender")
    print(f"  {sys_prompt[:200]}...")

    # 对每个 Agent 执行一轮决策
    subsidy_amount = 20.0
    threshold = 80.0

    print(f"\n决策测试（补贴金额={subsidy_amount}元，门槛={threshold}元）:")
    for agent in agents:
        result = agent.decide(subsidy_amount=subsidy_amount, threshold=threshold)
        accepted = result["redeemed"]
        reasoning = result.get("reasoning", "")[:80]
        confidence = result.get("confidence", 0.0)
        print(f"  [{agent.mental_account:<20s}] "
              f"决策: {'核销' if accepted else '不核销'} | "
              f"信心: {confidence:.2f} | 推理: {reasoning}...")

    print()
    return agents


# ===========================================================================
# 完整多 Agent 多轮仿真
# ===========================================================================

def run_full_simulation(args, llm_client: LLMClient) -> Optional[pd.DataFrame]:
    """运行完整仿真并输出汇总"""
    print("=" * 60)
    print("多 Agent 多轮仿真实验")
    print("=" * 60)

    np.random.seed(args.seed)

    # 创建 Agent 社会
    print(f"\n创建 {args.n_agents} 个 LLM Agent...")
    start_time = time.time()

    society = LLMAgentSociety(
        n_agents=args.n_agents,
        use_mock=(llm_client.backend == "mock"),
        backend=llm_client.backend,
        api_key=llm_client.api_key,
        seed=args.seed,
    )

    # 统计心理账户分布
    account_counts: dict = {}
    for agent in society.agents:
        acc = agent.mental_account
        account_counts[acc] = account_counts.get(acc, 0) + 1
    print(f"心理账户分布: {account_counts}")

    # 执行多轮仿真
    print(f"\n开始 {args.n_rounds} 轮仿真...")
    sim_start = time.time()

    for r in range(args.n_rounds):
        subsidy = args.base_subsidy + r * args.subsidy_increment
        threshold = args.base_threshold + r * args.threshold_increment
        result = society.run_round(subsidy_amount=subsidy, threshold=threshold)
        mode_label = "MOCK" if society.use_mock else "LLM"
        if not args.quiet:
            print(f"  Round {result['round']} [{mode_label}]: "
                  f"subsidy=¥{subsidy:.0f}, threshold=¥{threshold:.0f}, "
                  f"redemption_rate={result['redemption_rate']:.2%} "
                  f"({result['n_redeemed']}/{society.n_agents})")

    sim_time = time.time() - sim_start
    total_time = time.time() - start_time

    # 输出汇总
    summary_df = society.get_summary()
    print(f"\n{'=' * 60}")
    print("仿真结果汇总")
    print(f"{'=' * 60}")
    print(f"  仿真耗时: {sim_time:.2f}s (总计: {total_time:.2f}s)")
    print(f"  平均核销率: {summary_df['redemption_rate'].mean():.2%}")
    print()
    print(summary_df.to_string(index=False))

    # 心理账户维度汇总
    traj_df = society.get_trajectory_df()
    if not traj_df.empty:
        print(f"\n各心理账户核销率:")
        for acc, group in traj_df.groupby("mental_account"):
            rate = group["redeemed"].mean()
            bar = "█" * int(rate * 30)
            print(f"  {acc:<25s}: {rate:.2%} {bar}")

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"llm_agent_simulation_{timestamp}.json")

    output_data = {
        "config": {
            "n_agents": args.n_agents,
            "n_rounds": args.n_rounds,
            "backend": llm_client.backend,
            "seed": args.seed,
            "base_subsidy": args.base_subsidy,
            "subsidy_increment": args.subsidy_increment,
            "base_threshold": args.base_threshold,
            "threshold_increment": args.threshold_increment,
        },
        "summary": summary_df.to_dict(orient="records"),
        "trajectories": traj_df.to_dict(orient="records") if not traj_df.empty else [],
        "simulation_time": sim_time,
        "total_time": total_time,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存到: {output_path}")

    # 同时保存 CSV
    if not traj_df.empty:
        csv_path = output_path.replace(".json", ".csv")
        traj_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"决策轨迹 CSV 已保存到: {csv_path}")

    print()
    return summary_df


# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Agent 补贴仿真实验"
    )
    parser.add_argument(
        "--n-agents", type=int, default=20,
        help="Agent 数量 (默认: 20)"
    )
    parser.add_argument(
        "--n-rounds", type=int, default=6,
        help="仿真轮数 (默认: 6)"
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["openai", "deepseek", "anthropic", "mock"],
        help="强制指定 LLM 后端 (默认: 自动选择)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM 模型名称 (OpenAI: gpt-4o-mini; DeepSeek: deepseek-chat / deepseek-reasoner)"
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="自定义 API base URL (DeepSeek 默认: https://api.deepseek.com/v1)"
    )
    parser.add_argument(
        "--base-subsidy", type=float, default=10.0,
        help="初始补贴金额 (默认: 10.0)"
    )
    parser.add_argument(
        "--subsidy-increment", type=float, default=2.0,
        help="每轮补贴递增金额 (默认: 2.0)"
    )
    parser.add_argument(
        "--base-threshold", type=float, default=30.0,
        help="初始使用门槛 (默认: 30.0)"
    )
    parser.add_argument(
        "--threshold-increment", type=float, default=5.0,
        help="每轮门槛递增值 (默认: 5.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果输出 JSON 路径 (可选)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/results",
        help="输出目录 (默认: data/results)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式（减少每轮输出）"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="快速演示模式（5 个 Agent，2 轮）"
    )
    return parser.parse_args()


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    args = parse_args()

    # 演示模式：使用内置 demo 函数
    if args.demo:
        print("=" * 60)
        print("LLM Agent 仿真 — 演示模式")
        print("=" * 60)
        result = run_llm_simulation(
            n_agents=5,
            n_rounds=2,
            use_mock=True,
        )
        print("\n演示完成。")
        return

    # 环境检查
    mode = check_environment()

    # 确定后端
    backend = args.backend
    if backend is None:
        if os.environ.get("OPENAI_API_KEY"):
            backend = "openai"
        elif os.environ.get("DEEPSEEK_API_KEY"):
            backend = "deepseek"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            backend = "anthropic"
        else:
            backend = "mock"

    api_key = None
    base_url = args.base_url
    model = args.model

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        model = model or "gpt-4o-mini"
    elif backend == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        model = model or "deepseek-chat"
        base_url = base_url or "https://api.deepseek.com/v1"
    elif backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        model = model or "claude-3-haiku-20240307"

    llm_client = LLMClient(
        backend=backend,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    # 单 Agent 测试
    run_single_agent_test(args, llm_client)
    print()

    # 完整仿真
    run_full_simulation(args, llm_client)

    print("=" * 60)
    print("所有实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
