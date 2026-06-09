#!/usr/bin/env python3
"""
验证商家原型 + v2 LLM Agent Prompt 是否正常工作
"""

import sys
import os
import json

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_merchant_prototype():
    """测试商家原型模块"""
    print("=" * 60)
    print("Test 1: MerchantPrototype")
    print("=" * 60)

    from src.simulation.merchant_prototype import (
        MerchantPrototype,
        MerchantPrototypeFactory,
        MerchantRegistry,
        build_merchant_registry,
    )

    # 检查数据文件是否存在
    order_path = "data/神券订单数据样例.xlsx"
    behavior_path = "data/用户行为序列.xlsx"

    if not os.path.exists(order_path) or not os.path.exists(behavior_path):
        print(f"  SKIP: data files not found ({order_path}, {behavior_path})")
        print("  -> 测试商家原型基础功能（无数据）")
        # 无数据模式下仍然可以测试基础结构
        proto = MerchantPrototype(
            prototype_id="fast_food",
            display_name="快餐简餐",
            bu="外卖",
            category_l1="美食",
            category_l2="快餐简餐",
            avg_order_price=35.6,
            price_range=(10.0, 80.0),
            median_order_price=31.5,
            avg_subsidy=7.8,
            subsidy_rate=0.22,
            typical_shops=["肯德基", "麦当劳", "南京小吃"],
            typical_skus=["鸡腿堡套餐", "酸辣粉", "卤肉饭"],
            n_orders=287,
            tags=["外卖主力", "高频刚需"],
        )
        print(f"  [OK] MerchantPrototype created: {proto.display_name}")

        # 测试场景描述
        scene = proto.describe_scene(subsidy_amount=10, threshold=30, time_of_day="午餐时段")
        print(f"  [OK] describe_scene(): {scene[:80]}...")
        return True

    # 使用真实数据构建
    factory = MerchantPrototypeFactory()
    factory.load_data(order_path=order_path, behavior_path=behavior_path)
    prototypes = factory.build()

    print(f"  [OK] Built {len(prototypes)} prototypes from data")
    for pid, p in list(prototypes.items())[:5]:
        print(f"    - {pid}: {p.display_name} | avg_price={p.avg_order_price:.1f} | shops={len(p.typical_shops)}")

    # 测试注册表
    registry = MerchantRegistry(prototypes=prototypes)
    sampled = registry.sample(bu="外卖")
    print(f"  [OK] MerchantRegistry.sample(bu='外卖'): {sampled.display_name}")

    # 测试场景描述
    proto = list(prototypes.values())[0]
    scene = proto.describe_scene(subsidy_amount=10, threshold=30, time_of_day="午餐时段")
    print(f"  [OK] describe_scene(): {scene[:80]}...")

    # 测试to_context_config
    ctx = proto.to_context_config(time_of_day=1, competition_intensity=0.3)
    print(f"  [OK] to_context_config(): merchant_cat={ctx.merchant_category}, price={ctx.price_level}")

    return True


def test_v2_prompt():
    """测试v2情境化Prompt"""
    print("\n" + "=" * 60)
    print("Test 2: v2 PromptTemplate (情境化)")
    print("=" * 60)

    from src.simulation.llm_agent import PromptTemplate
    from src.simulation.merchant_prototype import MerchantPrototype

    # 构建一个虚拟商家原型
    proto = MerchantPrototype(
        prototype_id="fast_food",
        display_name="快餐简餐",
        bu="外卖",
        category_l1="美食",
        category_l2="快餐简餐",
        avg_order_price=35.6,
        price_range=(10.0, 80.0),
        median_order_price=31.5,
        typical_shops=["肯德基宅急送", "麦当劳", "南京小吃"],
        typical_skus=["鸡腿堡套餐", "酸辣粉"],
        n_orders=287,
    )

    # v2 system prompt
    sys_prompt = PromptTemplate.get_system_prompt("windfall_spender", use_v2=True)
    print(f"  [OK] v2 system_prompt (windfall_spender): {len(sys_prompt)} chars")
    assert "横财" in sys_prompt or "意外" in sys_prompt

    # v2 user prompt（含场景描述）
    scene = proto.describe_scene(subsidy_amount=10, threshold=30, time_of_day="午餐时段")
    user_prompt = PromptTemplate.get_user_prompt(
        use_v2=True,
        scene_description=scene,
        consumption_freq=5,
        recent_subsidy="否",
        fatigue_level="低",
        fatigue_explanation="近期补贴较少，你对新的优惠还有新鲜感",
    )
    print(f"  [OK] v2 user_prompt: {len(user_prompt)} chars")
    assert "肯德基" in user_prompt or "快餐" in user_prompt
    print(f"  [OK] Scene injected into prompt: {scene[:60]}...")

    # v1 backward compatibility
    sys_v1 = PromptTemplate.get_system_prompt("price_sensitive", use_v2=False)
    user_v1 = PromptTemplate.get_user_prompt(
        use_v2=False,
        subsidy_amount=10,
        threshold=30,
        consumption_freq=5,
        recent_subsidy="否",
        fatigue_level="低",
    )
    print(f"  [OK] v1 backward compat: system={len(sys_v1)} chars, user={len(user_v1)} chars")

    return True


def test_v2_llm_agent():
    """测试v2 LLMSubsidyAgent（mock模式）"""
    print("\n" + "=" * 60)
    print("Test 3: v2 LLMSubsidyAgent (mock mode)")
    print("=" * 60)

    from src.simulation.llm_agent import LLMClient, LLMSubsidyAgent

    # 创建mock client
    client = LLMClient(backend="mock")

    agent = LLMSubsidyAgent(
        agent_id="test_v2_0",
        mental_account="windfall_spender",
        price_sensitivity=0.6,
        income_level=3,
        consumption_freq=5,
        llm_client=client,
        use_v2_prompt=True,
    )
    print(f"  [OK] LLMSubsidyAgent created (v2_prompt=True)")

    from src.simulation.merchant_prototype import MerchantPrototype

    # 构建商家原型
    merchant = MerchantPrototype(
        prototype_id="fast_food",
        display_name="快餐简餐",
        bu="外卖",
        category_l1="美食",
        category_l2="快餐简餐",
        avg_order_price=35.6,
        price_range=(10.0, 80.0),
        median_order_price=31.5,
        typical_shops=["肯德基宅急送", "麦当劳"],
        typical_skus=["鸡腿堡套餐", "酸辣粉"],
        n_orders=287,
    )

    # v2 决策（含商家原型）
    result = agent.decide(
        subsidy_amount=10,
        threshold=30,
        merchant=merchant,
        time_of_day="午餐时段",
    )
    print(f"  [OK] v2 decide(): redeemed={result['redeemed']}")
    print(f"  [OK] reasoning: {result.get('reasoning', '')[:80]}...")

    assert "merchant" in agent.trajectory[-1]
    print(f"  [OK] trajectory has 'merchant': {agent.trajectory[-1]['merchant']}")

    return True


def test_v2_society():
    """测试v2 LLMAgentSociety（mock模式）"""
    print("\n" + "=" * 60)
    print("Test 4: v2 LLMAgentSociety (mock + merchant registry)")
    print("=" * 60)

    from src.simulation.llm_agent import LLMClient, LLMAgentSociety
    from src.simulation.merchant_prototype import (
        MerchantPrototype, MerchantRegistry
    )

    client = LLMClient(backend="mock")

    # 手动创建两个原型
    p1 = MerchantPrototype(
        prototype_id="fast_food", display_name="快餐简餐", bu="外卖",
        category_l1="美食", category_l2="快餐简餐",
        avg_order_price=35.6, price_range=(10, 80), median_order_price=31.5,
        typical_shops=["肯德基", "麦当劳"], typical_skus=["鸡腿堡套餐"],
        n_orders=287,
    )
    p2 = MerchantPrototype(
        prototype_id="drinks", display_name="饮品", bu="外卖",
        category_l1="饮品", category_l2="奶茶",
        avg_order_price=25.0, price_range=(10, 60), median_order_price=20.0,
        typical_shops=["茉酸奶", "JuiceLab"], typical_skus=["杨枝甘露"],
        n_orders=66,
    )
    registry = MerchantRegistry(prototypes={"fast_food": p1, "drinks": p2})

    society = LLMAgentSociety(
        n_agents=4,
        use_mock=True,
        merchant_registry=registry,
    )
    print(f"  [OK] LLMAgentSociety created with merchant_registry")

    result = society.run_round(subsidy_amount=10, threshold=30)
    print(f"  [OK] run_round(): redemption_rate={result['redemption_rate']:.2%}")

    return True


def test_integration_mesa():
    """测试与mesa_agent_model的集成"""
    print("\n" + "=" * 60)
    print("Test 5: Integration with mesa_agent_model")
    print("=" * 60)

    from src.simulation.mesa_agent_model import SubsidyModel, StrategyType
    from src.simulation.merchant_prototype import (
        MerchantPrototype, MerchantRegistry
    )

    # 创建商家注册表
    p1 = MerchantPrototype(
        prototype_id="fast_food", display_name="快餐简餐", bu="外卖",
        category_l1="美食", category_l2="快餐简餐",
        avg_order_price=35.6, price_range=(10, 80), median_order_price=31.5,
        typical_shops=["肯德基", "麦当劳"], typical_skus=["鸡腿堡套餐"],
        n_orders=287,
    )
    registry = MerchantRegistry(prototypes={"fast_food": p1})

    model = SubsidyModel(
        n_agents=10,
        strategy="cognitive",
        seed=42,
        behavior_chain_enabled=True,
        merchant_registry=registry,
    )
    print(f"  [OK] SubsidyModel created with merchant_registry")

    # 检查 _assign_contexts 是否正常
    model.step()
    print(f"  [OK] model.step() completed, round={model.current_round}")

    # 检查是否有 Agent 拿到了 current_merchant
    n_with_merchant = sum(
        1 for a in model.agents if getattr(a, 'current_merchant', None) is not None
    )
    print(f"  [OK] {n_with_merchant}/{model.n_agents} agents got current_merchant")

    return True


if __name__ == "__main__":
    print("\n 商家原型 + v2 LLM Agent Prompt 验证\n")

    tests = [
        ("MerchantPrototype", test_merchant_prototype),
        ("v2 PromptTemplate", test_v2_prompt),
        ("v2 LLMSubsidyAgent", test_v2_llm_agent),
        ("v2 LLMAgentSociety", test_v2_society),
        ("Integration mesa", test_integration_mesa),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, "PASS" if ok else "FAIL"))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  [{status}] {name}")
    print()
