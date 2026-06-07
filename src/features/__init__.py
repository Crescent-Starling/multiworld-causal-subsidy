"""
特征工程与数据生成模块
=====================

提供合成数据生成功能，用于框架演示和测试。

公共接口:
- SyntheticDataConfig: 合成数据配置
- generate_user_profiles: 生成用户画像
- generate_order_data: 生成订单数据
- generate_causal_inference_data: 生成因果推断数据
- generate_all_data: 一键生成全部数据
"""

from .data_generator import (
    SyntheticDataConfig,
    generate_user_profiles,
    generate_order_data,
    generate_causal_inference_data,
    generate_all_data,
)

__all__ = [
    "SyntheticDataConfig",
    "generate_user_profiles",
    "generate_order_data",
    "generate_causal_inference_data",
    "generate_all_data",
]
