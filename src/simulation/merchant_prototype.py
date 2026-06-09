"""
商家原型模块 (Merchant Prototype)
===================================
从美团真实订单数据聚类构建商家原型，为 LLM Agent 提供具体消费场景锚定。

核心洞察：
  当前 LLM Agent 的推理高度同质化——只有抽象数字（补贴10元/门槛150元），
  LLM 只能做算术比较（折扣率），无法联想具体消费场景。
  商家原型让 Prompt 从"补贴10元"变为"你打开美团，肯德基有满30减10"，
  使 LLM 的推理从「计算折扣率」切换到「评估具体场景中的需求和机会成本」。

数据来源：
  - 神券订单950条：POI分类(61种) / BU(18种) / bu_name(4种) / 订单金额 / 美补金额
  - 用户行为序列478条：点击店铺(店铺名+品类+均单价) / 搜索 / 下单

原型设计：
  从 POI分类 + BU 聚合，保留出现频次≥5的品类，
  再合并为 10 个商家原型，覆盖外卖/到餐/闪购/乐生活四大业务线。

参考文献:
  - 美团分类体系：行业→业态→细分品类→场景特色（四级递进）
"""

from __future__ import annotations

import re
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


# ===========================================================================
# 商家原型数据结构
# ===========================================================================

@dataclass
class MerchantPrototype:
    """
    单个商家原型

    由真实美团数据聚类而来，代表一类商家（如快餐简餐、饮品、超市等），
    包含品类标签、均价区间、典型店铺名、时段分布、典型SKU等。

    用于：
    1. 注入 LLM Agent Prompt——从抽象数字到具体场景
    2. 扩展 ContextConfig——从5维粗粒度到更丰富的商家描述
    3. 支持用户×商家×场景的三元交互决策建模
    """

    # 原型标识
    prototype_id: str = ""                # e.g. "fast_food"
    display_name: str = ""                # e.g. "快餐简餐"
    bu: str = ""                          # 业务线: 外卖/到餐/闪购/乐生活

    # 品类标签（层级）
    category_l1: str = ""                 # 一级分类: 美食/超市便利/饮品/运动健身/丽人...
    category_l2: str = ""                 # 二级分类: 快餐简餐/中式正餐/奶茶/咖啡...
    poi_categories: List[str] = field(default_factory=list)  # 来源POI分类列表

    # 价格信息
    avg_order_price: float = 0.0          # 均单价（元）
    price_range: Tuple[float, float] = (0.0, 0.0)  # (最低, 最高) 订单金额
    median_order_price: float = 0.0       # 中位数订单金额

    # 补贴信息
    avg_subsidy: float = 0.0             # 平均美补金额
    subsidy_rate: float = 0.0            # 补贴占比（美补/订单金额）

    # 时段分布（各时段订单占比）
    peak_hours: Dict[str, float] = field(default_factory=dict)
    # e.g. {"上午": 0.15, "下午": 0.25, "晚间": 0.45, "深夜": 0.15}

    # 典型店铺名（从行为序列提取的真实店铺名）
    typical_shops: List[str] = field(default_factory=list)

    # 典型SKU/商品（手动补充 + 从下单数据提取）
    typical_skus: List[str] = field(default_factory=list)

    # 订单量（该原型覆盖的原始订单数）
    n_orders: int = 0

    # 标签（用于LLM Prompt场景描述）
    tags: List[str] = field(default_factory=list)

    def describe_scene(self, subsidy_amount: float, threshold: float,
                       time_of_day: str = "午餐时段") -> str:
        """
        生成情境化场景描述——直接注入LLM Prompt

        从抽象的"补贴10元/门槛150元"变为
        "你打开美团外卖，看到肯德基(西式快餐,人均36元)有满30减10的神券"

        参数:
            subsidy_amount: 补贴金额
            threshold: 使用门槛
            time_of_day: 当前时段描述

        返回:
            场景描述文本
        """
        # 随机选择一个典型店铺
        shop = np.random.choice(self.typical_shops) if self.typical_shops else self.display_name

        # 补贴力度描述
        discount_pct = subsidy_amount / threshold * 100 if threshold > 0 else 0
        if discount_pct >= 30:
            strength_desc = "力度不错"
        elif discount_pct >= 15:
            strength_desc = "力度还可以"
        else:
            strength_desc = "力度一般"

        # 价格水平描述
        if self.avg_order_price <= 25:
            price_level_desc = "平价"
        elif self.avg_order_price <= 60:
            price_level_desc = "中等消费"
        elif self.avg_order_price <= 120:
            price_level_desc = "偏高消费"
        else:
            price_level_desc = "高消费"

        # 业务线场景词
        bu_scene = {
            "外卖": "你打开美团外卖App",
            "到餐": "你在大众点评上找附近餐厅",
            "闪购": "你在美团闪购上买东西",
            "乐生活": "你在美团上找生活服务",
        }.get(self.bu, "你打开美团")

        # 品类场景词
        cat_scene = ""
        if self.category_l1 == "美食":
            cat_scene = f"，看到「{shop}」({self.category_l2}，人均约{self.avg_order_price:.0f}元)"
        elif self.category_l1 == "饮品":
            cat_scene = f"，看到「{shop}」(饮品店，人均约{self.avg_order_price:.0f}元)"
        elif self.category_l1 in ("超市便利",):
            cat_scene = f"，看到「{shop}」(超市，人均约{self.avg_order_price:.0f}元)"
        elif self.category_l1 == "运动健身":
            cat_scene = f"，看到「{shop}」(健身房，月卡约{self.avg_order_price:.0f}元)"
        elif self.category_l1 == "丽人":
            cat_scene = f"，看到「{shop}」({self.category_l2}，人均约{self.avg_order_price:.0f}元)"
        else:
            cat_scene = f"，看到「{shop}」({self.display_name}，人均约{self.avg_order_price:.0f}元)"

        # SKU提示
        sku_hint = ""
        if self.typical_skus:
            sku_sample = np.random.choice(self.typical_skus, size=min(2, len(self.typical_skus)), replace=False)
            sku_hint = f"，比如{'、'.join(sku_sample)}"

        scene = (
            f"{bu_scene}{cat_scene}有满{threshold:.0f}减{subsidy_amount:.0f}的神券"
            f"（{strength_desc}）。"
            f"这是{price_level_desc}类型的{self.category_l1}店铺{sku_hint}。"
            f"现在是{time_of_day}。"
        )

        return scene

    def to_context_config(self, time_of_day: int = 0,
                          session_intent: int = 0,
                          competition_intensity: float = 0.3) -> "ContextConfig":
        """
        将商家原型映射到 ContextConfig

        商家原型的品类信息映射到 ContextConfig.merchant_category：
        0=到餐(美食/餐饮), 1=闪购(食材/水果/酒水), 2=超市便利,
        3=果蔬, 4=其他(运动健身/丽人/休闲娱乐)
        """
        from src.simulation.behavioral_kernel import ContextConfig

        # 品类→merchant_category映射
        cat_map = {
            "美食": 0, "餐饮": 0, "饮品": 0, "甜点": 0,
            "超市便利": 2, "食材": 1, "水果": 3, "酒水茶饮": 1,
            "运动健身": 4, "丽人": 4, "休闲娱乐": 4,
            "休闲食品": 1, "美妆日化": 4, "母婴玩具": 4,
            "养车/用车": 4, "鲜花绿植": 3, "亲子": 4,
            "日用百货": 2,
        }
        merchant_cat = cat_map.get(self.category_l1, 4)

        # 价格水平映射
        if self.avg_order_price <= 25:
            price_level = 0
        elif self.avg_order_price <= 80:
            price_level = 1
        else:
            price_level = 2

        return ContextConfig(
            merchant_category=merchant_cat,
            price_level=price_level,
            time_of_day=time_of_day,
            session_intent=session_intent,
            competition_intensity=competition_intensity,
        )


# ===========================================================================
# 商家原型工厂
# ===========================================================================

class MerchantPrototypeFactory:
    """
    从美团数据构建商家原型

    流程：
    1. 读取神券订单数据（POI分类/BU/订单金额/美补金额）
    2. 读取行为序列数据（点击店铺→店铺名+品类+均单价）
    3. 按 POI分类 聚合统计量
    4. 合并为10个商家原型（覆盖4大业务线）
    5. 补充典型SKU/时段分布/标签
    """

    # POI分类 → 原型映射（将61种POI分类合并为10个原型）
    PROTOTYPE_MAPPING = {
        # 1. 快餐简餐（外卖主力）
        "快餐简餐": "fast_food",
        "小吃快餐": "fast_food",
        "小吃": "fast_food",

        # 2. 中式正餐
        "中式正餐": "chinese_dinner",
        "川湘菜": "chinese_dinner",
        "粤菜": "chinese_dinner",
        "东北菜": "chinese_dinner",

        # 3. 西餐/日韩料理
        "西餐": "western_food",
        "日韩料理": "western_food",

        # 4. 饮品/奶茶/咖啡
        "饮品": "drinks",
        "奶茶": "drinks",
        "咖啡": "drinks",
        "甜点饮品": "drinks",
        "甜点": "drinks",
        "面包蛋糕甜品": "drinks",
        "冰凉甜点": "drinks",

        # 5. 火锅/烧烤
        "火锅": "hotpot_bbq",
        "烧烤烤肉": "hotpot_bbq",

        # 6. 超市便利
        "小型超市": "supermarket",
        "大型超市/卖场": "supermarket",
        "便利店": "supermarket",
        "菜市场": "supermarket",

        # 7. 运动健身
        "健身中心": "fitness",

        # 8. 丽人（美甲/美发/按摩）
        "美甲": "beauty",
        "美发": "beauty",
        "按摩/足疗": "beauty",

        # 9. 水果生鲜
        "果切店": "fruit_fresh",
        "整果店": "fruit_fresh",
        "生鲜蔬果": "fruit_fresh",

        # 10. 其他（酒水/零食等闪购）
        "酒水店": "flash_buy_other",
        "零食店": "flash_buy_other",
        "水果": "flash_buy_other",
        "酒水茶饮": "flash_buy_other",
    }

    # 原型元数据（手动补充的标签和典型SKU）
    PROTOTYPE_META = {
        "fast_food": {
            "display_name": "快餐简餐",
            "bu": "外卖",
            "category_l1": "美食",
            "category_l2": "快餐简餐",
            "typical_skus": ["鸡腿堡套餐", "宫保鸡丁盖饭", "酸辣粉", "卤肉饭套餐"],
            "tags": ["外卖主力", "高频刚需", "低客单价", "快节奏"],
            "peak_hours": {"上午": 0.08, "下午": 0.12, "晚间": 0.65, "深夜": 0.15},
        },
        "chinese_dinner": {
            "display_name": "中式正餐",
            "bu": "外卖",
            "category_l1": "美食",
            "category_l2": "中式正餐",
            "typical_skus": ["酸菜鱼双人餐", "小炒肉套餐", "水煮牛肉", "回锅肉盖饭"],
            "tags": ["聚餐场景", "中等客单价", "品质要求高"],
            "peak_hours": {"上午": 0.05, "下午": 0.15, "晚间": 0.60, "深夜": 0.20},
        },
        "western_food": {
            "display_name": "西式快餐/日韩料理",
            "bu": "外卖",
            "category_l1": "美食",
            "category_l2": "西餐/日韩料理",
            "typical_skus": ["炸鸡套餐", "意式披萨", "寿司拼盘", "咖喱饭"],
            "tags": ["年轻人偏好", "品牌忠诚度高", "标准化出品"],
            "peak_hours": {"上午": 0.05, "下午": 0.20, "晚间": 0.55, "深夜": 0.20},
        },
        "drinks": {
            "display_name": "饮品/奶茶/咖啡",
            "bu": "外卖",
            "category_l1": "饮品",
            "category_l2": "奶茶/咖啡",
            "typical_skus": ["珍珠奶茶", "生椰拿铁", "杨枝甘露", "美式咖啡"],
            "tags": ["下午茶场景", "低客单价", "高频消费", "社交属性"],
            "peak_hours": {"上午": 0.10, "下午": 0.45, "晚间": 0.35, "深夜": 0.10},
        },
        "hotpot_bbq": {
            "display_name": "火锅/烧烤",
            "bu": "到餐",
            "category_l1": "美食",
            "category_l2": "火锅/烧烤",
            "typical_skus": ["双人火锅套餐", "烤肉拼盘", "毛肚", "牛肉卷"],
            "tags": ["聚餐场景", "高客单价", "社交属性强", "到餐为主"],
            "peak_hours": {"上午": 0.02, "下午": 0.08, "晚间": 0.70, "深夜": 0.20},
        },
        "supermarket": {
            "display_name": "超市便利",
            "bu": "闪购",
            "category_l1": "超市便利",
            "category_l2": "超市/便利店",
            "typical_skus": ["矿泉水整箱", "零食大礼包", "洗衣液", "纸巾"],
            "tags": ["刚需日用", "闪购主力", "多品类覆盖"],
            "peak_hours": {"上午": 0.15, "下午": 0.25, "晚间": 0.45, "深夜": 0.15},
        },
        "fitness": {
            "display_name": "运动健身",
            "bu": "乐生活",
            "category_l1": "运动健身",
            "category_l2": "健身中心",
            "typical_skus": ["月卡", "季卡", "私教体验课", "团课"],
            "tags": ["乐生活", "长期消费", "高客单价", "健康意识"],
            "peak_hours": {"上午": 0.10, "下午": 0.25, "晚间": 0.50, "深夜": 0.15},
        },
        "beauty": {
            "display_name": "丽人（美甲/美发/按摩）",
            "bu": "乐生活",
            "category_l1": "丽人",
            "category_l2": "美甲/美发/按摩",
            "typical_skus": ["美甲套餐", "剪发+护理", "肩颈按摩60分钟", "精油SPA"],
            "tags": ["乐生活", "体验型消费", "女性用户为主", "高客单价"],
            "peak_hours": {"上午": 0.05, "下午": 0.40, "晚间": 0.45, "深夜": 0.10},
        },
        "fruit_fresh": {
            "display_name": "水果生鲜",
            "bu": "闪购",
            "category_l1": "超市便利",
            "category_l2": "水果生鲜",
            "typical_skus": ["车厘子250g", "阳光玫瑰葡萄", "鲜切水果拼盘", "牛奶1L"],
            "tags": ["闪购", "健康消费", "季节性", "低客单价"],
            "peak_hours": {"上午": 0.10, "下午": 0.30, "晚间": 0.45, "深夜": 0.15},
        },
        "flash_buy_other": {
            "display_name": "酒水零食/闪购杂类",
            "bu": "闪购",
            "category_l1": "超市便利",
            "category_l2": "酒水零食",
            "typical_skus": ["啤酒6罐装", "坚果零食礼盒", "进口红酒", "酸奶整箱"],
            "tags": ["闪购", "囤货场景", "优惠敏感", "非刚需"],
            "peak_hours": {"上午": 0.05, "下午": 0.20, "晚间": 0.50, "深夜": 0.25},
        },
    }

    def __init__(self, data_dir: Optional[str] = None, random_state: int = 42):
        """
        参数:
            data_dir: 美团数据目录路径（含神券订单+行为序列xlsx）
            random_state: 随机种子
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.rng = np.random.RandomState(random_state)
        self._order_df: Optional[pd.DataFrame] = None
        self._behavior_df: Optional[pd.DataFrame] = None
        self._prototypes: Dict[str, MerchantPrototype] = {}

    def load_data(self, order_path: Optional[str] = None,
                  behavior_path: Optional[str] = None) -> None:
        """加载美团数据"""
        if order_path:
            self._order_df = pd.read_excel(order_path)
        elif self.data_dir:
            p = self.data_dir / "神券订单数据样例.xlsx"
            if p.exists():
                self._order_df = pd.read_excel(str(p))

        if behavior_path:
            self._behavior_df = pd.read_excel(behavior_path)
        elif self.data_dir:
            p = self.data_dir / "用户行为序列.xlsx"
            if p.exists():
                self._behavior_df = pd.read_excel(str(p))

    def build(self) -> Dict[str, MerchantPrototype]:
        """
        构建所有商家原型

        返回:
            Dict[prototype_id, MerchantPrototype]
        """
        # Step 1: 从订单数据聚合POI分类统计
        poi_stats = self._aggregate_poi_stats()

        # Step 2: 从行为序列提取典型店铺名
        shop_names = self._extract_shop_names()

        # Step 3: 合并构建原型
        for proto_id, meta in self.PROTOTYPE_META.items():
            # 查找属于此原型的POI分类
            matched_pois = [
                poi for poi, pid in self.PROTOTYPE_MAPPING.items()
                if pid == proto_id
            ]

            # 从订单统计中汇总
            matched_stats = []
            for poi in matched_pois:
                if poi in poi_stats:
                    matched_stats.append(poi_stats[poi])

            # 合并统计量
            if matched_stats:
                avg_price = np.mean([s["avg_price"] for s in matched_stats])
                median_price = np.median([s["median_price"] for s in matched_stats])
                min_price = min(s["min_price"] for s in matched_stats)
                max_price = max(s["max_price"] for s in matched_stats)
                avg_subsidy = np.mean([s["avg_subsidy"] for s in matched_stats])
                subsidy_rate = np.mean([s["subsidy_rate"] for s in matched_stats])
                n_orders = sum(s["n_orders"] for s in matched_stats)
            else:
                avg_price = 50.0
                median_price = 40.0
                min_price, max_price = 10.0, 200.0
                avg_subsidy = 7.0
                subsidy_rate = 0.15
                n_orders = 0

            # 典型店铺名
            typical_shops = []
            for poi in matched_pois:
                if poi in shop_names:
                    typical_shops.extend(shop_names[poi])
            # 去重
            typical_shops = list(dict.fromkeys(typical_shops))

            # 如果没有真实店铺名，用原型display_name代替
            if not typical_shops:
                typical_shops = [meta["display_name"] + "店"]

            proto = MerchantPrototype(
                prototype_id=proto_id,
                display_name=meta["display_name"],
                bu=meta["bu"],
                category_l1=meta["category_l1"],
                category_l2=meta["category_l2"],
                poi_categories=matched_pois,
                avg_order_price=round(avg_price, 1),
                price_range=(round(min_price, 1), round(max_price, 1)),
                median_order_price=round(median_price, 1),
                avg_subsidy=round(avg_subsidy, 1),
                subsidy_rate=round(subsidy_rate, 3),
                peak_hours=meta["peak_hours"],
                typical_shops=typical_shops[:10],  # 最多保留10个
                typical_skus=meta["typical_skus"],
                n_orders=n_orders,
                tags=meta["tags"],
            )

            self._prototypes[proto_id] = proto

        return self._prototypes

    def _aggregate_poi_stats(self) -> Dict[str, Dict[str, float]]:
        """从订单数据聚合各POI分类的统计量"""
        stats = {}

        if self._order_df is not None and "POI分类" in self._order_df.columns:
            for poi, group in self._order_df.groupby("POI分类"):
                stats[poi] = {
                    "avg_price": float(group["订单金额"].mean()),
                    "median_price": float(group["订单金额"].median()),
                    "min_price": float(group["订单金额"].min()),
                    "max_price": float(group["订单金额"].max()),
                    "avg_subsidy": float(group["美补金额"].mean()),
                    "subsidy_rate": float(group["美补金额"].mean() / max(group["订单金额"].mean(), 1)),
                    "n_orders": len(group),
                }

        return stats

    def _extract_shop_names(self) -> Dict[str, List[str]]:
        """从行为序列提取各POI分类的典型店铺名"""
        shop_names: Dict[str, List[str]] = {}

        if self._behavior_df is not None:
            click_shop = self._behavior_df[
                self._behavior_df["行为类型"] == "点击店铺"
            ]["具体内容"].tolist()

            for content in click_shop:
                # 解析格式: 店铺名(品类_二级分类,单均XX元)
                m = re.match(r'(.+?)\((.+?),(?:单均|价格)(\d+\.?\d*)元\)', str(content))
                if m:
                    shop_name = m.group(1).strip()
                    category = m.group(2).strip()
                    # 反查 category → POI分类
                    # category 格式: "美食_西餐", "甜点饮品_饮品", "餐饮_小吃快餐"
                    cat_parts = category.split("_")
                    cat_l2 = cat_parts[-1] if cat_parts else category

                    # 将品类映射到原型ID，再反向找POI分类
                    for poi, proto_id in self.PROTOTYPE_MAPPING.items():
                        if cat_l2 in poi or poi in cat_l2:
                            if poi not in shop_names:
                                shop_names[poi] = []
                            if shop_name not in shop_names[poi]:
                                shop_names[poi].append(shop_name)
                            break

        return shop_names

    def get_prototypes_by_bu(self, bu: str) -> List[MerchantPrototype]:
        """获取指定业务线的所有原型"""
        return [p for p in self._prototypes.values() if p.bu == bu]

    def get_prototype_summary_df(self) -> pd.DataFrame:
        """获取所有原型的汇总表"""
        rows = []
        for pid, p in self._prototypes.items():
            rows.append({
                "prototype_id": pid,
                "display_name": p.display_name,
                "bu": p.bu,
                "category_l1": p.category_l1,
                "avg_price": p.avg_order_price,
                "median_price": p.median_order_price,
                "avg_subsidy": p.avg_subsidy,
                "subsidy_rate": p.subsidy_rate,
                "n_orders": p.n_orders,
                "typical_shops": "、".join(p.typical_shops[:3]),
                "typical_skus": "、".join(p.typical_skus[:3]),
                "tags": "、".join(p.tags),
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            pid: {
                "prototype_id": p.prototype_id,
                "display_name": p.display_name,
                "bu": p.bu,
                "category_l1": p.category_l1,
                "category_l2": p.category_l2,
                "poi_categories": p.poi_categories,
                "avg_order_price": p.avg_order_price,
                "price_range": list(p.price_range),
                "median_order_price": p.median_order_price,
                "avg_subsidy": p.avg_subsidy,
                "subsidy_rate": p.subsidy_rate,
                "peak_hours": p.peak_hours,
                "typical_shops": p.typical_shops,
                "typical_skus": p.typical_skus,
                "n_orders": p.n_orders,
                "tags": p.tags,
            }
            for pid, p in self._prototypes.items()
        }


# ===========================================================================
# 全局商家注册表
# ===========================================================================

class MerchantRegistry:
    """
    全局商家注册表——仿真过程中按品类分配商家原型

    替代原来的 category_preference（5维粗粒度），
    使用10个商家原型实现更真实的场景生成。
    """

    def __init__(self, prototypes: Optional[Dict[str, MerchantPrototype]] = None,
                 random_state: int = 42):
        self.prototypes = prototypes or {}
        self.rng = np.random.RandomState(random_state)
        self._proto_list: List[MerchantPrototype] = []
        self._proto_weights: np.ndarray = np.array([])

        if prototypes:
            self._build_sampling_index()

    def _build_sampling_index(self) -> None:
        """构建采样索引（按订单量加权）"""
        self._proto_list = list(self.prototypes.values())
        weights = np.array([max(p.n_orders, 1) for p in self._proto_list], dtype=float)
        self._proto_weights = weights / weights.sum()

    def register(self, prototype: MerchantPrototype) -> None:
        """注册一个商家原型"""
        self.prototypes[prototype.prototype_id] = prototype
        self._build_sampling_index()

    def sample(self, bu: Optional[str] = None,
               category_l1: Optional[str] = None) -> MerchantPrototype:
        """
        采样一个商家原型

        参数:
            bu: 限定业务线（如 "外卖"）
            category_l1: 限定一级品类

        返回:
            采样到的商家原型
        """
        candidates = list(self.prototypes.values())
        if bu:
            candidates = [p for p in candidates if p.bu == bu]
        if category_l1:
            candidates = [p for p in candidates if p.category_l1 == category_l1]

        if not candidates:
            candidates = list(self.prototypes.values())

        # 按订单量加权采样
        weights = np.array([max(p.n_orders, 1) for p in candidates], dtype=float)
        weights = weights / weights.sum()

        idx = self.rng.choice(len(candidates), p=weights)
        return candidates[idx]

    def sample_for_user(self, user_income_level: int,
                        user_category_preference: Optional[np.ndarray] = None) -> MerchantPrototype:
        """
        为特定用户采样商家原型

        考虑用户收入水平（高收入→高客单价品类概率更大）
        和品类偏好（如偏好分布）

        参数:
            user_income_level: 用户收入等级 1-5
            user_category_preference: 品类偏好分布 (10维, 对应10个原型)
        """
        candidates = list(self.prototypes.values())

        if user_category_preference is not None and len(user_category_preference) == len(candidates):
            # 使用用户品类偏好
            probs = user_category_preference / user_category_preference.sum()
        else:
            # 基于收入水平调整：高收入→高客单价品类
            base_weights = np.array([max(p.n_orders, 1) for p in candidates], dtype=float)
            price_boost = np.array([
                np.exp(0.1 * (user_income_level - 3) * (p.avg_order_price / 100 - 0.5))
                for p in candidates
            ])
            weights = base_weights * price_boost
            probs = weights / weights.sum()

        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]

    def get_all_ids(self) -> List[str]:
        """获取所有原型ID"""
        return list(self.prototypes.keys())

    def get(self, prototype_id: str) -> Optional[MerchantPrototype]:
        """按ID获取原型"""
        return self.prototypes.get(prototype_id)


# ===========================================================================
# 便捷函数：从美团数据快速构建注册表
# ===========================================================================

def build_merchant_registry(data_dir: str = "data",
                            random_state: int = 42) -> MerchantRegistry:
    """
    从美团数据构建商家注册表（一行代码搞定）

    参数:
        data_dir: 美团数据目录路径
        random_state: 随机种子

    返回:
        MerchantRegistry 实例
    """
    factory = MerchantPrototypeFactory(data_dir=data_dir, random_state=random_state)
    factory.load_data()
    prototypes = factory.build()
    registry = MerchantRegistry(prototypes=prototypes, random_state=random_state)
    return registry


def build_merchant_registry_from_data(
    order_path: str,
    behavior_path: str,
    random_state: int = 42,
) -> MerchantRegistry:
    """
    从指定数据文件路径构建商家注册表

    参数:
        order_path: 神券订单Excel路径
        behavior_path: 用户行为序列Excel路径
        random_state: 随机种子

    返回:
        MerchantRegistry 实例
    """
    factory = MerchantPrototypeFactory(random_state=random_state)
    factory.load_data(order_path=order_path, behavior_path=behavior_path)
    prototypes = factory.build()
    registry = MerchantRegistry(prototypes=prototypes, random_state=random_state)
    return registry
