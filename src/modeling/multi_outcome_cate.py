"""
多结果 CATE 估计模块（行为链各步骤）

核心思路（Option A + Option C 混合）：
  1. Option A（独立模型）：对行为链每一步独立训练一个 CATE 模型（X-Learner/DR-Learner）
     - 优点：与现有 CausalMLWrapper 接口兼容；快速验证
     - 缺点：忽略步骤间相关性
  2. Option C（序列 CATE，可选）：仅用前序步骤=1 的子群估计后续步骤的 CATE
     - 优点：因果识别更严谨（控制选择偏差）
     - 默认关闭，通过 sequential=True 启用

用法示例：
    estimator = MultiOutcomeCATE(
        outcome_names=["clicked", "carted", "paid", "redeemed"],
        learner_type="xlearner",
    )
    result = estimator.fit_predict(bc_df, feature_cols, treatment_col="treatment")
    # result 是包含 cate_clicked / cate_carted / ... 列的 DataFrame

参考文献：
- Chernozhukov et al. (2018) Double/debiased machine learning for treatment and causal parameters.
  Econometrics Journal, 21(1), C1-C68.
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification and Sensitivity Analysis
  for Multiple Causal Mechanisms. Political Analysis, 18(4), 455-470.
- Li & Kannan (2016). Attributing Conversions in a Multichannel Online Advertising Environment.
  Marketing Science, 35(3), 457-481.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

from src.modeling.causalml_wrapper import CausalMLWrapper, CausalMLConfig


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class MultiOutcomeCATE:
    """
    行为链多结果异质性处理效应估计

    为行为链每一步（clicked / carted / paid / redeemed）分别估计 CATE，
    支持独立模型（Option A）和序列子群模型（Option C）。

    参数
    ----------
    outcome_names : list[str]
        行为链步骤名称列表，按先后顺序，如 ["clicked", "carted", "paid", "redeemed"]。
    learner_type : str
        CATE 学习器类型，支持 "xlearner" | "drlearner" | "tlearner"。
    sequential : bool
        True 时启用 Option C（序列 CATE）：每步只用前序=1 的子群估计。
        False 时使用 Option A（全量独立估计）。
    """

    def __init__(
        self,
        outcome_names: Optional[List[str]] = None,
        learner_type: str = "xlearner",
        sequential: bool = False,
    ):
        if outcome_names is None:
            outcome_names = ["clicked", "carted", "paid", "redeemed"]
        self.outcome_names = outcome_names
        self.learner_type = learner_type
        self.sequential = sequential

        self._wrappers: Dict[str, CausalMLWrapper] = {}
        self._feature_cols: List[str] = []
        self._treatment_col: str = "treatment"
        self.cate_estimates: Dict[str, np.ndarray] = {}
        self.ate_estimates: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        treatment_col: str = "treatment",
    ) -> pd.DataFrame:
        """
        对行为链各步骤分别拟合 CATE 模型，并返回带 CATE 列的 DataFrame。

        返回的 DataFrame 新增列：
          cate_{step}  — 每个用户在该步骤的估计 CATE
          ate_{step}   — 该步骤的平均处理效应（ATE）

        参数
        ----------
        df : pd.DataFrame
            行为链数据，需包含 feature_cols + treatment_col + outcome_names 列。
        feature_cols : list[str]
            用户特征列名。
        treatment_col : str
            处理变量列名。

        返回
        -------
        pd.DataFrame
            原始 df 附加 cate_{step} 列。
        """
        self._feature_cols = feature_cols
        self._treatment_col = treatment_col
        result_df = df.copy()

        for i, step in enumerate(self.outcome_names):
            if step not in df.columns:
                warnings.warn(f"[MultiOutcomeCATE] 列 '{step}' 不在 df 中，跳过。")
                continue

            if self.sequential and i > 0:
                # Option C：仅用前一步=1 的子群
                prev_step = self.outcome_names[i - 1]
                sub_df = df[df[prev_step] == 1].copy()
                if len(sub_df) < 50:
                    warnings.warn(
                        f"[MultiOutcomeCATE] 步骤 '{step}' 序列子群仅 {len(sub_df)} 行，"
                        f"回退到全量估计。"
                    )
                    sub_df = df.copy()
            else:
                # Option A：全量估计
                sub_df = df.copy()

            print(f"  [{step}] 训练 {self.learner_type} (n={len(sub_df)})", end="", flush=True)
            cate_arr, ate = self._fit_one_step(sub_df, feature_cols, treatment_col, step)
            print(f" → ATE={ate:.4f}")

            self.cate_estimates[step] = cate_arr
            self.ate_estimates[step] = ate

            # 对齐回原始 df（序列子群时 cate_arr 可能比 df 短）
            if len(cate_arr) == len(df):
                result_df[f"cate_{step}"] = cate_arr
            else:
                # 仅子群有值，其他行填 0
                cate_full = np.zeros(len(df))
                cate_full[sub_df.index] = cate_arr
                result_df[f"cate_{step}"] = cate_full

        return result_df

    # ------------------------------------------------------------------
    # 单步 CATE 估计
    # ------------------------------------------------------------------

    def _fit_one_step(
        self,
        sub_df: pd.DataFrame,
        feature_cols: List[str],
        treatment_col: str,
        outcome_col: str,
    ):
        """拟合单步 CATE，返回 (cate_array, ate_float)"""
        config = CausalMLConfig(learner_type=self.learner_type)
        wrapper = CausalMLWrapper(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapper.fit_predict(sub_df, feature_cols, treatment_col, outcome_col)

        self._wrappers[outcome_col] = wrapper
        cate_arr = wrapper.cate_estimates
        ate = float(np.mean(cate_arr))
        return cate_arr, ate

    # ------------------------------------------------------------------
    # 预测（推断阶段）
    # ------------------------------------------------------------------

    def predict_cate(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        对新数据预测各步骤 CATE（需先调用 fit_predict）。

        返回 {step_name: cate_array}。
        """
        if not self._wrappers:
            raise RuntimeError("请先调用 fit_predict() 拟合模型。")

        results = {}
        for step, wrapper in self._wrappers.items():
            cate_arr = wrapper.predict_cate(df, self._feature_cols)
            results[step] = cate_arr
        return results

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate_all_steps(
        self,
        df: pd.DataFrame,
        true_cate_prefix: str = "true_cate_",
    ) -> Dict[str, Dict]:
        """
        对各步骤 CATE 质量做评估（若有真实 CATE 列）。

        返回 {step: {"corr": float, "mae": float, "ate": float}}
        """
        from scipy.stats import pearsonr
        import warnings

        results = {}
        for step in self.outcome_names:
            true_col = f"{true_cate_prefix}{step}"
            cate_col = f"cate_{step}"
            if cate_col not in df.columns:
                continue

            info: Dict = {"ate": self.ate_estimates.get(step, None)}
            if true_col in df.columns:
                cate_pred = df[cate_col].values
                cate_true = df[true_col].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = pearsonr(cate_pred, cate_true)
                mae = float(np.mean(np.abs(cate_pred - cate_true)))
                info.update({"corr": round(float(corr), 4), "mae": round(mae, 4)})
            results[step] = info
        return results

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def get_weighted_cate(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        计算加权综合 CATE 评分（用于补贴 Agent 排序）。

        默认权重：click=0.35, cart=0.15, pay=0.15, redeem=0.35
        """
        default_weights = {
            "clicked": 0.35,
            "carted":  0.15,
            "paid":    0.15,
            "redeemed": 0.35,
        }
        if weights is None:
            weights = default_weights

        weighted = np.zeros(len(df))
        total_w = 0.0
        for step, w in weights.items():
            col = f"cate_{step}"
            if col in df.columns:
                weighted += w * df[col].values
                total_w += w

        if total_w > 0:
            weighted /= total_w
        return weighted

    def summary(self) -> pd.DataFrame:
        """输出各步骤 ATE 摘要 DataFrame"""
        rows = []
        for step in self.outcome_names:
            rows.append({
                "step": step,
                "ate": round(self.ate_estimates.get(step, float("nan")), 4),
                "n_fitted": len(self.cate_estimates.get(step, [])),
            })
        return pd.DataFrame(rows)
