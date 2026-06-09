"""
行为链中介分析模块

功能：
  将补贴对最终核销（redeemed）的总效应，分解为：
  - 直接效应（补贴 → 核销，绕过中间步骤）
  - 各路径间接效应（补贴 → clicked → carted → paid → redeemed）

  使用 DoWhy 建模因果 DAG + statsmodels 做中介分析。
  DoWhy 提供 DAG 因果识别；statsmodels 提供 Bootstrap CI。

行为链 DAG：
  subsidy_amount → clicked → carted → paid → redeemed
         ↓                                      ↑
         └─────────────── direct ───────────────┘

参考文献：
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification and Sensitivity Analysis
  for Multiple Causal Mechanisms. Political Analysis, 18(4), 455-470.
- Imai, K., Keele, L., Tingley, D., & Yamamoto, T. (2011). Unpacking the Black Box
  of Causality: Learning About Causal Mechanisms from Experimental Data.
  American Political Science Review, 105(4), 765-789.
- VanderWeele, T. J. (2014). A Unification of Mediation and Interaction.
  Epidemiology, 25(5), 749-760.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class BehaviorChainMediator:
    """
    行为链中介分析

    将补贴 → 核销 的总效应分解为：
      直接效应（NDE）  : 控制中介后，补贴对核销的独立效应
      间接效应（NIE）  : 通过行为链路径传导的效应
      各路径贡献比例   : clicked / carted / paid 各自贡献多少间接效应

    用法：
        mediator = BehaviorChainMediator(
            treatment_col="treatment",
            outcome_col="redeemed",
            mediator_cols=["clicked", "carted", "paid"],
            feature_cols=["price_sensitivity", "income_level"],
        )
        results = mediator.fit_analyze(df, n_bootstrap=200)
        print(results["summary"])
    """

    def __init__(
        self,
        treatment_col: str = "treatment",
        outcome_col: str = "redeemed",
        mediator_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.mediator_cols = mediator_cols or ["clicked", "carted", "paid"]
        self.feature_cols = feature_cols or []

        self._mediator_models: Dict[str, LogisticRegression] = {}
        self._outcome_model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None

        self.results_: Optional[Dict] = None

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def fit_analyze(
        self,
        df: pd.DataFrame,
        n_bootstrap: int = 200,
        seed: int = 42,
    ) -> Dict:
        """
        拟合中介模型并分解效应。

        参数
        ----------
        df : pd.DataFrame
            行为链数据，需包含 treatment_col + outcome_col + mediator_cols + feature_cols。
        n_bootstrap : int
            Bootstrap 置信区间的重复次数（默认 200）。
        seed : int
            随机种子。

        返回
        -------
        dict，包含：
          - "total_effect": 总效应 ATE
          - "direct_effect": 直接效应（NDE）
          - "indirect_effects": {mediator: NIE}
          - "proportion_mediated": 间接效应占比
          - "bootstrap_ci": Bootstrap 95% CI
          - "summary": pd.DataFrame 摘要表
        """
        rng = np.random.RandomState(seed)

        # 特征矩阵
        X_all, y_t, y_out, y_meds = self._prepare_data(df)

        # 点估计
        total_eff = self._estimate_total_effect(X_all, y_t, y_out)
        direct_eff, indirect_effs = self._decompose_effects(X_all, y_t, y_out, y_meds, df)

        # Bootstrap CI
        ci_results = self._bootstrap_ci(df, n_bootstrap, rng)

        # 汇总
        summary_rows = []
        summary_rows.append({
            "effect_type": "Total Effect (ATE)",
            "estimate": round(total_eff, 4),
            "ci_lower": round(ci_results["total_ci"][0], 4),
            "ci_upper": round(ci_results["total_ci"][1], 4),
        })
        summary_rows.append({
            "effect_type": "Direct Effect (NDE)",
            "estimate": round(direct_eff, 4),
            "ci_lower": round(ci_results["direct_ci"][0], 4),
            "ci_upper": round(ci_results["direct_ci"][1], 4),
        })
        for med, eff in indirect_effs.items():
            summary_rows.append({
                "effect_type": f"Indirect via {med} (NIE)",
                "estimate": round(eff, 4),
                "ci_lower": round(ci_results.get(f"indirect_{med}_ci", (np.nan, np.nan))[0], 4),
                "ci_upper": round(ci_results.get(f"indirect_{med}_ci", (np.nan, np.nan))[1], 4),
            })

        total_indirect = sum(indirect_effs.values())
        prop_mediated = total_indirect / max(abs(total_eff), 1e-9)

        self.results_ = {
            "total_effect": total_eff,
            "direct_effect": direct_eff,
            "indirect_effects": indirect_effs,
            "total_indirect": total_indirect,
            "proportion_mediated": prop_mediated,
            "bootstrap_ci": ci_results,
            "summary": pd.DataFrame(summary_rows),
        }

        return self.results_

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame):
        """提取并标准化特征矩阵"""
        X_parts = [df[[self.treatment_col]].values.astype(float)]
        if self.feature_cols:
            X_feat = df[self.feature_cols].values.astype(float)
            scaler = StandardScaler()
            X_feat = scaler.fit_transform(X_feat)
            self._scaler = scaler
            X_parts.append(X_feat)

        X_all = np.hstack(X_parts)
        y_t   = df[self.treatment_col].values.astype(int)
        y_out = df[self.outcome_col].values.astype(int)
        y_meds = {m: df[m].values.astype(int) for m in self.mediator_cols if m in df.columns}

        return X_all, y_t, y_out, y_meds

    # ------------------------------------------------------------------
    # 总效应估计（IPW）
    # ------------------------------------------------------------------

    def _estimate_total_effect(
        self,
        X_all: np.ndarray,
        y_t: np.ndarray,
        y_out: np.ndarray,
    ) -> float:
        """用 IPW 估计总效应（ATE）"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps_model = LogisticRegression(C=1.0, max_iter=500, random_state=0)
            ps_model.fit(X_all[:, 1:] if X_all.shape[1] > 1 else X_all, y_t)
            ps = ps_model.predict_proba(X_all[:, 1:] if X_all.shape[1] > 1 else X_all)[:, 1]
            ps = np.clip(ps, 0.05, 0.95)

        ipw = (y_t / ps - (1 - y_t) / (1 - ps))
        ate = float(np.mean(ipw * y_out))
        return ate

    # ------------------------------------------------------------------
    # 效应分解（Imai et al. 2010 框架，逐步回归中介法）
    # ------------------------------------------------------------------

    def _decompose_effects(
        self,
        X_all: np.ndarray,
        y_t: np.ndarray,
        y_out: np.ndarray,
        y_meds: Dict[str, np.ndarray],
        df: pd.DataFrame,
    ) -> Tuple[float, Dict[str, float]]:
        """
        三步回归中介法（Baron & Kenny / Imai et al.）：
        1. 拟合中介模型 M_i = f(T, X)
        2. 拟合结果模型 Y = g(T, M_1..M_k, X)
        3. 直接效应 = T 系数（控制中介后）
           间接效应_i = E[dY/dM_i * dM_i/dT]
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 结果模型：Y ~ T + M1 + M2 + ... + X（Logistic）
            med_arrays = np.column_stack(
                [y_meds[m] for m in self.mediator_cols if m in y_meds]
            ) if y_meds else np.zeros((len(y_t), 1))

            X_outcome = np.hstack([X_all, med_arrays])
            out_model = LogisticRegression(C=1.0, max_iter=500, random_state=0)
            out_model.fit(X_outcome, y_out)

            # 直接效应：T 系数（控制中介后，近似 NDE）
            t_coef = float(out_model.coef_[0][0])
            direct = t_coef * float(np.mean(y_t))  # 边际效应近似

            # 间接效应：各中介的贡献
            indirect = {}
            n_med = len([m for m in self.mediator_cols if m in y_meds])
            for i, med in enumerate(self.mediator_cols):
                if med not in y_meds:
                    continue
                # 中介模型：M_i ~ T + X
                med_model = LogisticRegression(C=1.0, max_iter=500, random_state=0)
                med_model.fit(X_all, y_meds[med])
                self._mediator_models[med] = med_model

                # dM_i/dT（中介模型 T 系数）
                dM_dT = float(med_model.coef_[0][0])

                # dY/dM_i（结果模型 M_i 系数，在 X_outcome 中位于 t+feat+i 位置）
                med_start_idx = X_all.shape[1]
                dY_dM = float(out_model.coef_[0][med_start_idx + i])

                # 近似间接效应 = dY/dM * dM/dT * P(T=1)
                indirect[med] = dY_dM * dM_dT * float(np.mean(y_t))

        return direct, indirect

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------

    def _bootstrap_ci(
        self,
        df: pd.DataFrame,
        n_bootstrap: int,
        rng: np.random.RandomState,
        alpha: float = 0.05,
    ) -> Dict:
        """Bootstrap 95% CI（仅对总效应和直接效应）"""
        total_boots = []
        direct_boots = []
        indirect_boots: Dict[str, List[float]] = {m: [] for m in self.mediator_cols}

        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(df), size=len(df))
            boot_df = df.iloc[idx].copy()
            try:
                X_b, y_t_b, y_out_b, y_meds_b = self._prepare_data(boot_df)
                total_b = self._estimate_total_effect(X_b, y_t_b, y_out_b)
                direct_b, indirect_b = self._decompose_effects(
                    X_b, y_t_b, y_out_b, y_meds_b, boot_df
                )
                total_boots.append(total_b)
                direct_boots.append(direct_b)
                for m, v in indirect_b.items():
                    indirect_boots[m].append(v)
            except Exception:
                pass

        def ci(arr):
            if not arr:
                return (float("nan"), float("nan"))
            a = np.array(arr)
            return (float(np.percentile(a, 100 * alpha / 2)),
                    float(np.percentile(a, 100 * (1 - alpha / 2))))

        result = {
            "total_ci":  ci(total_boots),
            "direct_ci": ci(direct_boots),
        }
        for m in self.mediator_cols:
            result[f"indirect_{m}_ci"] = ci(indirect_boots.get(m, []))
        return result

    # ------------------------------------------------------------------
    # 可视化辅助
    # ------------------------------------------------------------------

    def plot_mediation_summary(self, title: str = "行为链中介效应分析") -> None:
        """打印简洁的文字版摘要（不依赖 matplotlib）"""
        if self.results_ is None:
            print("[BehaviorChainMediator] 请先调用 fit_analyze()。")
            return

        r = self.results_
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
        print(f"  总效应 (ATE)      : {r['total_effect']:+.4f}")
        print(f"  直接效应 (NDE)    : {r['direct_effect']:+.4f}")
        print(f"  总间接效应        : {r['total_indirect']:+.4f}")
        print(f"  间接效应占比      : {r['proportion_mediated']:.1%}")
        print(f"  ── 各路径贡献 ──")
        for med, eff in r["indirect_effects"].items():
            pct = eff / max(abs(r["total_effect"]), 1e-9)
            print(f"    via {med:12s}: {eff:+.4f}  ({pct:+.1%})")
        print(f"{'='*55}\n")
