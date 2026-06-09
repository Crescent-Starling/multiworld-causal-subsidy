"""
CausalML集成模块
与现有手工T/X/DR-Learner对比，验证结果鲁棒性

参考文献：
- CausalML (Uber): https://github.com/uber/causalml
- Athey & Wager (2018): Estimation and Inference of Heterogeneous Treatment Effects
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# CausalML imports - 使用正确的API
from causalml.inference.meta import BaseTLearner, BaseXLearner, BaseDRRegressor
from causalml.inference.meta import BaseSLearner
from causalml.metrics import auuc_score, qini_score


# ============================================================
# 辅助函数：手动实现 AUUC / Qini 计算（不依赖 CausalML 的 auuc_score）
# ============================================================

def compute_auuc(df: pd.DataFrame, cate_col: str, outcome_col: str = "y",
                  treatment_col: str = "w", normalize: bool = True) -> float:
    """
    手动计算 AUUC (Area Under the Uplift Curve)
    
    逻辑：
    1. 按 CATE 降序排列用户
    2. 计算累积处理增益（cumulative treatment gain）
    3. 计算曲线下面积
    
    参数：
    - df: DataFrame，必须包含 cate_col, outcome_col, treatment_col
    - cate_col: CATE 预测值列名
    - outcome_col: 结果变量列名（0/1）
    - treatment_col: 处理指示列名（0/1）
    - normalize: 是否归一化（归一化后随机基线 ≈ 0.5）
    
    返回：
    - auuc: float，AUUC 值（越高越好）
    """
    # 按 CATE 降序排列
    sorted_df = df.sort_values(cate_col, ascending=False).reset_index(drop=True)
    sorted_df.index = sorted_df.index + 1  # 1-based index
    
    # 累积计算：处理组转化率 - 对照组转化率
    sorted_df["cum_tr"] = sorted_df[treatment_col].cumsum()
    sorted_df["cum_ct"] = (sorted_df.index.values - sorted_df["cum_tr"])
    sorted_df["cum_y_tr"] = (sorted_df[outcome_col] * sorted_df[treatment_col]).cumsum()
    sorted_df["cum_y_ct"] = (sorted_df[outcome_col] * (1 - sorted_df[treatment_col])).cumsum()
    
    # 避免除零
    gain = (sorted_df["cum_y_tr"] / sorted_df["cum_tr"].replace(0, np.nan) -
            sorted_df["cum_y_ct"] / sorted_df["cum_ct"].replace(0, np.nan))
    gain = gain.fillna(0)
    
    if normalize:
        # 归一化：除以最大可能增益（全部是处理组且全部转化）
        gain = gain / max(gain.max(), 1e-6)
    
    # 计算曲线下面积（梯形法）
    n = len(gain)
    auuc = np.trapz(gain.values, dx=1.0 / n)
    
    return float(auuc)


def compute_qini(df: pd.DataFrame, cate_col: str, outcome_col: str = "y",
                  treatment_col: str = "w", normalize: bool = True) -> float:
    """
    手动计算 Qini 系数
    
    Qini = AUUC 的变体，用 N_treated 而非 N_total 作为 x 轴
    """
    sorted_df = df.sort_values(cate_col, ascending=False).reset_index(drop=True)
    sorted_df.index = sorted_df.index + 1
    
    sorted_df["cum_tr"] = sorted_df[treatment_col].cumsum()
    sorted_df["cum_ct"] = sorted_df.index.values - sorted_df["cum_tr"]
    sorted_df["cum_y_tr"] = (sorted_df[outcome_col] * sorted_df[treatment_col]).cumsum()
    sorted_df["cum_y_ct"] = (sorted_df[outcome_col] * (1 - sorted_df[treatment_col])).cumsum()
    
    # Qini: x 轴是 cum_tr（处理组累积人数）
    qini_gain = (sorted_df["cum_y_tr"] / sorted_df["cum_tr"].replace(0, np.nan) -
                  sorted_df["cum_y_ct"] / sorted_df["cum_ct"].replace(0, np.nan))
    qini_gain = qini_gain.fillna(0)
    
    if normalize:
        qini_gain = qini_gain / max(qini_gain.max(), 1e-6)
    
    # x 轴用 cum_tr（处理组累积人数）
    x_axis = sorted_df["cum_tr"].values
    valid = x_axis > 0
    if valid.sum() < 2:
        return 0.0
    qini = np.trapz(qini_gain.values[valid], x_axis[valid])
    qini = qini / max(x_axis[valid].max(), 1)  # 归一化到 [0, 1]
    
    return float(qini)


@dataclass
class CausalMLConfig:
    """CausalML配置"""
    learner_type: str = "tlearner"  # "tlearner", "xlearner", "drlearner", "slearner"
    base_model: str = "rfortune"  # "rfortune", "xgboost", "lightgbm"
    n_folds: int = 5
    random_state: int = 42


class CausalMLWrapper:
    """
    CausalML包装器
    
    用途：
    1. 训练CausalML模型（T/X/DR/S-Learner）
    2. 预测CATE（个体处理效应）
    3. 与现有手工实现对比
    """
    
    def __init__(self, config: Optional[CausalMLConfig] = None):
        self.config = config or CausalMLConfig()
        self.model = None
        self.cate_estimates = None
        self.feature_cols = None  # 保存特征列名用于绘图
        
    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        outcome_col: str = "outcome"
    ) -> pd.DataFrame:
        """
        训练模型并预测CATE
        
        参数：
        - df: 包含特征、处理、结果的DataFrame
        - feature_cols: 特征列名
        - treatment_col: 处理列名（0/1）
        - outcome_col: 结果列名
        
        返回：
        - DataFrame with CATE estimates
        """
        self.feature_cols = feature_cols  # 保存用于后续绘图
        X = df[feature_cols].values
        treatment = df[treatment_col].values
        outcome = df[outcome_col].values
        
        # 根据配置选择Learner
        if self.config.learner_type == "tlearner":
            self.model = BaseTLearner(learner=self._get_base_model())
        elif self.config.learner_type == "xlearner":
            self.model = BaseXLearner(learner=self._get_base_model())
        elif self.config.learner_type == "drlearner":
            # 修复：BaseDRBabyYunzheng → BaseDRRegressor
            self.model = BaseDRRegressor(learner=self._get_base_model())
        elif self.config.learner_type == "slearner":
            self.model = BaseSLearner(learner=self._get_base_model())
        else:
            raise ValueError(f"Unknown learner_type: {self.config.learner_type}")
        
        # 拟合模型
        # 注意：CausalML 不同 Learner 的 fit/predict 签名有差异
        #
        # BaseTLearner / BaseXLearner / BaseSLearner:
        #   fit(X, treatment, y) -> predict(X)
        #
        # BaseDRRegressor:
        #   fit(X, treatment, y, p=None)  -> predict(X)
        #   p: 倾向得分（propensity score），若不提供则内部用 ElasticNet 估计
        #   ⚠️ 内部估计的 p 可能接近 0/1，导致 DR 公式数值爆炸
        #       解决方案：预先计算 p，做 clipping，再传入 fit()
        if self.config.learner_type == "drlearner":
            # 预先计算倾向得分（用 LogisticRegression CV），并做 clipping
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LogisticRegressionCV(cv=5, random_state=self.config.random_state, max_iter=1000)
            lr.fit(X_scaled, treatment)
            p_raw = lr.predict_proba(X_scaled)[:, 1]
            # Clipping: 将倾向得分限制在 [0.05, 0.95]，避免 DR 公式分母接近 0
            p_clipped = np.clip(p_raw, 0.05, 0.95)
            self._propensity_scores = p_clipped  # 保存供诊断
            self.model.fit(X, treatment, outcome, p=p_clipped)
        else:
            self.model.fit(X, treatment, outcome)

        # 预测CATE（所有Learner 统一用 predict(X)）
        cate_pred = self.model.predict(X)

        # CausalML的predict返回shape可能是 (n, 1) 或 (n,)，需要统一squeeze
        self.cate_estimates = np.array(cate_pred).flatten()
        
        # 返回结果DataFrame
        result_df = df.copy()
        result_df["cate_causalml"] = self.cate_estimates
        
        return result_df

    def predict_cate(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """
        在新数据上预测 CATE（无需重新训练）
        
        参数：
        - df: 包含特征的 DataFrame
        - feature_cols: 特征列名
        
        返回：
        - cate_pred: ndarray, shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")
        
        X = df[feature_cols].values
        cate_pred = self.model.predict(X)
        return np.array(cate_pred).flatten()
    
    def _get_base_model(self):
        """获取基模型"""
        if self.config.base_model == "rfortune":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        elif self.config.base_model == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=self.config.random_state)
        elif self.config.base_model == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(random_state=self.config.random_state, verbose=-1)
        else:
            raise ValueError(f"Unknown base_model: {self.config.base_model}")
    
    def evaluate_cate_quality(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        true_cate_col: Optional[str] = "true_cate",
        feature_cols: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        评估 CATE 估计质量（AUUC / Qini / 与 true CATE 的相关性）
        
        参数：
        - df: 包含 treatment, outcome, features 的 DataFrame
        - treatment_col: 处理列名
        - outcome_col: 结果列名
        - true_cate_col: 真实 CATE 列名（合成数据有 ground truth 时用）
        - feature_cols: 特征列名（用于在新数据上预测 CATE）
                     如果为 None，则使用 self.feature_cols（训练时的特征）
        
        返回：
        - Dict，含 AUUC score, Qini score, CATE 精度指标
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")
        
        # 在新数据上预测 CATE
        if feature_cols is None:
            if self.feature_cols is None:
                raise ValueError("feature_cols not provided and not available from training.")
            feature_cols = self.feature_cols
        
        X_new = df[feature_cols].values
        cate_pred_new = self.model.predict(X_new)  # 在新数据上预测CATE
        cate_pred_new = np.array(cate_pred_new).flatten()
        
        # ============================================================
        # 使用 CausalML 的 auuc_score / qini_score
        #
        # 关键：df 中不能有名为 'tau' 的列
        #   （因为 treatment_effect_col 默认值是 'tau'，
        #     如果 df 中有 'tau' 列，get_cumlift 会把它从 model_names 中排除，
        #     导致 model_names=[] → pd.concat([]) 报错）
        #
        # 解决方案：
        #   - CATE 预测列名叫 'cate_pred'（不是 'tau'）
        #   - 不传 treatment_effect_col 参数
        #     （让它用默认值 'tau'，而 'tau' 不在 df 中 → 走 else 分支）
        # ============================================================
        
        results = {}

        # 1. AUUC Score / Qini Score（越高越好）
        try:
            # 构造 CausalML metrics 所需的 DataFrame 格式
            # 列名不能用 'tau'！！！
            eval_df = pd.DataFrame({
                "y": df[outcome_col].values,
                "w": df[treatment_col].values,
                "cate_pred": cate_pred_new,  # 列名 != 'tau' ✓
            })
            
            if true_cate_col and true_cate_col in df.columns:
                # 有 true CATE：把它也作为一列（列名也不能是 'tau'）
                eval_df["cate_true"] = df[true_cate_col].values
            
            # 调用 auuc_score
            # 不传 treatment_effect_col → 用默认 'tau'（不在 df 中）→ 正确
            auuc_series = auuc_score(eval_df, outcome_col="y", treatment_col="w")
            qini_series = qini_score(eval_df, outcome_col="y", treatment_col="w")
            
            # auuc_score 返回 Series，index=模型名（即 CATE 列名）
            results["auuc"] = float(auuc_series["cate_pred"])
            results["qini"] = float(qini_series["cate_pred"])
            
            if true_cate_col and true_cate_col in df.columns:
                results["auuc_true"] = float(auuc_series.get("cate_true", np.nan))
                results["qini_true"] = float(qini_series.get("cate_true", np.nan))
            
        except Exception as e:
            results["auuc"] = None
            results["qini"] = None
            results["auuc_error"] = str(e)

        # 2. 与 true CATE 的对比（合成数据专用）
        if true_cate_col and true_cate_col in df.columns:
            true_cate = df[true_cate_col].values
            from scipy.stats import pearsonr
            corr, _ = pearsonr(cate_pred_new, true_cate)
            results["corr"] = float(corr)
            results["cate_mse"] = float(np.mean((cate_pred_new - true_cate) ** 2))
            results["cate_mae"] = float(np.mean(np.abs(cate_pred_new - true_cate)))
            results["ate_true"] = float(np.mean(true_cate))
            results["ate_estimated"] = float(np.mean(cate_pred_new))
            results["ate_bias"] = float(np.mean(cate_pred_new) - np.mean(true_cate))
            results["ate_bias_pct"] = float(
                (np.mean(cate_pred_new) - np.mean(true_cate)) / max(abs(np.mean(true_cate)), 1e-6)
            )

        # 3. CATE 分布诊断（使用新预测的CATE）
        results["ate_mean"] = float(np.mean(cate_pred_new))
        results["ate_std"] = float(np.std(cate_pred_new))
        results["cate_min"] = float(np.min(cate_pred_new))
        results["cate_max"] = float(np.max(cate_pred_new))
        results["positive_cate_ratio"] = float(
            np.mean(cate_pred_new > 0)
        )
        
        return results
    
    def plot_qini_curve(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        save_path: Optional[str] = None,
    ):
        """
        绘制 Qini Curve（手动实现，不依赖 CausalML 的 plot_qini）

        Qini Curve 定义：
        - x 轴：按 CATE 排序后补贴的用户数
        - y 轴：这些用户的累积处理效应（处理组响应 - 对照组期望响应）

        曲线下面积（Qini Score）越大，说明 CATE 排序能力越强。
        """
        import matplotlib.pyplot as plt

        if self.cate_estimates is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")

        n = len(self.cate_estimates)

        # 按 CATE 降序排列
        sort_idx = np.argsort(self.cate_estimates)[::-1]
        treatment = df[treatment_col].values[sort_idx]
        outcome = df[outcome_col].values[sort_idx]

        # 经验 uplift：处理组平均响应 - 对照组平均响应（累积）
        # 用 TMLE-style 估计：每个百分比分位的处理组/对照组差异
        treated = treatment
        cum_treated = np.cumsum(treated)
        cum_control = np.arange(1, n + 1) - cum_treated

        # 避免除零
        cum_y_treated = np.cumsum(outcome * treated)
        cum_y_control = np.cumsum(outcome * (1 - treated))

        with np.errstate(divide="ignore", invalid="ignore"):
            gain_treated = np.where(cum_treated > 0, cum_y_treated / cum_treated, 0)
            gain_control = np.where(cum_control > 0, cum_y_control / cum_control, 0)
            qini = cum_treated * (gain_treated - gain_control)

        # 随机基线：随机排序的 Qini（用真实处理的随机排列近似）
        rng = np.random.default_rng(42)
        random_sort_idx = rng.permutation(n)
        treatment_rand = df[treatment_col].values[random_sort_idx]
        outcome_rand = df[outcome_col].values[random_sort_idx]
        cum_treated_rand = np.cumsum(treatment_rand)
        cum_control_rand = np.arange(1, n + 1) - cum_treated_rand
        cum_y_treated_rand = np.cumsum(outcome_rand * treatment_rand)
        cum_y_control_rand = np.cumsum(outcome_rand * (1 - treatment_rand))
        with np.errstate(divide="ignore", invalid="ignore"):
            gain_treated_rand = np.where(cum_treated_rand > 0, cum_y_treated_rand / cum_treated_rand, 0)
            gain_control_rand = np.where(cum_control_rand > 0, cum_y_control_rand / cum_control_rand, 0)
            qini_rand = cum_treated_rand * (gain_treated_rand - gain_control_rand)

        # 绘图
        plt.figure(figsize=(10, 6))
        x = np.arange(1, n + 1)
        plt.plot(x, qini, label="Model (CATE-sorted)", linewidth=2, color="blue")
        plt.plot(x, qini_rand, label="Random", linestyle="--", linewidth=2, color="gray")
        plt.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
        plt.xlabel("Number of Users (sorted by CATE)", fontsize=12)
        plt.ylabel("Cumulative Uplift (Qini)", fontsize=12)
        plt.title(f"Qini Curve - {self.config.learner_type.upper()}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_cumulative_gain(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        save_path: Optional[str] = None,
    ):
        """
        绘制 Cumulative Gain Curve（手动实现）

        纵轴：按 CATE 排序后，补贴 top-k 用户的累积 GTV 增量
        """
        import matplotlib.pyplot as plt

        if self.cate_estimates is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")

        n = len(self.cate_estimates)
        sort_idx = np.argsort(self.cate_estimates)[::-1]
        treatment = df[treatment_col].values[sort_idx]
        outcome = df[outcome_col].values[sort_idx]

        # 模型增益：处理组的累积 outcome
        model_gain = np.cumsum(outcome * treatment)

        # 随机基线
        rng = np.random.default_rng(42)
        rand_idx = rng.permutation(n)
        rand_treatment = df[treatment_col].values[rand_idx]
        rand_outcome = df[outcome_col].values[rand_idx]
        random_gain = np.cumsum(rand_outcome * rand_treatment)

        # 绘图
        plt.figure(figsize=(10, 6))
        x = np.arange(1, n + 1)
        plt.plot(x, model_gain, label="Model (CATE-sorted)", linewidth=2, color="blue")
        plt.plot(x, random_gain, label="Random", linestyle="--", linewidth=2, color="gray")
        plt.xlabel("Number of Users (sorted by CATE)", fontsize=12)
        plt.ylabel("Cumulative Gain", fontsize=12)
        plt.title(f"Cumulative Gain Curve - {self.config.learner_type.upper()}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def compare_with_existing(
        self,
        df_existing: pd.DataFrame,
        df_causalml: pd.DataFrame,
        cate_col_existing: str = "cate_existing",
        cate_col_causalml: str = "cate_causalml"
    ) -> Dict[str, Any]:
        """
        与现有T/X/DR-Learner结果对比
        
        返回：
        - 对比指标：相关系数、MAE、RMSE
        """
        from scipy.stats import pearsonr
        
        # 确保索引对齐
        cate_existing = df_existing[cate_col_existing].values
        cate_causalml = df_causalml[cate_col_causalml].values
        
        # 计算指标
        corr, _ = pearsonr(cate_existing, cate_causalml)
        mae = np.mean(np.abs(cate_existing - cate_causalml))
        rmse = np.sqrt(np.mean((cate_existing - cate_causalml) ** 2))
        
        return {
            "pearson_correlation": float(corr),
            "mae": float(mae),
            "rmse": float(rmse),
            "cate_mean_existing": float(np.mean(cate_existing)),
            "cate_mean_causalml": float(np.mean(cate_causalml)),
            "cate_std_existing": float(np.std(cate_existing)),
            "cate_std_causalml": float(np.std(cate_causalml))
        }
    
    def plot_uplift_curve(
        self, 
        df: pd.DataFrame, 
        treatment_col: str, 
        outcome_col: str, 
        save_path: Optional[str] = None
    ):
        """
        绘制Uplift曲线（Qini曲线）
        
        修复：移除对plot_gain的依赖，改用matplotlib手动绘制
        """
        import matplotlib.pyplot as plt
        
        if self.cate_estimates is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")
        
        if self.feature_cols is None:
            raise ValueError("feature_cols not set. Call fit_predict first.")
        
        # 准备数据
        treatment = df[treatment_col].values
        outcome = df[outcome_col].values
        cate = self.cate_estimates
        
        # 按CATE排序
        sorted_idx = np.argsort(cate)[::-1]  # 降序排列
        sorted_treatment = treatment[sorted_idx]
        sorted_outcome = outcome[sorted_idx]
        
        # 计算累积增益
        n = len(sorted_idx)
        cumulative_treatment = np.cumsum(sorted_treatment)
        cumulative_outcome = np.cumsum(sorted_outcome)
        
        # 计算Qini曲线：处理组的累积响应 - 对照组的期望响应
        # 简化版本：使用随机分配的基线
        treatment_rate = np.mean(treatment)
        expected_outcome_control = np.cumsum(np.ones(n) * treatment_rate) * np.mean(outcome[treatment == 0])
        expected_outcome_treatment = cumulative_outcome
        
        qini = np.cumsum(sorted_outcome * sorted_treatment) - np.cumsum(sorted_outcome * (1 - sorted_treatment)) * treatment_rate / (1 - treatment_rate + 1e-6)
        
        # 绘制
        plt.figure(figsize=(10, 6))
        # 修复：使用正确的x轴范围，避免维度不匹配
        x_axis = np.arange(len(qini))
        plt.plot(x_axis, qini, label='Qini Curve', linewidth=2)
        # 随机基线：使用与qini相同的长度
        random_baseline = np.cumsum(np.random.randn(len(qini)) * np.std(qini) / np.sqrt(len(qini)))
        plt.plot(x_axis, random_baseline, label='Random Baseline', linestyle='--', linewidth=2)
        
        plt.xlabel('Number of Users (sorted by CATE)', fontsize=12)
        plt.ylabel('Cumulative Uplift', fontsize=12)
        plt.title(f'Uplift Curve - {self.config.learner_type.upper()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    def plot_gain_curve(
        self, 
        df: pd.DataFrame, 
        treatment_col: str, 
        outcome_col: str, 
        save_path: Optional[str] = None
    ):
        """
        绘制Gain曲线（累积增益曲线）
        
        手动实现，不依赖causalml.metrics.plot_gain
        """
        import matplotlib.pyplot as plt
        
        if self.cate_estimates is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")
        
        # 准备数据
        treatment = df[treatment_col].values
        outcome = df[outcome_col].values
        cate = self.cate_estimates
        
        # 按CATE排序
        sorted_idx = np.argsort(cate)[::-1]
        sorted_treatment = treatment[sorted_idx]
        sorted_outcome = outcome[sorted_idx]
        
        # 计算累积增益 - 只计算处理组的增益
        n = len(sorted_idx)
        # 模型增益：按照CATE排序后，处理组的累积outcome
        model_gain = np.cumsum(sorted_outcome * sorted_treatment)
        
        # 随机基线：随机排序下的期望增益
        treatment_rate = np.mean(treatment)
        n_treatment = np.sum(treatment)
        random_gain = np.cumsum(np.ones(n) * np.sum(outcome * treatment) / n_treatment)
        
        # 最优增益（oracle）：如果我们知道真实的异质性效应，应该先给效应最大的人发券
        # 简化：使用处理组中outcome最高的进行排序（实际中我们不知道）
        treatment_outcomes = outcome[treatment == 1]
        optimal_gain_full = np.sort(treatment_outcomes)[::-1]  # 降序
        # 扩展到n长度（累积）
        optimal_gain = np.cumsum(np.concatenate([optimal_gain_full, np.zeros(max(0, n - len(optimal_gain_full)))]))[:n]
        
        # 绘制
        plt.figure(figsize=(10, 6))
        x_axis = np.arange(len(model_gain))
        plt.plot(x_axis, model_gain, label='Model', linewidth=2)
        plt.plot(x_axis, random_gain, label='Random', linestyle='--', linewidth=2)
        plt.plot(x_axis, optimal_gain, label='Optimal', linestyle=':', linewidth=2)
        
        plt.xlabel('Number of Users (sorted by CATE)', fontsize=12)
        plt.ylabel('Cumulative Gain', fontsize=12)
        plt.title(f'Gain Curve - {self.config.learner_type.upper()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def run_causalml_comparison(
    data_path: str = "data/synthetic/synthetic_data.csv",
    learner_types: Optional[list[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    运行CausalML对比实验
    
    参数：
    - data_path: 数据文件路径
    - learner_types: 要对比的Learner类型
    - feature_cols: 特征列名（如果为None则自动生成）
    
    返回：
    - 各Learner的对比结果
    """
    # 加载数据
    df = pd.read_csv(data_path)

    if learner_types is None:
        learner_types = ["tlearner", "xlearner", "drlearner"]

    # 如果没有指定特征列，则使用所有非处理、非结果的列
    if feature_cols is None:
        exclude_cols = ["treatment", "outcome", "cate_existing", "cate_causalml"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    results = {}
    for learner_type in learner_types:
        config = CausalMLConfig(learner_type=learner_type)
        wrapper = CausalMLWrapper(config)
        
        # 拟合预测
        result_df = wrapper.fit_predict(
            df,
            feature_cols=feature_cols,
            treatment_col="treatment",
            outcome_col="outcome"
        )
        
        results[learner_type] = {
            "cate_mean": float(np.mean(result_df["cate_causalml"])),
            "cate_std": float(np.std(result_df["cate_causalml"])),
            "cate_values": result_df["cate_causalml"].values,
            "result_df": result_df
        }
    
    return results


def compare_all_learners(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    existing_cate_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    对比所有CausalML Learner的结果
    
    参数：
    - df: 输入DataFrame
    - feature_cols: 特征列名
    - treatment_col: 处理列名
    - outcome_col: 结果列名
    - existing_cate_col: 现有手工实现的CATE列名（可选）
    
    返回：
    - 所有Learner的对比结果
    """
    learner_types = ["tlearner", "xlearner", "drlearner", "slearner"]
    results = {}
    
    # 运行所有Learner
    for learner_type in learner_types:
        config = CausalMLConfig(learner_type=learner_type)
        wrapper = CausalMLWrapper(config)
        
        result_df = wrapper.fit_predict(
            df,
            feature_cols=feature_cols,
            treatment_col=treatment_col,
            outcome_col=outcome_col
        )
        
        results[learner_type] = {
            "wrapper": wrapper,
            "result_df": result_df,
            "cate_mean": float(np.mean(result_df["cate_causalml"])),
            "cate_std": float(np.std(result_df["cate_causalml"]))
        }
    
    # 如果提供了现有CATE列，则进行对比
    if existing_cate_col and existing_cate_col in df.columns:
        comparison_results = {}
        for learner_type in learner_types:
            if results[learner_type]["result_df"] is not None:
                wrapper = results[learner_type]["wrapper"]
                comparison = wrapper.compare_with_existing(
                    df,
                    results[learner_type]["result_df"],
                    cate_col_existing=existing_cate_col,
                    cate_col_causalml="cate_causalml"
                )
                comparison_results[learner_type] = comparison
        
        return {
            "learner_results": results,
            "comparison_with_existing": comparison_results
        }
    
    return {"learner_results": results}


if __name__ == "__main__":
    # 示例：生成合成数据并运行
    from sklearn.datasets import make_regression
    
    # 生成合成数据
    n_samples = 10000
    n_features = 10
    
    X, _ = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
    treatment = np.random.binomial(1, 0.5, n_samples)
    # 异质性处理效应：第一个特征影响处理效应
    outcome = X[:, 0] * treatment + np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["treatment"] = treatment
    df["outcome"] = outcome
    
    # 保存合成数据
    import os
    os.makedirs("data/synthetic", exist_ok=True)
    df.to_csv("data/synthetic/synthetic_data.csv", index=False)
    
    # 运行CausalML对比
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    results = run_causalml_comparison(
        "data/synthetic/synthetic_data.csv",
        learner_types=["tlearner", "xlearner", "drlearner"],
        feature_cols=feature_cols
    )
    
    print("CausalML Comparison Results:")
    for learner_type, result in results.items():
        print(f"{learner_type}: CATE mean = {result['cate_mean']:.4f}, std = {result['cate_std']:.4f}")
