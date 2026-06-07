"""
CausalML集成模块
与现有手工T/X/DR-Learner对比，验证结果鲁棒性

参考文献：
- CausalML (Uber): https://github.com/uber/causalml
- Athey & Wager (2018): Estimation and Inference of Heterogeneous Treatment Effects
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# CausalML imports - 使用正确的API
from causalml.inference.meta import BaseTLearner, BaseXLearner, BaseDRRegressor
from causalml.inference.meta import BaseSLearner


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
        self.model.fit(X, treatment, outcome)
        
        # 预测CATE - 修复：统一输出格式
        cate_pred = self.model.predict(X)
        # CausalML的predict返回shape可能是 (n, 1) 或 (n,)，需要统一squeeze
        self.cate_estimates = np.array(cate_pred).flatten()
        
        # 返回结果DataFrame
        result_df = df.copy()
        result_df["cate_causalml"] = self.cate_estimates
        
        return result_df
    
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
    learner_types: list[str] = ["tlearner", "xlearner", "drlearner"],
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
