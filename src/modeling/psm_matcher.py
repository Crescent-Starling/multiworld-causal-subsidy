"""
PSM倾向得分匹配模块
与meituan-subsidy-efficiency的L1层对标

参考文献：
- Wzh20040721/meituan-subsidy-efficiency (GitHub)
- Rubin (1973): Matching to Remove Bias in Observational Studies
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment


@dataclass
class PSMConfig:
    """PSM配置"""
    caliper: float = 0.05  # 卡尺匹配阈值
    method: str = "nearest"  # "nearest", "caliper", "optimal"
    n_neighbors: int = 1  # 最近邻数量
    random_state: int = 42


class PSMMatcher:
    """
    倾向得分匹配器
    
    用途：
    1. 计算倾向得分
    2. 进行匹配（最近邻/卡尺/最优）
    3. 评估匹配质量（SMD）
    4. 对比IPW与PSM的ATE估计
    """
    
    def __init__(self, config: Optional[PSMConfig] = None):
        self.config = config or PSMConfig()
        self.propensity_scores = None
        self.matched_pairs = None
        self.match_quality = None
        
    def compute_propensity_scores(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        计算倾向得分
        
        参数：
        - df: 数据DataFrame
        - treatment_col: 处理变量列名（0/1）
        - feature_cols: 用于计算倾向得分的特征列
        
        返回：
        - 倾向得分数组
        """
        X = df[feature_cols].values
        treatment = df[treatment_col].values
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 用逻辑回归估计倾向得分
        lr = LogisticRegression(random_state=self.config.random_state, max_iter=1000)
        lr.fit(X_scaled, treatment)
        
        # 预测概率（倾向得分）
        self.propensity_scores = lr.predict_proba(X_scaled)[:, 1]
        
        # 评估倾向得分模型（AUC）
        auc = roc_auc_score(treatment, self.propensity_scores)
        print(f"Propensity Score Model AUC: {auc:.4f}")
        
        return self.propensity_scores
    
    def match(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        propensity_scores: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        进行匹配
        
        参数：
        - df: 数据DataFrame
        - treatment_col: 处理变量列名
        - propensity_scores: 倾向得分（若None则自动计算）
        
        返回：
        - 匹配后的DataFrame
        """
        if propensity_scores is None:
            if self.propensity_scores is None:
                raise ValueError("Must compute propensity scores first")
            propensity_scores = self.propensity_scores
        
        treatment = df[treatment_col].values
        
        # 分离处理组和对照组
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        treated_scores = propensity_scores[treated_idx]
        control_scores = propensity_scores[control_idx]
        
        # 匹配
        if self.config.method == "nearest":
            matched_pairs = self._nearest_neighbor_match(treated_idx, control_idx, treated_scores, control_scores)
        elif self.config.method == "caliper":
            matched_pairs = self._caliper_match(treated_idx, control_idx, treated_scores, control_scores)
        elif self.config.method == "optimal":
            matched_pairs = self._optimal_match(treated_idx, control_idx, treated_scores, control_scores)
        else:
            raise ValueError(f"Unknown matching method: {self.config.method}")
        
        self.matched_pairs = matched_pairs
        
        # 构建匹配后的DataFrame
        matched_df = self._build_matched_dataframe(df, matched_pairs, treatment_col)
        
        # 评估匹配质量
        self.match_quality = self.evaluate_match_quality(df, matched_df, treatment_col)
        
        return matched_df
    
    def _nearest_neighbor_match(
        self,
        treated_idx: np.ndarray,
        control_idx: np.ndarray,
        treated_scores: np.ndarray,
        control_scores: np.ndarray
    ) -> List[Tuple[int, int]]:
        """最近邻匹配"""
        matched_pairs = []
        
        for i, (t_idx, t_score) in enumerate(zip(treated_idx, treated_scores)):
            # 找到得分最接近的对照组
            distances = np.abs(control_scores - t_score)
            nearest_idx = control_idx[np.argmin(distances)]
            matched_pairs.append((t_idx, nearest_idx))
        
        return matched_pairs
    
    def _caliper_match(
        self,
        treated_idx: np.ndarray,
        control_idx: np.ndarray,
        treated_scores: np.ndarray,
        control_scores: np.ndarray
    ) -> List[Tuple[int, int]]:
        """卡尺匹配（仅匹配得分差异在caliper内的对）"""
        matched_pairs = []
        used_control = set()

        for i, (t_idx, t_score) in enumerate(zip(treated_idx, treated_scores)):
            # 找到在caliper内的对照组
            distances = np.abs(control_scores - t_score)
            within_caliper = np.where(distances < self.config.caliper)[0]

            if len(within_caliper) > 0:
                # 选择距离最接近的且未使用的对照组
                valid_mask = np.array([control_idx[j] not in used_control for j in within_caliper])
                valid_within = within_caliper[valid_mask]

                if len(valid_within) > 0:
                    # 在有效对照组中选距离最小的
                    best_j = valid_within[np.argmin(distances[valid_within])]
                    nearest = control_idx[best_j]
                    matched_pairs.append((t_idx, nearest))
                    used_control.add(nearest)

        return matched_pairs

    def _optimal_match(
        self,
        treated_idx: np.ndarray,
        control_idx: np.ndarray,
        treated_scores: np.ndarray,
        control_scores: np.ndarray
    ) -> List[Tuple[int, int]]:
        """最优匹配（最小化全局倾向得分距离之和）

        使用匈牙利算法（linear_sum_assignment）求解二部图最小权重匹配。
        适用于处理组和对照组数量不等的情况（取 min(n_treated, n_control) 对）。
        """
        # 构建距离矩阵：cost[i, j] = |treated_score[i] - control_score[j]|
        cost_matrix = np.abs(treated_scores[:, None] - control_scores[None, :])

        # 应用卡尺约束：超出 caliper 的距离设为无穷大
        if self.config.caliper is not None and self.config.caliper > 0:
            cost_matrix[cost_matrix > self.config.caliper] = 1e10

        # 匈牙利算法求解最小权重匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            # 跳过超出卡尺的匹配
            if cost_matrix[r, c] >= 1e10:
                continue
            matched_pairs.append((treated_idx[r], control_idx[c]))

        return matched_pairs

    def _build_matched_dataframe(
        self,
        df: pd.DataFrame,
        matched_pairs: List[Tuple[int, int]],
        treatment_col: str
    ) -> pd.DataFrame:
        """构建匹配后的DataFrame"""
        matched_indices = []
        for t_idx, c_idx in matched_pairs:
            matched_indices.extend([t_idx, c_idx])
        
        matched_df = df.iloc[matched_indices].copy()
        return matched_df
    
    def evaluate_match_quality(
        self,
        original_df: pd.DataFrame,
        matched_df: pd.DataFrame,
        treatment_col: str
    ) -> Dict[str, Any]:
        """
        评估匹配质量

        指标：
        - SMD（标准化均值差）：对所有数值特征计算，< 0.1 表示平衡良好
        - 匹配率：成功匹配的处理组比例
        """
        # 获取所有数值型特征列（排除处理变量）
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != treatment_col]

        smd_original_list = []
        smd_matched_list = []

        for col in feature_cols:
            # 原始数据SMD
            treated_orig = original_df[original_df[treatment_col] == 1][col]
            control_orig = original_df[original_df[treatment_col] == 0][col]
            pooled_std_orig = np.sqrt((treated_orig.var() + control_orig.var()) / 2)
            if pooled_std_orig > 1e-10:
                smd_orig = (treated_orig.mean() - control_orig.mean()) / pooled_std_orig
            else:
                smd_orig = 0.0
            smd_original_list.append(abs(smd_orig))

            # 匹配后SMD
            treated_match = matched_df[matched_df[treatment_col] == 1][col]
            control_match = matched_df[matched_df[treatment_col] == 0][col]
            pooled_std_match = np.sqrt((treated_match.var() + control_match.var()) / 2)
            if pooled_std_match > 1e-10:
                smd_match = (treated_match.mean() - control_match.mean()) / pooled_std_match
            else:
                smd_match = 0.0
            smd_matched_list.append(abs(smd_match))

        # 报告最大和平均SMD
        max_smd_original = max(smd_original_list) if smd_original_list else 0.0
        max_smd_matched = max(smd_matched_list) if smd_matched_list else 0.0
        mean_smd_original = np.mean(smd_original_list) if smd_original_list else 0.0
        mean_smd_matched = np.mean(smd_matched_list) if smd_matched_list else 0.0

        match_rate = len(matched_df) / max(len(original_df), 1)

        return {
            "smd_original_max": max_smd_original,
            "smd_original_mean": mean_smd_original,
            "smd_matched_max": max_smd_matched,
            "smd_matched_mean": mean_smd_matched,
            "smd_by_feature_original": dict(zip(feature_cols, smd_original_list)),
            "smd_by_feature_matched": dict(zip(feature_cols, smd_matched_list)),
            "match_rate": match_rate,
            "n_matched_pairs": len(matched_df) // 2,
            "balanced": max_smd_matched < 0.1,  # 最大SMD < 0.1 表示平衡
        }
    
    def estimate_ate_after_matching(self, matched_df: pd.DataFrame, outcome_col: str, treatment_col: str) -> float:
        """匹配后估计ATE"""
        treated_outcome = matched_df[matched_df[treatment_col] == 1][outcome_col].mean()
        control_outcome = matched_df[matched_df[treatment_col] == 0][outcome_col].mean()
        ate = treated_outcome - control_outcome
        return ate
    
    def compare_ipw_vs_psm(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        ipw_ate: float
    ) -> Dict[str, Any]:
        """
        对比IPW与PSM的ATE估计
        
        参数：
        - ipw_ate: IPW估计的ATE
        """
        # 计算倾向得分
        feature_cols = [col for col in df.columns if col not in [treatment_col, outcome_col]]
        self.compute_propensity_scores(df, treatment_col, feature_cols)
        
        # 匹配
        matched_df = self.match(df, treatment_col)
        
        # 估计ATE
        psm_ate = self.estimate_ate_after_matching(matched_df, outcome_col, treatment_col)
        
        return {
            "ipw_ate": ipw_ate,
            "psm_ate": psm_ate,
            "difference": psm_ate - ipw_ate,
            "relative_difference": (psm_ate - ipw_ate) / abs(ipw_ate) if ipw_ate != 0 else None,
            "match_quality": self.match_quality
        }


def run_psm_example():
    """运行PSM示例"""
    # 生成合成数据
    np.random.seed(42)
    n_samples = 10000
    
    # 特征
    X1 = np.random.randn(n_samples)  # 连续特征
    X2 = np.random.binomial(1, 0.5, n_samples)  # 二元特征
    
    # 处理分配（与特征相关）
    propensity = 1 / (1 + np.exp(-0.5 * X1 + 0.3 * X2))
    treatment = np.random.binomial(1, propensity, n_samples)
    
    # 结果（处理效应 = 2.0）
    outcome = 1.0 * X1 + 0.5 * X2 + treatment * 2.0 + np.random.randn(n_samples)
    
    df = pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "treatment": treatment,
        "outcome": outcome
    })
    
    # 运行PSM
    config = PSMConfig(method="nearest", caliper=0.05)
    matcher = PSMMatcher(config)
    
    # 计算倾向得分
    propensity_scores = matcher.compute_propensity_scores(df, "treatment", ["X1", "X2"])
    
    # 匹配
    matched_df = matcher.match(df, "treatment", propensity_scores)
    
    # 估计ATE
    psm_ate = matcher.estimate_ate_after_matching(matched_df, "outcome", "treatment")
    print(f"PSM ATE: {psm_ate:.4f}")
    print(f"True ATE: 2.0000")
    print(f"Match Quality: {matcher.match_quality}")
    
    return psm_ate, matcher.match_quality


if __name__ == "__main__":
    run_psm_example()
