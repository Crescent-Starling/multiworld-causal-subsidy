"""
运行CausalML对比实验

对比CausalML与现有手工T/X/DR-Learner的结果
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modeling.causalml_wrapper import CausalMLWrapper, CausalMLConfig, run_causalml_comparison


def main():
    """主函数"""
    print("=" * 60)
    print("CausalML Comparison Experiment")
    print("=" * 60)
    
    # 生成合成数据
    print("\n1. Generating synthetic data...")
    n_samples = 10000
    n_features = 10
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # 异质性处理效应：CATE = X[:, 0] * treatment
    outcome = X[:, 0] * treatment + np.random.normal(0, 1, n_samples)
    
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["treatment"] = treatment
    df["outcome"] = outcome
    
    # 保存数据
    os.makedirs("data/synthetic", exist_ok=True)
    df.to_csv("data/synthetic/synthetic_data.csv", index=False)
    print(f"  Synthetic data saved to data/synthetic/synthetic_data.csv")
    print(f"  Samples: {n_samples}, Features: {n_features}")
    
    # 运行CausalML对比
    print("\n2. Running CausalML comparison...")
    learner_types = ["tlearner", "xlearner", "drlearner"]
    results = run_causalml_comparison("data/synthetic/synthetic_data.csv", learner_types)
    
    # 打印结果
    print("\n3. Results:")
    print("-" * 60)
    for learner_type, result in results.items():
        print(f"{learner_type.upper()}:")
        print(f"  CATE mean = {result['cate_mean']:.4f}")
        print(f"  CATE std  = {result['cate_std']:.4f}")
        print()
    
    # 保存结果
    print("4. Saving results...")
    result_summary = pd.DataFrame([
        {
            "learner_type": learner_type,
            "cate_mean": result["cate_mean"],
            "cate_std": result["cate_std"]
        }
        for learner_type, result in results.items()
    ])
    result_summary.to_csv("output/causalml_comparison.csv", index=False)
    print(f"  Results saved to output/causalml_comparison.csv")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    main()
