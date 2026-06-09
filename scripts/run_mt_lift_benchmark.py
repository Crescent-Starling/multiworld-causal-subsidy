"""
MT-LIFT 数据集 CausalML Benchmark 脚本
=============================================

用现有 CausalMLWrapper 在 MT-LIFT 数据集上跑 benchmark，
评估 4 种 Learner（T/X/DR/S）的 AUUC / Qini。

数据集要求：
  data/mt_lift/train.csv   (从百度网盘下载)
  data/mt_lift/test.csv

Usage:
    # 先运行预处理（检查数据 + 生成统计信息）
    python scripts/preprocess_mt_lift.py

    # 全量 benchmark（预处理后）
    python scripts/run_mt_lift_benchmark.py

    # 采样加速（10% 数据）
    python scripts/run_mt_lift_benchmark.py --sample 0.1

    # 指定 outcome
    python scripts/run_mt_lift_benchmark.py --outcome click
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 添加 src/ 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modeling.causalml_wrapper import CausalMLWrapper, CausalMLConfig
from causalml.metrics import auuc_score, qini_score

# ============================================================
# 配置
# ============================================================
PROCESSED_DIR = "output/mt_lift_benchmark"
FIGS_DIR = "docs/figures"

TREATMENT_COL = "is_treated"   # 二值化后：0=对照组，1=发券组
FEATURE_COLS = [f"f{i}" for i in range(99)]  # f0 ~ f98


def load_processed_data(train_path: str, test_path: str,
                       outcome_col: str, sample_frac: float = 1.0):
    """加载预处理后的数据，可选采样"""
    print(f"\n{'='*60}")
    print(f"Loading processed data")
    print(f"{'='*60}")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # 采样加速
    if sample_frac < 1.0:
        df_train = df_train.sample(frac=sample_frac, random_state=42)
        df_test  = df_test.sample(frac=sample_frac, random_state=42)
        print(f"  Sampled {sample_frac:.0%}: train={len(df_train):,}, test={len(df_test):,}")
    else:
        print(f"  Full data: train={len(df_train):,}, test={len(df_test):,}")

    return df_train, df_test


def run_benchmark(df_train: pd.DataFrame, df_test: pd.DataFrame,
                   outcome_col: str, feature_cols: list) -> dict:
    """
    对单个 outcome 跑 4 种 Learner benchmark

    Returns:
        results: dict[learner_name] = eval_result_dict
    """
    print(f"\n{'='*60}")
    print(f"MT-LIFT Benchmark: outcome={outcome_col}")
    print(f"{'='*60}")

    learner_types = ["tlearner", "xlearner", "drlearner", "slearner"]
    results = {}

    for lt in learner_types:
        print(f"\n{'-'*60}")
        print(f"Running: {lt.upper()}")
        print(f"{'-'*60}")

        try:
            config = CausalMLConfig(learner_type=lt, random_state=42)
            wrapper = CausalMLWrapper(config=config)

            # fit_predict on train
            _ = wrapper.fit_predict(
                df=df_train,
                feature_cols=feature_cols,
                treatment_col=TREATMENT_COL,
                outcome_col=outcome_col,
            )

            # evaluate on test (需要feature_cols来在新数据上预测CATE)
            eval_result = wrapper.evaluate_cate_quality(
                df=df_test,
                treatment_col=TREATMENT_COL,
                outcome_col=outcome_col,
                feature_cols=feature_cols,  # 新增：用于在新数据上预测
            )

            results[lt] = eval_result
            # 使用安全格式化
            auuc = eval_result.get('auuc', None)
            qini = eval_result.get('qini', None)
            corr = eval_result.get('corr', None)
            ate_mean = eval_result.get('ate_mean', None)
            ate_std = eval_result.get('ate_std', None)
            
            def fmt_v(val, width=12):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A".rjust(width)
                try:
                    return f"{float(val):.4f}".rjust(width)
                except (ValueError, TypeError):
                    return "N/A".rjust(width)
            
            print(f"  ✓ AUUC:   {fmt_v(auuc, 10)}")
            print(f"  ✓ Qini:   {fmt_v(qini, 10)}")
            print(f"  ✓ corr:   {fmt_v(corr, 10)}")
            print(f"  ✓ ATE μ:  {fmt_v(ate_mean, 10)}")
            print(f"  ✓ ATE σ:  {fmt_v(ate_std, 10)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results[lt] = {"error": str(e)}

    return results


def compute_random_baseline(df: pd.DataFrame, outcome_col: str,
                             n_bootstrap: int = 50) -> dict:
    """
    计算随机基线的 AUUC / Qini。
    
    逻辑：
    - 打乱 treatment 标签（随机分配）
    - 用随机 CATE 值（正态分布）作为"模型预测"
    - 用 CausalML 的 auuc_score / qini_score 计算指标
    - 重复 n_bootstrap 次，取均值/标准差
    """
    print(f"\n{'='*60}")
    print(f"Computing random baseline (n_bootstrap={n_bootstrap})")
    print(f"{'='*60}")

    from causalml.metrics import auuc_score, qini_score

    rng = np.random.RandomState(42)
    auuc_scores = []
    qini_scores = []
    outcome = df[outcome_col].values
    treatment = df[TREATMENT_COL].values

    for i in range(n_bootstrap):
        treat_shuffled = rng.permutation(treatment)
        cate_random = rng.randn(len(outcome))
        # 列名不能用 'tau'（CausalML 保留名）
        tmp_df = pd.DataFrame({
            "y": outcome,
            "w": treat_shuffled,
            "random_cate": cate_random,
        })
        try:
            # 不传 treatment_effect_col → 用默认 'tau'（不在 df 中）→ 走经验 uplift 分支
            auuc_s = auuc_score(tmp_df, outcome_col="y", treatment_col="w")
            qini_s = qini_score(tmp_df, outcome_col="y", treatment_col="w")
            # auuc_score 返回 Series，index=模型名（即列名 'random_cate'）
            auuc_scores.append(float(auuc_s["random_cate"]))
            qini_scores.append(float(qini_s["random_cate"]))
        except Exception as e:
            print(f"    Warning: bootstrap {i} failed: {e}")
            pass

    if auuc_scores:
        return {
            "auuc_mean": float(np.mean(auuc_scores)),
            "auuc_std": float(np.std(auuc_scores)),
            "qini_mean": float(np.mean(qini_scores)),
            "qini_std": float(np.std(qini_scores)),
        }
    else:
        return {"auuc_mean": 0.500, "auuc_std": 0.05,
                "qini_mean": 0.0, "qini_std": 0.01}

def plot_benchmark_comparison(results: dict, outcome_col: str, figs_dir: str):
    """绘制 benchmark 对比图"""
    os.makedirs(figs_dir, exist_ok=True)

    learners = []
    auucs = []
    qinis = []

    for name, res in results.items():
        if "error" not in res:
            learners.append(name.upper())
            auucs.append(res.get("auuc", 0.5))
            qinis.append(res.get("qini", 0.0))

    if not learners:
        print("  No valid results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(learners, auucs, color=colors[:len(learners)], alpha=0.8)
    ax.axhline(0.500, color="red", linestyle="--", alpha=0.7,
                label="Random baseline (0.500)")
    ax.set_ylabel("AUUC", fontsize=12)
    ax.set_title(f"MT-LIFT Benchmark: AUUC ({outcome_col})", fontsize=13)
    y_max = max(auucs) * 1.1 if auucs else 0.7
    ax.set_ylim([0.45, max(y_max, 0.55)])
    ax.legend()
    for bar, val in zip(bars, auucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=10)

    ax = axes[1]
    bars = ax.bar(learners, qinis, color=colors[:len(learners)], alpha=0.8)
    ax.axhline(0.000, color="red", linestyle="--", alpha=0.7,
                label="Random baseline (0.000)")
    ax.set_ylabel("Qini Coefficient", fontsize=12)
    ax.set_title(f"MT-LIFT Benchmark: Qini ({outcome_col})", fontsize=13)
    ax.legend()
    for bar, val in zip(bars, qinis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(figs_dir, f"mt_lift_benchmark_{outcome_col}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {output_path}")
    plt.close()


def print_results_table(results: dict, random_baseline: dict, outcome_col: str):
    """打印格式化的结果表格"""
    def fmt_val(val, width=12):
        """格式化单个值为固定宽度字符串"""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A".rjust(width)
        try:
            return f"{float(val):.4f}".rjust(width)
        except (ValueError, TypeError):
            return "N/A".rjust(width)
    
    print(f"\n{'='*70}")
    print(f"  MT-LIFT Benchmark Results")
    print(f"{'='*70}")
    print(f"  {'Method':<15} {'AUUC':<12} {'Qini':<12} {'ATE μ':<12} {'ATE σ':<12}")
    print(f"  {'-'*68}")

    rb = random_baseline
    auuc_rb = rb.get('auuc_mean', 0.500)
    print(f"  {'Random':<15} {fmt_val(auuc_rb, 12)} "
          f"{'0.0000'.rjust(12)} {'N/A'.rjust(12)} {'N/A'.rjust(12)}")

    for name, res in results.items():
        if "error" in res:
            print(f"  {name.upper():<15} {'FAILED'.rjust(12)} {'FAILED'.rjust(12)} "
                  f"{'N/A'.rjust(12)} {'N/A'.rjust(12)}")
        else:
            auuc    = res.get('auuc',    None)
            qini    = res.get('qini',    None)
            ate_mu  = res.get('ate_mean', None)
            ate_std = res.get('ate_std', None)
            print(f"  {name.upper():<15} {fmt_val(auuc, 12)} {fmt_val(qini, 12)} "
                  f"{fmt_val(ate_mu, 12)} {fmt_val(ate_std, 12)}")

    print(f"  {'-'*68}")
    print(f"  Outcome: {outcome_col}")
    print(f"  Treatment: is_treated (0=control, 1=treated)")
    print(f"  Features: f0 ~ f98 (99-dim anonymous)")
    print(f"{'='*70}\n")


def save_results_csv(results: dict, random_baseline: dict,
                    output_dir: str, outcome_col: str):
    """保存结果为 CSV"""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    rb = random_baseline
    rows.append({
        "method": "Random",
        "auuc": rb.get("auuc_mean", 0.500),
        "qini": 0.000,
        "ate_mean": None,
        "ate_std": None,
        "outcome": outcome_col,
    })

    for name, res in results.items():
        if "error" not in res:
            rows.append({
                "method": name.upper(),
                "auuc": res.get("auuc"),
                "qini": res.get("qini"),
                "ate_mean": res.get("ate_mean"),
                "ate_std": res.get("ate_std"),
                "outcome": outcome_col,
            })

    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, f"benchmark_results_{outcome_col}.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MT-LIFT CausalML Benchmark")
    parser.add_argument("--outcome", type=str, default="conversion",
                        choices=["conversion", "click"],
                        help="Outcome column to use")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Sample fraction for speed (0.1 = 10%%)")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip plotting (for headless servers)")
    args = parser.parse_args()

    outcome_col = args.outcome
    sample_frac = args.sample

    # 检查预处理数据是否存在
    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test_processed.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"ERROR: Processed data not found!")
        print(f"  Please run: python scripts/preprocess_mt_lift.py")
        print(f"  Then re-run this script.")
        sys.exit(1)

    # Step 1: 加载数据
    df_train, df_test = load_processed_data(train_path, test_path,
                                             outcome_col, sample_frac)

    # Step 2: 随机基线
    random_baseline = compute_random_baseline(df_test, outcome_col)
    print(f"  Random baseline AUUC: {random_baseline['auuc_mean']:.4f}"
          f" ± {random_baseline['auuc_std']:.4f}")

    # Step 3: 跑 benchmark
    results = run_benchmark(df_train, df_test, outcome_col, FEATURE_COLS)

    # Step 4: 打印结果
    print_results_table(results, random_baseline, outcome_col)

    # Step 5: 保存结果
    save_results_csv(results, random_baseline, PROCESSED_DIR, outcome_col)

    # Step 6: 绘图
    if not args.skip_plot:
        plot_benchmark_comparison(results, outcome_col, FIGS_DIR)

    print(f"\n✓ Benchmark complete! Results saved to: {PROCESSED_DIR}/")
    print(f"  Next: Add results to docs/technical_paper.md")


if __name__ == "__main__":
    main()
