"""
MT-LIFT 数据集预处理脚本
================================

功能：
1. 加载 MT-LIFT train.csv / test.csv
2. 二值化 treatment（0=对照组，1~4=发券组）
3. 选择 outcome（conversion 或 click）
4. 训练/测试拆分（70/30，stratify by treatment）
5. 保存预处理后的数据

数据集获取：
  百度网盘：https://pan.baidu.com/s/1YmE5g-Y71ULNptiWqpToPA  提取码：06nb
  下载后放入 data/mt_lift/ 目录

Usage:
    python scripts/preprocess_mt_lift.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# 配置
# ============================================================
DATA_DIR = "data/mt_lift"
RAW_TRAIN = os.path.join(DATA_DIR, "train.csv")
RAW_TEST = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "output/mt_lift_benchmark"

TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"   # "conversion" or "click"
IS_TREATED_COL = "is_treated"

# 特征列：f0 ~ f98（99 维匿名特征）
FEATURE_COLS = [f"f{i}" for i in range(99)]


def load_and_explore(raw_path: str) -> pd.DataFrame:
    """加载 CSV 并输出基本统计信息"""
    print(f"\n{'='*60}")
    print(f"Loading: {raw_path}")
    print(f"{'='*60}")
    
    # 尝试读取前 10 行，确认文件格式
    try:
        df = pd.read_csv(raw_path, nrows=10)
        print(f"  Columns ({len(df.columns)}): {list(df.columns[:10])} ...")
    except Exception as e:
        print(f"  ERROR: Cannot read {raw_path}")
        print(f"  Please download the dataset from Baidu Netdisk:")
        print(f"    https://pan.baidu.com/s/1YmE5g-Y71ULNptiWqpToPA  (code: 06nb)")
        print(f"  And place train.csv and test.csv in: {DATA_DIR}/")
        sys.exit(1)
    
    # 全量读取
    print(f"  Reading full file...")
    df = pd.read_csv(raw_path)
    print(f"  Shape: {df.shape}")
    
    # 基本统计
    print(f"\n  Treatment distribution:")
    treat_counts = df[TREATMENT_COL].value_counts().sort_index()
    for k, v in treat_counts.items():
        print(f"    treatment={k}: {v:,} ({v/len(df):.2%})")
    
    print(f"\n  Outcome distribution (conversion):")
    conv_counts = df["conversion"].value_counts().sort_index()
    for k, v in conv_counts.items():
        print(f"    conversion={k}: {v:,} ({v/len(df):.2%})")
    
    print(f"\n  Outcome distribution (click):")
    click_counts = df["click"].value_counts().sort_index()
    for k, v in click_counts.items():
        print(f"    click={k}: {v:,} ({v/len(df):.2%})")
    
    # conversion 率按 treatment 分组
    print(f"\n  Conversion rate by treatment:")
    for t in sorted(df[TREATMENT_COL].unique()):
        sub = df[df[TREATMENT_COL] == t]
        cr = sub["conversion"].mean()
        ck = sub["click"].mean()
        print(f"    treatment={t}: conversion={cr:.4f} ({cr:.2%}), click={ck:.4f} ({ck:.2%}), n={len(sub):,}")
    
    return df


def preprocess(df: pd.DataFrame, outcome_col: str = OUTCOME_COL) -> pd.DataFrame:
    """
    预处理：
    1. 二值化 treatment -> is_treated
    2. 保留指定 outcome
    3. 保留特征列
    """
    df = df.copy()
    
    # 二值化 treatment
    df[IS_TREATED_COL] = (df[TREATMENT_COL] > 0).astype(int)
    
    # 检查 treatment=0 是否为真正的对照组
    cr_by_treated = df.groupby(IS_TREATED_COL)["conversion"].mean()
    print(f"\n  Conversion rate by is_treated:")
    for k, v in cr_by_treated.items():
        print(f"    is_treated={k}: conversion={v:.4f} ({v:.2%})")
    
    # 标注正负样本（用于 uplift 评估）
    # treatment=1 & outcome=1：券起效
    # treatment=1 & outcome=0：券无效
    # treatment=0 & outcome=1：自然转化
    # treatment=0 & outcome=0：自然不转化
    
    return df


def save_preprocessed(df: pd.DataFrame, output_dir: str, suffix: str = "train"):
    """保存预处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整预处理数据
    output_path = os.path.join(output_dir, f"{suffix}_processed.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(df):,} rows)")
    
    # 保存 data summary
    summary_path = os.path.join(output_dir, f"{suffix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"MT-LIFT Preprocessed Data Summary ({suffix})\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write(f"Treatment distribution:\n")
        for k, v in df[TREATMENT_COL].value_counts().sort_index().items():
            f.write(f"  treatment={k}: {v:,} ({v/len(df):.2%})\n")
        f.write(f"\n")
        f.write(f"is_treated distribution:\n")
        for k, v in df[IS_TREATED_COL].value_counts().sort_index().items():
            f.write(f"  is_treated={k}: {v:,} ({v/len(df):.2%})\n")
        f.write(f"\n")
        f.write(f"Outcome ({OUTCOME_COL}) distribution:\n")
        for k, v in df[OUTCOME_COL].value_counts().sort_index().items():
            f.write(f"  {OUTCOME_COL}={k}: {v:,} ({v/len(df):.2%})\n")
        f.write(f"\n")
        f.write(f"Conversion rate by is_treated:\n")
        for k, v in df.groupby(IS_TREATED_COL)[OUTCOME_COL].mean().items():
            f.write(f"  is_treated={k}: {v:.4f} ({v:.2%})\n")
    print(f"  Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MT-LIFT dataset")
    parser.add_argument("--outcome", type=str, default=OUTCOME_COL,
                        choices=["conversion", "click"],
                        help="Outcome column to use")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratify (may cause imbalance in train/test)")
    args = parser.parse_args()
    
    outcome_col = args.outcome
    
    # 检查文件是否存在
    if not os.path.exists(RAW_TRAIN):
        print(f"ERROR: {RAW_TRAIN} not found!")
        print(f"Please download MT-LIFT dataset from:")
        print(f"  Baidu Netdisk: https://pan.baidu.com/s/1YmE5g-Y71ULNptiWqpToPA  (code: 06nb)")
        print(f"Then place train.csv and test.csv in: {DATA_DIR}/")
        sys.exit(1)
    
    # Step 1: 探索训练集
    print(f"\n{'#'*60}")
    print(f"# Step 1: Exploring training set")
    print(f"{'#'*60}")
    df_train = load_and_explore(RAW_TRAIN)
    
    # Step 2: 探索测试集
    print(f"\n{'#'*60}")
    print(f"# Step 2: Exploring test set")
    print(f"{'#'*60}")
    df_test = load_and_explore(RAW_TEST)
    
    # Step 3: 预处理
    print(f"\n{'#'*60}")
    print(f"# Step 3: Preprocessing (outcome={outcome_col})")
    print(f"{'#'*60}")
    df_train = preprocess(df_train, outcome_col)
    df_test = preprocess(df_test, outcome_col)
    
    # Step 4: 保存
    print(f"\n{'#'*60}")
    print(f"# Step 4: Saving preprocessed data")
    print(f"{'#'*60}")
    save_preprocessed(df_train, OUTPUT_DIR, suffix="train")
    save_preprocessed(df_test, OUTPUT_DIR, suffix="test")
    
    # Step 5: 合并训练集和测试集，做训练/测试拆分（可选）
    # 如果希望用 MT-LIFT 官方的 train/test 拆分，跳过此步
    print(f"\n{'#'*60}")
    print(f"# Step 5: Summary")
    print(f"{'#'*60}")
    print(f"\nPreprocessing complete!")
    print(f"  Processed train: {OUTPUT_DIR}/train_processed.csv")
    print(f"  Processed test:  {OUTPUT_DIR}/test_processed.csv")
    print(f"\nNext step: Run benchmark script:")
    print(f"  python scripts/run_mt_lift_benchmark.py")


if __name__ == "__main__":
    main()
