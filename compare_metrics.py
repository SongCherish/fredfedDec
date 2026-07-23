"""
compare_metrics.py

轻量脚本：从多个 metrics.csv 文件读取实验记录，生成论文所需的对比图并保存结果。
支持：
 - 通过 --metrics-files 指定一个或多个 metrics.csv 路径（必选）
 - 可选 --labels 为每个文件指定显示名称（数量应与文件数一致）
 - 可选 --output-dir 指定输出目录（默认 paper_plots）
 - 可选 --show 在生成后显示图表

生成：method_comparison.png, learning_curves.png, methods_performance.csv

假设 metrics.csv 包含列：epoch, accuracy_test_after, accuracy_test_before, accuracy_val_after, accuracy_val_before 等。
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style='whitegrid')

# If you prefer to declare metrics.csv paths directly in code, set DECLARED_METRICS below.
# You can provide either:
# 1) a dict mapping metrics.csv path -> label, e.g.
#    DECLARED_METRICS = { r'D:\path\run1\metrics.csv': 'Ours', r'D:\path\run2\metrics.csv': 'Baseline' }
# 2) or a list of paths and an optional DECLARED_LABELS list, e.g.
#    DECLARED_METRICS = [r'D:\path\run1\metrics.csv', r'D:\path\run2\metrics.csv']
#    DECLARED_LABELS = ['Ours', 'Baseline']
# When DECLARED_METRICS is non-empty, the script will use these and ignore command-line --metrics-files.
DECLARED_METRICS = {
      r'D:\project\python\FL\FL-bench\out\fredfedrep\cifar100\2025-05-24-11-24-19\metrics.csv': 'Ours',
      r'D:\project\python\FL\FL-bench\out\fedrep\cifar100\2025-05-24-12-31-31\metrics.csv': 'FedAvg + Decoupled',
      r'D:\project\python\FL\FL-bench\out\fredfedrep\cifar100\2025-05-24-11-24-19\metrics.csv': 'FedAvg + Cluster',
      r'D:\project\python\FL\FL-bench\out\fedavg\cifar100\2025-05-24-11-24-19\metrics.csv': 'FedAvg'
}
DECLARED_LABELS = []

# 注意：如果多个字典键（path）完全相同，后面的会覆盖前面的，因为 dict 键必须唯一。
# 下面提供一个小工具函数：若 accuracy 列是 0-1 范围，则把它放大到 0-100 以便显示为百分比。
def normalize_accuracy_columns(df):
    """如果准确率列在 0-1 之间，则放大到 0-100。"""
    cols = ['accuracy_test_after', 'accuracy_test_before', 'accuracy_val_after', 'accuracy_val_before', 'test_acc']
    for c in cols:
        if c in df.columns:
            try:
                vmax = df[c].max()
                if pd.notna(vmax) and vmax <= 1.01:
                    df[c] = df[c] * 100.0
            except Exception:
                # 如果列包含非数值，跳过转换
                pass
    return df


def load_metrics(paths, labels=None):
    """读取多个 metrics.csv，返回 (keys, labels, dict_of_dfs)
    keys 是内部方法 id，labels 是用于展示的名称（与路径顺序一致）
    """
    dfs = {}
    keys = []
    display_names = []

    for i, p in enumerate(paths):
        p = Path(p)
        if not p.exists():
            print(f"Warning: metrics file not found: {p}")
            continue
        # 读取并标准化可能的 0-1 范围准确率到 0-100
        try:
            df = pd.read_csv(p)
            df = normalize_accuracy_columns(df)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
            continue
        # 使用基于索引的唯一 key 避免不同文件使用相同的文件名 'metrics' 时被覆盖
        key = f"m{i}"
        keys.append(key)
        dfs[key] = df
        if labels and i < len(labels):
            display_names.append(labels[i])
        else:
            # 尝试使用父目录作为更友好的名字
            display_names.append(p.parent.name or p.stem)
    return keys, display_names, dfs


def summarize_results(keys, labels, dfs):
    rows = []
    for k, lab in zip(keys, labels):
        df = dfs.get(k)
        if df is None or df.empty:
            continue
        # 最后一个记录的测试准确率，以及历史最佳
        if 'accuracy_test_after' in df.columns:
            final_test_acc = df['accuracy_test_after'].iloc[-1]
            best_test_acc = df['accuracy_test_after'].max()
        elif 'test_acc' in df.columns:
            final_test_acc = df['test_acc'].iloc[-1]
            best_test_acc = df['test_acc'].max()
        else:
            final_test_acc = np.nan
            best_test_acc = np.nan

        rows.append({'method': k, 'label': lab, 'final_test_acc': final_test_acc, 'best_test_acc': best_test_acc})
    return pd.DataFrame(rows)


def plot_method_comparison(results_df, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    df1 = results_df.sort_values('final_test_acc')
    ax1.barh(df1['label'], df1['final_test_acc'], color='skyblue')
    ax1.set_title('Final Test Accuracy')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_xlim(0, 100)
    for bar in ax1.patches:
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.2f}%", va='center')

    df2 = results_df.sort_values('best_test_acc')
    ax2.barh(df2['label'], df2['best_test_acc'], color='lightcoral')
    ax2.set_title('Best Test Accuracy')
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_xlim(0, 100)
    for bar in ax2.patches:
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.2f}%", va='center')

    plt.tight_layout()
    out_path = Path(out_dir) / 'method_comparison.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    # show after saving
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_learning_curves(keys, labels, dfs, out_dir):
    # 绘制每个方法的 accuracy_test_after 曲线，以及 (test_after - test_before) 提升曲线
    n = len(keys)
    if n == 0:
        return
    colors = sns.color_palette('tab10' if n <= 10 else 'tab20', n_colors=n)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for i, (k, lab) in enumerate(zip(keys, labels)):
        df = dfs.get(k)
        if df is None or df.empty:
            continue
        if 'epoch' in df.columns and 'accuracy_test_after' in df.columns:
            x = df['epoch']
            y = df['accuracy_test_after']
            # 平滑（可选）
            window = min(5, max(1, len(df)//8))
            if window > 1:
                y_plot = y.rolling(window=window, center=True, min_periods=1).mean()
            else:
                y_plot = y
            ax1.plot(x, y_plot, label=lab, color=colors[i % len(colors)])

        # improvement
        if 'accuracy_test_after' in df.columns and 'accuracy_test_before' in df.columns:
            imp = df['accuracy_test_after'] - df['accuracy_test_before']
            if 'epoch' in df.columns:
                x = df['epoch']
                window = min(5, max(1, len(df)//8))
                if window > 1:
                    imp_plot = imp.rolling(window=window, center=True, min_periods=1).mean()
                else:
                    imp_plot = imp
                ax2.plot(x, imp_plot, label=lab, color=colors[i % len(colors)])

    ax1.set_title('Test Accuracy Learning Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title('Accuracy Improvement from Local Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Improvement (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    out1 = Path(out_dir) / 'learning_curves.png'
    out2 = Path(out_dir) / 'improvement_curves.png'
    fig1.savefig(out1, dpi=300, bbox_inches='tight')
    # show learning curves figure
    plt.show()
    plt.close(fig1)

    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    # show improvement figure
    plt.show()
    plt.close(fig2)

    print(f"Saved: {out1} and {out2}")


def main():
    parser = argparse.ArgumentParser(description='Compare methods from metrics.csv files')
    parser.add_argument('--metrics-files', nargs='+', required=False, help='paths to metrics.csv')
    parser.add_argument('--labels', nargs='+', help='display names for the files (optional)')
    parser.add_argument('--output-dir', default=Path.cwd() / 'paper_plots', help='output directory')
    parser.add_argument('--show', action='store_true', help='show plots after generating')
    args = parser.parse_args()

    # If DECLARED_METRICS is set in the file, use it and ignore CLI metrics-files
    if DECLARED_METRICS:
        # Accept either a dict mapping path->label or a list of paths.
        if isinstance(DECLARED_METRICS, dict):
            metrics_files = list(DECLARED_METRICS.keys())
            labels = list(DECLARED_METRICS.values())
        else:
            metrics_files = DECLARED_METRICS
            labels = DECLARED_LABELS if DECLARED_LABELS else args.labels
    else:
        metrics_files = args.metrics_files
        labels = args.labels

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, labels, dfs = load_metrics(metrics_files, labels=labels)

    if not keys:
        print('No valid metrics files provided. Exiting.')
        return

    results_df = summarize_results(keys, labels, dfs)
    results_df.to_csv(Path(out_dir) / 'methods_performance.csv', index=False, encoding='utf-8-sig')
    print(f"Saved performance CSV to: {Path(out_dir) / 'methods_performance.csv'}")

    plot_method_comparison(results_df, out_dir)
    plot_learning_curves(keys, labels, dfs, out_dir)
    # plt.show()





if __name__ == '__main__':
    main()
