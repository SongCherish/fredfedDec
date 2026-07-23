import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置基础路径
base_path = Path(r'D:\project\python\FL\FL-bench\out')
dataset = 'fmnist'

# 定义要比较的方法列表（根据您的LaTeX表格）
methods = [
    'fredfedrep','fedavg', 'fedprox', 'moon', 'fedgen', 'fedrep', 'cfl',
    'fedper', 'fedbn', 'knnper', 'fedala', 'flute', 'floco'
]

# 方法的中文名称映射（与LaTeX表格保持一致）
method_names_cn = {
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx', 
    'moon': 'MOON',
    'fedgen': 'FedGen',
    'fedrep': 'FedRep',
    'cfl': 'CFL',
    'fedper': 'FedPer',
    'fedbn': 'FedBN',
    'knnper': 'KNN-Per',
    'fedala': 'FedALA',
    'flute': 'FLUTE',
    'floco': 'Floco',
'fredfedrep': 'Ours'
}

# 可选：通过命令行直接指定一个或多个 metrics.csv 文件来替代从 out/<method>/<dataset>/<timestamp>/metrics.csv 搜索
metrics_file_map = {}  # method_key -> Path to metrics.csv

# 如果希望直接在代码中声明要绘制的 metrics.csv 路径，可以在下面填写 DECLARED_METRICS。
# 例子：
# DECLARED_METRICS = {
#     'ours_run': r'D:\path\to\ours\metrics.csv',
#     'baseline': r'D:\path\to\baseline\metrics.csv'
# }
# 只需取消下面一行的注释并修改路径即可；若 DECLARED_METRICS 非空，脚本会使用其作为数据源并覆写 methods 列表。
DECLARED_METRICS = {
    'Ours':                 r'D:\project\python\FL\FL-bench\out\fredfedrep\cifar100\2025-05-24-11-24-19\metrics.csv',
    'FedAvg + Decoupled':   r'D:\project\python\FL\FL-bench\out\fedrep\cifar100\2025-05-24-12-31-31\metrics.csv',
    'FedAvg + Cluster':     r'D:\project\python\FL\FL-bench\out\fredfedrep\cifar100\2025-05-24-11-24-19\metrics.csv',
    'FedAvg':               r'D:\project\python\FL\FL-bench\out\fedavg\cifar100\2025-05-24-11-24-19\metrics.csv'
}

if DECLARED_METRICS:
    # 将声明的文件加入 metrics_file_map，并使用键作为 methods 列表
    metrics_file_map.update(DECLARED_METRICS)
    methods = list(DECLARED_METRICS.keys())
    # 若需要友好的显示名，可在此处为每个 key 补充到 method_names_cn
    for k in methods:
        method_names_cn.setdefault(k, k)

def get_latest_metrics(method):
    """获取指定方法的最新实验结果或直接从指定的 metrics.csv 文件读取"""
    # 优先检查是否通过命令行指定了 metrics 文件
    if metrics_file_map and method in metrics_file_map:
        metrics_file = Path(metrics_file_map[method])
        if metrics_file.exists():
            try:
                return pd.read_csv(metrics_file)
            except Exception as e:
                print(f"读取指定文件 {metrics_file} 的数据时出错: {e}")
                return None
        else:
            print(f"指定的 metrics 文件不存在: {metrics_file}")
            return None

    # 否则按原逻辑在 out/<method>/<dataset>/<timestamp>/metrics.csv 搜索
    method_path = base_path / method / dataset
    if not method_path.exists():
        return None
    
    # 获取所有时间戳目录并按时间排序，取最新的
    timestamp_dirs = [d for d in method_path.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        return None
    
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    metrics_file = latest_dir / 'metrics.csv'
    
    if metrics_file.exists():
        try:
            return pd.read_csv(metrics_file)
        except Exception as e:
            print(f"读取 {method} 的数据时出错: {e}")
            return None
    return None

def collect_final_results():
    """收集所有方法的最终测试结果"""
    results = []
    for method in methods:
        df = get_latest_metrics(method)
        if df is not None and not df.empty:
            final_test_acc = df['accuracy_test_after'].iloc[-1]
            best_test_acc = df['accuracy_test_after'].max()
            
            results.append({
                'method': method,
                'method_cn': method_names_cn.get(method, method),
                'final_test_acc': final_test_acc,
                'best_test_acc': best_test_acc
            })
    
    return pd.DataFrame(results)

# 1. 绘制方法性能对比柱状图
def plot_method_comparison(results_df):
    """绘制不同方法的性能对比柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 按最终准确率排序
    df_sorted = results_df.sort_values('final_test_acc', ascending=True)
    
    # 子图1: 最终测试准确率
    bars1 = ax1.barh(df_sorted['method_cn'], df_sorted['final_test_acc'], 
                     color='skyblue', alpha=0.8)
    ax1.set_title('不同联邦学习方法的最终测试准确率', fontsize=14, fontweight='bold')
    ax1.set_xlabel('测试准确率 (%)')
    ax1.set_xlim(0, 100)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontweight='bold')
    
    # 子图2: 最佳测试准确率
    df_sorted_best = results_df.sort_values('best_test_acc', ascending=True)
    bars2 = ax2.barh(df_sorted_best['method_cn'], df_sorted_best['best_test_acc'], 
                     color='lightcoral', alpha=0.8)
    ax2.set_title('不同联邦学习方法的最佳测试准确率', fontsize=14, fontweight='bold')
    ax2.set_xlabel('测试准确率 (%)')
    ax2.set_xlim(0, 100)
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# 2. 绘制学习曲线对比
def plot_learning_curves():
    """绘制多个方法的学习曲线对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 选择要显示的主要方法（根据LaTeX表格中的代表性方法）
    # 如果用户通过 DECLARED_METRICS 或命令行指定了 metrics 文件，则使用当前 methods（即指定的文件集合）
    if metrics_file_map:
        main_methods = methods
    else:
        main_methods = ['fedavg', 'fedprox', 'fedper', 'moon', 'fedrep', 'floco']
    # 为主方法生成颜色，长度与 main_methods 保持一致，避免与全局 methods 长度不匹配导致索引错误
    colors = plt.cm.tab20(np.linspace(0, 1, len(main_methods)))
    
    for i, method in enumerate(main_methods):
        df = get_latest_metrics(method)
        if df is not None:
            # 平滑处理
            window = min(5, len(df) // 4) if len(df) > 10 else 1
            if window > 1:
                df_smooth = df.rolling(window=window, center=True).mean()
            else:
                df_smooth = df
            
            # 测试准确率曲线
            ax1.plot(df_smooth['epoch'], df_smooth['accuracy_test_after'], 
                    label=method_names_cn[method], color=colors[i], linewidth=2, alpha=0.8)
            
            # 本地训练带来的提升
            improvement = df['accuracy_test_after'] - df['accuracy_test_before']
            if window > 1:
                improvement_smooth = improvement.rolling(window=window, center=True).mean()
            else:
                improvement_smooth = improvement
                
            ax2.plot(df_smooth['epoch'], improvement_smooth, 
                    label=method_names_cn[method], color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_title('主要方法的测试准确率学习曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('通信轮次')
    ax1.set_ylabel('测试准确率 (%)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('本地训练带来的准确率提升', fontsize=14, fontweight='bold')
    ax2.set_xlabel('通信轮次')
    ax2.set_ylabel('准确率提升 (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

# 3. 绘制方法类别性能对比
def plot_category_comparison(results_df):
    """按方法类别绘制性能对比"""
    # 将方法分类（根据LaTeX表格中的方法）
    traditional_fl = ['fedavg', 'fedprox', 'moon', 'fedgen', 'fedbn', 'fedala']
    personalized_fl = ['fedrep', 'fedper', 'knnper', 'cfl', 'flute', 'floco']
    
    categories = []
    category_names = []
    
    for method in results_df['method']:
        if method in traditional_fl:
            categories.append('传统联邦学习')
        elif method in personalized_fl:
            categories.append('个性化联邦学习')
        else:
            categories.append('其他方法')
        category_names.append(method_names_cn.get(method, method))
    
    results_df['category'] = categories
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按类别分组绘制
    categories_list = ['传统联邦学习', '个性化联邦学习', '其他方法']
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    x_pos = 0
    for cat_idx, category in enumerate(categories_list):
        cat_data = results_df[results_df['category'] == category].sort_values('final_test_acc', ascending=False)
        
        positions = np.arange(len(cat_data)) + x_pos
        bars = ax.bar(positions, cat_data['final_test_acc'], 
                     label=category, color=colors_list[cat_idx], alpha=0.8)
        
        # 添加方法名称标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   cat_data.iloc[i]['method_cn'], 
                   ha='center', va='bottom', rotation=45, fontsize=8)
        
        x_pos += len(cat_data) + 1
    
    ax.set_title('不同类别联邦学习方法性能对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('方法')
    ax.set_ylabel('最终测试准确率 (%)')
    ax.legend()
    ax.set_ylim(0, max(results_df['final_test_acc']) * 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# 4. 绘制性能提升对比
def plot_improvement_analysis():
    """分析不同方法的性能提升模式"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 选择几个代表性方法进行详细分析（根据LaTeX表格）
    representative_methods = ['fedavg', 'fedper', 'moon', 'fedrep']
    
    for idx, method in enumerate(representative_methods):
        ax = axes[idx // 2, idx % 2]
        df = get_latest_metrics(method)
        
        if df is not None:
            # 计算各种指标
            epochs = df['epoch']
            val_before = df['accuracy_val_before']
            val_after = df['accuracy_val_after']
            test_before = df['accuracy_test_before']
            test_after = df['accuracy_test_after']
            
            # 绘制多条曲线
            ax.plot(epochs, val_before, 'b--', label='验证集-训练前', alpha=0.7)
            ax.plot(epochs, val_after, 'b-', label='验证集-训练后', linewidth=2)
            ax.plot(epochs, test_before, 'r--', label='测试集-训练前', alpha=0.7)
            ax.plot(epochs, test_after, 'r-', label='测试集-训练后', linewidth=2)
            
            ax.set_title(f'{method_names_cn[method]} 详细性能分析', fontweight='bold')
            ax.set_xlabel('通信轮次')
            ax.set_ylabel('准确率 (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 主程序
def main():
    print("正在收集实验结果...")
    results_df = collect_final_results()
    
    if results_df.empty:
        print("未找到任何实验结果！")
        return
    
    print(f"成功收集到 {len(results_df)} 个方法的结果")
    print("\n各方法最终测试准确率:")
    for _, row in results_df.iterrows():
        print(f"{row['method_cn']}: {row['final_test_acc']:.2f}%")
    
    # 创建保存目录
    output_dir = Path(r'D:\project\python\FL\FL-bench\paper_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 绘制各种图表
    print("\n正在生成图表...")
    
    # 1. 方法性能对比柱状图
    fig1 = plot_method_comparison(results_df)
    fig1.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 方法性能对比图已保存")
    
    # 2. 学习曲线对比
    fig2 = plot_learning_curves()
    fig2.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 学习曲线对比图已保存")
    
    # 3. 方法类别对比
    fig3 = plot_category_comparison(results_df)
    fig3.savefig(output_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 方法类别对比图已保存")
    
    # 4. 性能提升分析
    fig4 = plot_improvement_analysis()
    fig4.savefig(output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 性能提升分析图已保存")
    
    # 保存数据表格
    results_df.to_csv(output_dir / 'methods_performance.csv', index=False, encoding='utf-8-sig')
    print("✓ 性能数据表格已保存")
    
    print(f"\n所有图表和数据已保存到: {output_dir}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制方法对比图，支持指定 metrics.csv 文件')
    parser.add_argument('--metrics-files', nargs='+', help='一个或多个 metrics.csv 文件的路径，用空格分隔')
    parser.add_argument('--labels', nargs='+', help='为每个 metrics 文件指定显示名称（可选，数量应与 metrics-files 对应）')
    parser.add_argument('--output-dir', default=r'D:\project\python\FL\FL-bench\paper_plots', help='保存图表和数据的目录')
    args = parser.parse_args()

    # 如果用户指定了 metrics 文件，则使用这些文件替代默认按方法名搜索的行为
    if args.metrics_files:
        metrics_files = [Path(p) for p in args.metrics_files]
        labels = args.labels if args.labels else None
        new_methods = []
        new_method_names_cn = {}
        for idx, mf in enumerate(metrics_files):
            key = mf.stem
            new_methods.append(key)
            metrics_file_map[key] = str(mf)
            if labels and idx < len(labels):
                new_method_names_cn[key] = labels[idx]
            else:
                new_method_names_cn[key] = mf.parent.name if mf.parent.name else mf.stem
        methods = new_methods
        method_names_cn.update(new_method_names_cn)

    # 支持通过命令行指定输出目录（main 内使用硬编码路径，若需要可进一步支持）
    main()