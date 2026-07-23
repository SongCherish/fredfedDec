import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 从LaTeX表格中提取的准确率数据
latex_data = {
    'Method': [
        'Ours', 'FedAvg', 'FedProx', 'MOON', 'FedGen', 'FedRep', 'CFL',
        'FedPer', 'FedBN', 'KNN-Per', 'FedALA', 'FLUTE', 'Floco'
    ],
    'FashionMNIST': [91.65, 80.46, 80.20, 80.54, 82.59, 90.31, 74.16, 
                   91.49, 80.46, 86.74, 80.51, 84.31, 81.14],
    'CIFAR-10': [66.31, 40.71, 40.52, 9.83, 38.06, 63.58, 40.71,
                67.99, 44.08, 61.44, 54.08, 50.89, 63.86],
    'CIFAR-100': [72.47, 10.08, 8.27, 9.90, 10.80, 68.47, 10.04,
                 65.20, 11.04, 39.10, 9.60, 51.39, 37.92]
}

# 方法内部名称映射（用于查找实验结果）
method_internal_names = {
    'Ours': None,  # 您需要填入自己的方法名
    'FedAvg': 'fedavg',
    'FedProx': 'fedprox', 
    'MOON': 'moon',
    'FedGen': 'fedgen',
    'FedRep': 'fedrep',
    'CFL': 'cfl',
    'FedPer': 'fedper',
    'FedBN': 'fedbn',
    'KNN-Per': 'knnper',
    'FedALA': 'fedala',
    'FLUTE': 'flute',
    'Floco': 'floco'
}

# 方法的中文名称映射
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
    'ours': 'Ours'  # 假设您的内部方法名是ours
}

# 设置基础路径
base_path = Path(r'D:\project\python\FL\FL-bench\out')
datasets = ['fmnist', 'cifar10', 'cifar100']

def get_latest_metrics(method, dataset):
    """获取指定方法和数据集的最新实验结果"""
    if method is None:  # 对于您的方法，如果没有实验结果则返回None
        return None
        
    method_path = base_path / method / dataset
    if not method_path.exists():
        print(f"警告：未找到 {method}/{dataset} 的实验结果")
        return None
    
    # 获取所有时间戳目录并按时间排序，取最新的
    timestamp_dirs = [d for d in method_path.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        print(f"警告：{method}/{dataset} 目录下没有时间戳子目录")
        return None
    
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    metrics_file = latest_dir / 'metrics.csv'
    
    if metrics_file.exists():
        try:
            return pd.read_csv(metrics_file)
        except Exception as e:
            print(f"读取 {method}/{dataset} 的数据时出错: {e}")
            return None
    print(f"警告：未找到 {method}/{dataset} 的metrics.csv文件")
    return None

def collect_experiment_results():
    """收集实验结果并与LaTeX表格数据对比"""
    results = []
    
    for latex_method, latex_data_row in zip(latex_data['Method'], 
                                            [latex_data['FashionMNIST'], 
                                             latex_data['CIFAR-10'], 
                                             latex_data['CIFAR-100']]):
        
        internal_name = method_internal_names.get(latex_method)
        
        for idx, (dataset_name, latex_acc) in enumerate(zip(
            ['fmnist', 'cifar10', 'cifar100'], 
            latex_data_row)):
            
            result_row = {
                'latex_method': latex_method,
                'internal_name': internal_name,
                'dataset': dataset_name,
                'latex_accuracy': latex_acc,
                'experiment_final_acc': None,
                'experiment_best_acc': None,
                'has_experiment': False
            }
            
            # 尝试获取实验结果
            if internal_name:
                df = get_latest_metrics(internal_name, dataset_name)
                if df is not None and not df.empty:
                    result_row['experiment_final_acc'] = df['accuracy_test_after'].iloc[-1]
                    result_row['experiment_best_acc'] = df['accuracy_test_after'].max()
                    result_row['has_experiment'] = True
            
            results.append(result_row)
    
    return pd.DataFrame(results)

def create_paper_plots():
    """基于LaTeX表格创建论文图表"""
    df_latex = pd.DataFrame(latex_data)
    
    # 创建输出目录
    output_dir = Path(r'D:\project\python\FL\FL-bench\paper_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 1. 主要性能对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    datasets = ['FashionMNIST', 'CIFAR-10', 'CIFAR-100']
    dataset_configs = ['(100 clients, 4 shards/client, 2nn)', 
                      '(100 clients, 4 shards/client, lenet5)', 
                      '(100 clients, 4 shards/client, lenet5)']
    
    for i, (dataset, config) in enumerate(zip(datasets, dataset_configs)):
        ax = axes[i]
        
        # 按性能排序
        df_sorted = df_latex.sort_values(dataset, ascending=True)
        
        # 创建颜色映射
        colors = []
        for method in df_sorted['Method']:
            if method == 'Ours':
                colors.append('#FF6B6B')  # 红色突出显示
            elif method in ['FedAvg', 'FedProx', 'MOON', 'FedGen', 'FedBN', 'FedALA']:
                colors.append('#4ECDC4')  # 传统方法用青色
            else:
                colors.append('#45B7D1')  # 个性化方法用蓝色
        
        # 绘制柱状图
        bars = ax.barh(df_sorted['Method'], df_sorted[dataset], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 标记最优值
        max_val = df_latex[dataset].max()
        second_max = df_latex[dataset].sort_values(ascending=False).iloc[1]
        
        for j, bar in enumerate(bars):
            width = bar.get_width()
            method_name = df_sorted.iloc[j]['Method']
            value = df_sorted.iloc[j][dataset]
            
            # 添加数值标签
            if value >= 50:
                ax.text(width - 1, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}%', ha='right', va='center', 
                       fontweight='bold', color='white', fontsize=10)
            else:
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}%', ha='left', va='center', 
                       fontweight='bold', color='black', fontsize=10)
            
            # 标记最优方法
            if value == max_val:
                ax.add_patch(Rectangle((0, bar.get_y()), width, bar.get_height(), 
                                      fill=False, edgecolor='gold', linewidth=3))
            elif value == second_max and dataset == 'CIFAR-10':  # CIFAR-10有下划线标记
                ax.add_patch(Rectangle((0, bar.get_y()), width, bar.get_height(), 
                                      fill=False, edgecolor='orange', linewidth=2))
        
        ax.set_title(f'{dataset}\n{config}', fontsize=12, fontweight='bold')
        ax.set_xlabel('准确率 (%)', fontsize=10)
        ax.set_xlim(0, max(df_latex[dataset]) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8, label='Ours'),
        plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8, label='传统联邦学习'),
        plt.Rectangle((0,0),1,1, facecolor='#45B7D1', alpha=0.8, label='个性化联邦学习')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=3, fontsize=12)
    
    plt.suptitle('不同联邦学习方法在三个数据集上的性能对比', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 保存主要对比图
    plt.savefig(output_dir / 'latex_main_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ LaTeX表格主要对比图已保存")
    plt.show()
    
    # 2. 性能提升分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 相对于FedAvg的提升
    ax1 = axes[0, 0]
    improvements = []
    method_names = []
    
    fedavg_row = df_latex[df_latex['Method'] == 'FedAvg'].iloc[0]
    
    for _, row in df_latex.iterrows():
        if row['Method'] != 'FedAvg':
            improvement = np.mean([
                row['FashionMNIST'] - fedavg_row['FashionMNIST'],
                row['CIFAR-10'] - fedavg_row['CIFAR-10'],
                row['CIFAR-100'] - fedavg_row['CIFAR-100']
            ])
            improvements.append(improvement)
            method_names.append(row['Method'])
    
    df_improvement = pd.DataFrame({
        'Method': method_names,
        'Improvement': improvements
    }).sort_values('Improvement', ascending=True)
    
    colors_list = []
    for method in df_improvement['Method']:
        if method == 'Ours':
            colors_list.append('#FF6B6B')
        elif method in ['FedAvg', 'FedProx', 'MOON', 'FedGen', 'FedBN', 'FedALA']:
            colors_list.append('#4ECDC4')
        else:
            colors_list.append('#45B7D1')
    
    bars = ax1.barh(df_improvement['Method'], df_improvement['Improvement'], 
                   color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('相对于FedAvg的平均性能提升', fontsize=14, fontweight='bold')
    ax1.set_xlabel('平均提升 (%)')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 数据集难度分析
    ax2 = axes[0, 1]
    dataset_means = {
        'FashionMNIST': df_latex['FashionMNIST'].mean(),
        'CIFAR-10': df_latex['CIFAR-10'].mean(),
        'CIFAR-100': df_latex['CIFAR-100'].mean()
    }
    
    bars = ax2.bar(dataset_means.keys(), dataset_means.values(), 
                   color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.8)
    ax2.set_title('数据集难度分析（所有方法平均准确率）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('平均准确率 (%)')
    ax2.set_ylim(0, max(dataset_means.values()) * 1.2)
    
    for i, (dataset, acc) in enumerate(dataset_means.items()):
        ax2.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontweight='bold')
    
    # 方法类别性能分布
    ax3 = axes[1, 0]
    traditional_methods = ['FedAvg', 'FedProx', 'MOON', 'FedGen', 'FedBN', 'FedALA']
    personalized_methods = ['FedRep', 'FedPer', 'KNN-Per', 'CFL', 'FLUTE', 'Floco']
    
    categories = ['Ours'] + ['Traditional FL'] * len(traditional_methods) + ['Personalized FL'] * len(personalized_methods)
    all_methods = ['Ours'] + traditional_methods + personalized_methods
    
    for dataset in ['FashionMNIST', 'CIFAR-10', 'CIFAR-100']:
        values = []
        for method in all_methods:
            values.append(df_latex[df_latex['Method'] == method][dataset].iloc[0])
        
        x = np.arange(len(all_methods))
        offset = ['FashionMNIST', 'CIFAR-10', 'CIFAR-100'].index(dataset) * 0.25
        ax3.scatter(x + offset, values, label=dataset, alpha=0.7, s=60)
    
    ax3.set_title('所有方法在三个数据集上的性能分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('方法')
    ax3.set_ylabel('准确率 (%)')
    ax3.set_xticks(range(len(all_methods)))
    ax3.set_xticklabels(all_methods, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 箱线图
    ax4 = axes[1, 1]
    df_melted = pd.melt(df_latex, id_vars=['Method'], 
                        value_vars=['FashionMNIST', 'CIFAR-10', 'CIFAR-100'],
                        var_name='Dataset', value_name='Accuracy')
    
    categories_mapped = []
    for method in df_melted['Method']:
        if method == 'Ours':
            categories_mapped.append('Ours')
        elif method in traditional_methods:
            categories_mapped.append('Traditional FL')
        else:
            categories_mapped.append('Personalized FL')
    
    df_melted['Category'] = categories_mapped
    
    sns.boxplot(data=df_melted, x='Dataset', y='Accuracy', hue='Category', 
                palette={'Ours': '#FF6B6B', 'Traditional FL': '#4ECDC4', 'Personalized FL': '#45B7D1'}, 
                ax=ax4, width=0.6)
    
    ax4.set_title('不同类别方法在三个数据集上的性能分布', fontsize=14, fontweight='bold')
    ax4.set_ylabel('准确率 (%)')
    ax4.legend(title='方法类别', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('联邦学习方法性能深度分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存分析图
    plt.savefig(output_dir / 'latex_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ LaTeX表格性能分析图已保存")
    plt.show()
    
    # 3. 保存LaTeX数据
    df_latex.to_csv(output_dir / 'latex_table_data.csv', index=False, encoding='utf-8-sig')
    print("✓ LaTeX表格数据已保存")
    
    return output_dir

def compare_experiment_with_latex():
    """比较实验结果与LaTeX表格数据"""
    print("正在比较实验结果与LaTeX表格数据...")
    
    experiment_results = collect_experiment_results()
    output_dir = Path(r'D:\project\python\FL\FL-bench\paper_plots')
    
    if experiment_results.empty:
        print("没有找到实验结果数据")
        return
    
    # 筛选出有实验结果的数据
    with_experiment = experiment_results[experiment_results['has_experiment'] == True]
    
    if with_experiment.empty:
        print("没有找到任何有效的实验结果")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, dataset in enumerate(['fmnist', 'cifar10', 'cifar100']):
        ax = axes[i]
        dataset_data = with_experiment[with_experiment['dataset'] == dataset]
        
        if dataset_data.empty:
            ax.text(0.5, 0.5, f'No data for {dataset}', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        methods = dataset_data['latex_method'].tolist()
        latex_accs = dataset_data['latex_accuracy'].tolist()
        exp_accs = dataset_data['experiment_final_acc'].tolist()
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, latex_accs, width, label='LaTeX表格数据', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, exp_accs, width, label='实验结果', 
                      color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('方法')
        ax.set_ylabel('准确率 (%)')
        ax.set_title(f'{dataset.upper()} - 表格数据 vs 实验结果')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('LaTeX表格数据与实际实验结果对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存对比图
    plt.savefig(output_dir / 'latex_vs_experiment_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ LaTeX与实验结果对比图已保存")
    plt.show()
    
    # 保存对比数据
    with_experiment.to_csv(output_dir / 'latex_vs_experiment_data.csv', index=False, encoding='utf-8-sig')
    print("✓ 对比数据已保存")

def main():
    print("基于LaTeX表格创建论文图表...")
    
    # 1. 创建基于LaTeX表格的图表
    output_dir = create_paper_plots()
    
    print(f"图表已保存到: {output_dir}")
    
    # 2. 尝试比较实验结果（如果有的话）
    try:
        compare_experiment_with_latex()
    except Exception as e:
        print(f"比较实验结果时出错: {e}")
        print("这可能是正常的，如果某些方法没有运行实验的话")
    
    print("\n图表说明：")
    print("1. latex_main_comparison.png - 基于LaTeX表格的主要对比图（论文首选）")
    print("2. latex_performance_analysis.png - 性能深度分析图")
    print("3. latex_table_data.csv - LaTeX表格原始数据")
    print("4. latex_vs_experiment_comparison.png - 实验结果对比（如果有实验数据）")

if __name__ == "__main__":
    # 需要导入Rectangle用于绘制边框
    from matplotlib.patches import Rectangle
    main()