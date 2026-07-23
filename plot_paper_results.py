import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 从您的LaTeX表格中提取的数据
data = {
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

df = pd.DataFrame(data)

# 方法分类
traditional_fl = ['FedAvg', 'FedProx', 'MOON', 'FedGen', 'FedBN', 'FedALA']
personalized_fl = ['FedRep', 'FedPer', 'KNN-Per', 'CFL', 'FLUTE', 'Floco']

# 为每个方法分配类别
method_categories = []
for method in df['Method']:
    if method == 'Ours':
        method_categories.append('Ours')
    elif method in traditional_fl:
        method_categories.append('Traditional FL')
    elif method in personalized_fl:
        method_categories.append('Personalized FL')
    else:
        method_categories.append('Other')

df['Category'] = method_categories

# 颜色设置
colors = {
    'Ours': '#FF6B6B',
    'Traditional FL': '#4ECDC4',
    'Personalized FL': '#45B7D1'
}

# 1. 主要性能对比图（三种数据集的柱状图）
def plot_main_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    datasets = ['FashionMNIST', 'CIFAR-10', 'CIFAR-100']
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # 按性能排序
        df_sorted = df.sort_values(dataset, ascending=True)
        
        # 创建颜色映射
        bar_colors = [colors[cat] for cat in df_sorted['Category']]
        
        # 绘制柱状图
        bars = ax.barh(df_sorted['Method'], df_sorted[dataset], 
                       color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 标记最优和次优
        max_val = df[dataset].max()
        second_max = df[dataset].sort_values(ascending=False).iloc[1]
        
        for j, bar in enumerate(bars):
            width = bar.get_width()
            method_name = df_sorted.iloc[j]['Method']
            value = df_sorted.iloc[j][dataset]
            
            # 添加数值标签
            if value >= 50:  # 高数值时放在柱子内
                ax.text(width - 2, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}%', ha='right', va='center', 
                       fontweight='bold', color='white', fontsize=10)
            else:  # 低数值时放在柱子外
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}%', ha='left', va='center', 
                       fontweight='bold', color='black', fontsize=10)
            
            # 标记最优方法
            if value == max_val:
                ax.add_patch(Rectangle((0, bar.get_y()), width, bar.get_height(), 
                                      fill=False, edgecolor='gold', linewidth=3))
            elif value == second_max and dataset == 'CIFAR-10':  # CIFAR-10有下划线
                ax.add_patch(Rectangle((0, bar.get_y()), width, bar.get_height(), 
                                      fill=False, edgecolor='orange', linewidth=2))
        
        ax.set_title(f'{dataset}\n(100 clients, 4 shards/client)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('准确率 (%)', fontsize=12)
        ax.set_xlim(0, max(df[dataset]) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
    
    # 添加图例
    legend_elements = [Rectangle((0,0),1,1, facecolor=colors['Ours'], alpha=0.8, label='Ours'),
                      Rectangle((0,0),1,1, facecolor=colors['Traditional FL'], alpha=0.8, label='传统联邦学习'),
                      Rectangle((0,0),1,1, facecolor=colors['Personalized FL'], alpha=0.8, label='个性化联邦学习')]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=3, fontsize=12)
    
    plt.suptitle('不同联邦学习方法在三个数据集上的性能对比', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

# 2. 性能提升分析图
def plot_improvement_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: 相对于FedAvg的提升
    ax1 = axes[0, 0]
    improvements_fedavg = []
    methods = []
    categories = []
    
    for _, row in df.iterrows():
        if row['Method'] != 'FedAvg':  # 排除FedAvg本身
            improvements = []
            for dataset in ['FashionMNIST', 'CIFAR-10', 'CIFAR-100']:
                improvement = row[dataset] - df[df['Method'] == 'FedAvg'][dataset].iloc[0]
                improvements.append(improvement)
            
            avg_improvement = np.mean(improvements)
            improvements_fedavg.append(avg_improvement)
            methods.append(row['Method'])
            categories.append(row['Category'])
    
    df_improvement = pd.DataFrame({
        'Method': methods,
        'Improvement': improvements_fedavg,
        'Category': categories
    })
    
    df_improvement_sorted = df_improvement.sort_values('Improvement', ascending=True)
    
    colors_list = [colors[cat] for cat in df_improvement_sorted['Category']]
    bars = ax1.barh(df_improvement_sorted['Method'], df_improvement_sorted['Improvement'], 
                   color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('相对于FedAvg的平均性能提升', fontsize=14, fontweight='bold')
    ax1.set_xlabel('平均提升 (%)')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 子图2: 数据集难度分析
    ax2 = axes[0, 1]
    dataset_difficulty = {
        'FashionMNIST': df['FashionMNIST'].mean(),
        'CIFAR-10': df['CIFAR-10'].mean(),
        'CIFAR-100': df['CIFAR-100'].mean()
    }
    
    bars = ax2.bar(dataset_difficulty.keys(), dataset_difficulty.values(), 
                   color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.8)
    ax2.set_title('数据集难度分析（所有方法平均准确率）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('平均准确率 (%)')
    ax2.set_ylim(0, max(dataset_difficulty.values()) * 1.2)
    
    for i, (dataset, acc) in enumerate(dataset_difficulty.items()):
        ax2.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontweight='bold')
    
    # 子图3: 方法类别性能对比
    ax3 = axes[1, 0]
    category_performance = df.groupby('Category')[['FashionMNIST', 'CIFAR-10', 'CIFAR-100']].mean()
    
    x = np.arange(len(category_performance.columns))
    width = 0.25
    
    for i, category in enumerate(category_performance.index):
        offset = (i - 1) * width
        bars = ax3.bar(x + offset, category_performance.loc[category], 
                      width, label=category, color=colors[category], alpha=0.8)
    
    ax3.set_title('不同类别方法的平均性能', fontsize=14, fontweight='bold')
    ax3.set_xlabel('数据集')
    ax3.set_ylabel('平均准确率 (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(category_performance.columns)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 性能分布箱线图
    ax4 = axes[1, 1]
    
    # 重塑数据用于箱线图
    df_melted = pd.melt(df, id_vars=['Method', 'Category'], 
                        value_vars=['FashionMNIST', 'CIFAR-10', 'CIFAR-100'],
                        var_name='Dataset', value_name='Accuracy')
    
    sns.boxplot(data=df_melted, x='Dataset', y='Accuracy', hue='Category', 
                palette=colors, ax=ax4, width=0.6)
    
    ax4.set_title('不同方法类别在三个数据集上的性能分布', fontsize=14, fontweight='bold')
    ax4.set_ylabel('准确率 (%)')
    ax4.legend(title='方法类别', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('联邦学习方法性能深度分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# 3. 雷达图比较
def plot_radar_comparison():
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 选择代表性方法进行比较
    representative_methods = ['Ours', 'FedAvg', 'FedPer', 'FedRep']
    datasets = ['FashionMNIST', 'CIFAR-10', 'CIFAR-100']
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 绘制每个方法的雷达图
    method_colors = {
        'Ours': '#FF6B6B',
        'FedAvg': '#4ECDC4', 
        'FedPer': '#45B7D1',
        'FedRep': '#96CEB4'
    }
    
    for method in representative_methods:
        values = []
        for dataset in datasets:
            accuracy = df[df['Method'] == method][dataset].iloc[0]
            # 归一化到0-100范围（已经在0-100，所以直接使用）
            values.append(accuracy)
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, 
               color=method_colors[method])
        ax.fill(angles, values, alpha=0.1, color=method_colors[method])
    
    # 设置图表
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 100)
    ax.set_ylabel('准确率 (%)')
    ax.set_title('代表性方法性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    return fig

# 4. 热力图
def plot_heatmap():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备热力图数据
    heatmap_data = df.set_index('Method')[['FashionMNIST', 'CIFAR-10', 'CIFAR-100']]
    
    # 创建热力图
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                cbar_kws={'label': '准确率 (%)'}, ax=ax, 
                linewidths=0.5, linecolor='black')
    
    # 标记最优值
    for i, dataset in enumerate(['FashionMNIST', 'CIFAR-10', 'CIFAR-100']):
        max_val = df[dataset].max()
        max_row = df[df[dataset] == max_val].index[0]
        ax.add_patch(Rectangle((i, max_row), 1, 1, fill=False, 
                              edgecolor='gold', linewidth=3))
    
    ax.set_title('联邦学习方法性能热力图', fontsize=16, fontweight='bold')
    ax.set_ylabel('方法')
    ax.set_xlabel('数据集')
    
    return fig

# 主函数
def main():
    output_dir = r'D:\project\python\FL\FL-bench\paper_plots'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成论文图表...")
    
    # 1. 主要性能对比图
    fig1 = plot_main_comparison()
    fig1.savefig(f'{output_dir}/main_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 主要性能对比图已保存")
    
    # 2. 性能提升分析
    fig2 = plot_improvement_analysis()
    fig2.savefig(f'{output_dir}/improvement_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 性能提升分析图已保存")
    
    # 3. 雷达图
    fig3 = plot_radar_comparison()
    fig3.savefig(f'{output_dir}/radar_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 雷达对比图已保存")
    
    # 4. 热力图
    fig4 = plot_heatmap()
    fig4.savefig(f'{output_dir}/heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 热力图已保存")
    
    # 保存原始数据
    df.to_csv(f'{output_dir}/paper_table_data.csv', index=False, encoding='utf-8-sig')
    print("✓ 原始数据已保存")
    
    print(f"\n所有图表已保存到: {output_dir}")
    print("\n图表说明：")
    print("1. main_comparison.png - 主要结果对比，适合放在论文正文")
    print("2. improvement_analysis.png - 深度分析，适合放在实验分析部分")
    print("3. radar_comparison.png - 雷达图对比，适合展示多维度性能")
    print("4. heatmap.png - 热力图，直观展示整体性能分布")
    
    plt.show()

if __name__ == "__main__":
    main()