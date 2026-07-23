import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = r'D:\project\python\FL\FL-bench\out\fedavg\fmnist\2025-05-05-15-09-52\metrics.csv'
data = pd.read_csv(file_path)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('FedAvg on Fashion-MNIST: 训练过程准确率变化', fontsize=16, fontweight='bold')

# 子图1: 验证集准确率对比
axes[0, 0].plot(data['epoch'], data['accuracy_val_before'], 'b-', label='本地训练前', alpha=0.7)
axes[0, 0].plot(data['epoch'], data['accuracy_val_after'], 'b-', label='本地训练后', linewidth=2)
axes[0, 0].set_title('验证集准确率对比')
axes[0, 0].set_xlabel('通信轮次 (Round)')
axes[0, 0].set_ylabel('准确率 (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 测试集准确率对比
axes[0, 1].plot(data['epoch'], data['accuracy_test_before'], 'r-', label='本地训练前', alpha=0.7)
axes[0, 1].plot(data['epoch'], data['accuracy_test_after'], 'r-', label='本地训练后', linewidth=2)
axes[0, 1].set_title('测试集准确率对比')
axes[0, 1].set_xlabel('通信轮次 (Round)')
axes[0, 1].set_ylabel('准确率 (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 子图3: 本地训练提升效果
val_improvement = data['accuracy_val_after'] - data['accuracy_val_before']
test_improvement = data['accuracy_test_after'] - data['accuracy_test_before']

axes[1, 0].plot(data['epoch'], val_improvement, 'g-', label='验证集提升', alpha=0.7)
axes[1, 0].plot(data['epoch'], test_improvement, 'orange', label='测试集提升', alpha=0.7)
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_title('本地训练带来的准确率提升')
axes[1, 0].set_xlabel('通信轮次 (Round)')
axes[1, 0].set_ylabel('准确率提升 (%)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 子图4: 最终准确率对比（仅显示训练后的准确率）
axes[1, 1].plot(data['epoch'], data['accuracy_val_after'], 'b-', label='验证集准确率', linewidth=2)
axes[1, 1].plot(data['epoch'], data['accuracy_teswot_after'], 'r-', label='测试集准确率', linewidth=2)
axes[1, 1].set_title('最终模型性能')
axes[1, 1].set_xlabel('通信轮次 (Round)')
axes[1, 1].set_ylabel('准确率 (%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 添加统计信息
final_val_acc = data['accuracy_val_after'].iloc[-1]
final_test_acc = data['accuracy_test_after'].iloc[-1]
max_val_acc = data['accuracy_val_after'].max()
max_test_acc = data['accuracy_test_after'].max()

stats_text = f'最终验证集准确率: {final_val_acc:.2f}%\n'
stats_text += f'最终测试集准确率: {final_test_acc:.2f}%\n'
stats_text += f'最高验证集准确率: {max_val_acc:.2f}%\n'
stats_text += f'最高测试集准确率: {max_test_acc:.2f}%'

axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图片
output_path = r'D:\project\python\FL\FL-bench\out\fedavg\fmnist\2025-05-05-15-09-52\training_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'图表已保存到: {output_path}')

# 显示图表
plt.show()

# 创建第二个更简洁的图表
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 绘制平滑曲线
window_size = 5
val_after_smooth = data['accuracy_val_after'].rolling(window=window_size).mean()
test_after_smooth = data['accuracy_test_after'].rolling(window=window_size).mean()

ax.plot(data['epoch'], val_after_smooth, 'b-', label='验证集准确率', linewidth=2)
ax.plot(data['epoch'], test_after_smooth, 'r-', label='测试集准确率', linewidth=2)

# 添加阴影区域表示标准差
val_std = data['accuracy_val_after'].rolling(window=window_size).std()
test_std = data['accuracy_test_after'].rolling(window=window_size).std()

ax.fill_between(data['epoch'], 
                val_after_smooth - val_std, 
                val_after_smooth + val_std, 
                alpha=0.2, color='blue')
ax.fill_between(data['epoch'], 
                test_after_smooth - test_std, 
                test_after_smooth + test_std, 
                alpha=0.2, color='red')

ax.set_title('FedAvg在Fashion-MNIST上的学习曲线（平滑后）', fontsize=14, fontweight='bold')
ax.set_xlabel('通信轮次 (Round)')
ax.set_ylabel('准确率 (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加最佳点标记
best_val_idx = data['accuracy_val_after'].idxmax()
best_test_idx = data['accuracy_test_after'].idxmax()

ax.plot(data.loc[best_val_idx, 'epoch'], data.loc[best_val_idx, 'accuracy_val_after'], 
        'bo', markersize=8, label=f'最佳验证: {data.loc[best_val_idx, "accuracy_val_after"]:.2f}%')
ax.plot(data.loc[best_test_idx, 'epoch'], data.loc[best_test_idx, 'accuracy_test_after'], 
        'ro', markersize=8, label=f'最佳测试: {data.loc[best_test_idx, "accuracy_test_after"]:.2f}%')

plt.tight_layout()

# 保存第二个图表
output_path2 = r'D:\project\python\FL\FL-bench\out\fedavg\fmnist\2025-05-05-15-09-52\smooth_curves.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f'平滑曲线图已保存到: {output_path2}')

plt.show()