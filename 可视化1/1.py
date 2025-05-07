import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建数据
features = [
    "能源消费总量(万吨标准煤)",
    "煤炭占能源消费总量的比重(%)",
    "石油占能源消费总量的比重(%)",
    "天然气占能源消费总量的比重(%)",
    "一次电力能源总量的比重(%)",
    "煤炭消费量(万吨)",
    "焦炭消费量(万吨)",
    "原油消费量(万吨)",
    "汽油消费量(万吨)",
    "煤油消费量(万吨)",
    "柴油消费量(万吨)",
    "燃料油消费量(万吨)",
    "天然气消费量(亿立方米)",
    "电力消费量(亿千瓦小时)",
    "国内生产总值(亿元)",
    "第二产业增加值(亿元)",
    "人均国内生产总值(元)"
]

importance = [
    0.10, 1.00, 3.50, 0.80, 10.90,
    10.00, 11.70, 5.60, 15.50, 0.90,
    0.50, 18.20, 0.50, 0.10, 4.60,
    15.40, 0.80
]

# 创建DataFrame
df = pd.DataFrame({
    '特征': features,
    '重要性': importance
})

# 按重要性排序
df = df.sort_values('重要性', ascending=False)

# 1. 特征重要性热力图
plt.figure(figsize=(16, 4))
# 创建自定义颜色映射 - 从浅到深
colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
cmap = LinearSegmentedColormap.from_list("custom_blue", colors)

# 创建热力图数据
heatmap_data = pd.DataFrame(df['重要性'].values.reshape(1, -1),
                           columns=df['特征'])

# 绘制热力图
ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmap,
                linewidths=1, linecolor='white', cbar_kws={'label': '特征重要性'})
plt.title('GBDT模型特征重要性系数矩阵', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks([])  # 隐藏y轴标签
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT特征重要性系数矩阵.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 水平条形图 - 特征重要性
plt.figure(figsize=(12, 10))
bars = plt.barh(df['特征'], df['重要性'], color='steelblue')

# 在条形上添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
             ha='left', va='center', fontsize=10)

plt.xlabel('重要性')
plt.title('GBDT模型 - 特征重要性排序')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT特征重要性条形图.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 模拟一些预测数据用于可视化
np.random.seed(42)
n_samples = 50
y_true = np.random.normal(50, 10, n_samples)
y_pred = y_true + np.random.normal(0, 5, n_samples)

# 预测值与真实值对比散点图
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_true, y_pred, alpha=0.7, c='steelblue', s=60)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

# 计算R²和MSE用于显示
from sklearn.metrics import r2_score, mean_squared_error
test_r2 = r2_score(y_true, y_pred)
test_mse = mean_squared_error(y_true, y_pred)

# 添加R²值直接显示在图上
r2_text = f'R² = {test_r2:.4f}'
plt.annotate(r2_text, xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# 添加MSE值
mse_text = f'MSE = {test_mse:.4f}'
plt.annotate(mse_text, xy=(0.05, 0.89), xycoords='axes fraction',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('GBDT模型 - 预测值与真实值对比', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT预测值与真实值对比.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 残差分析图
residuals = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, c='steelblue')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('GBDT模型 - 残差分析')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT残差分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 模拟系数与置信区间
np.random.seed(42)
coefficients = np.random.normal(0, 1, len(features))
confidence = np.random.uniform(0.1, 0.5, len(features))

# 创建系数DataFrame
coef_df = pd.DataFrame({
    '特征': features,
    '系数': coefficients,
    '置信区间': confidence
})
coef_df = coef_df.sort_values('系数', key=abs, ascending=False)

plt.figure(figsize=(12, 8))
plt.errorbar(coef_df['特征'], coef_df['系数'],
            yerr=coef_df['置信区间'],
            fmt='o', capsize=5, color='steelblue')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel('特征')
plt.ylabel('系数值')
plt.title('GBDT模型 - 系数及其不确定性')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT系数及不确定性.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 三维柱状图 - 特征重要性
# 选择前10个最重要的特征
top_features = df.head(10)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置柱状图的位置
x_pos = np.arange(len(top_features))
y_pos = np.zeros_like(x_pos)
z_pos = np.zeros_like(x_pos)
dx = 0.8  # 柱宽
dy = 0.8  # 柱深
dz = top_features['重要性'].values  # 柱高

# 创建自定义颜色映射 - 使用更鲜艳的颜色
custom_cmap = plt.cm.get_cmap('viridis', len(top_features))
colors = [custom_cmap(i) for i in range(len(top_features))]

# 绘制3D柱状图
bars = ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True, alpha=0.8)

# 在每个柱子上方添加数值标签
for i, (x, y, z) in enumerate(zip(x_pos, y_pos, dz)):
    ax.text(x + dx/2, y + dy/2, z + 0.1, f'{z:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 设置坐标轴标签
ax.set_ylabel('', labelpad=10)
ax.set_zlabel('重要性', fontsize=12, labelpad=10)

# 设置标题
ax.set_title('特征重要性三维柱形图（前10个特征）', fontsize=16, pad=20)

# 调整视角 - 使用更好的视角
ax.view_init(elev=35, azim=30)

# 添加图例 - 将完整特征名称作为图例
legend_elements = [plt.Rectangle((0,0), 1, 1, color=colors[i]) for i in range(len(top_features))]
ax.legend(legend_elements, top_features['特征'],
          loc='upper right', bbox_to_anchor=(1.1, 0.9), fontsize=8)

# 保存图像
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT特征重要性三维柱形图.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 预测值与真实值的时间序列图
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_true)), y_true, 'b-', label='真实值', alpha=0.7)
plt.plot(range(len(y_pred)), y_pred, 'r--', label='预测值', alpha=0.7)
plt.xlabel('样本索引')
plt.ylabel('值')
plt.title('GBDT模型 - 预测值与真实值时间序列对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT预测值与真实值时间序列.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 模型学习曲线 (模拟)
iterations = np.arange(1, 101)
train_error = 1 / (0.1 * iterations) + 0.2 + np.random.normal(0, 0.05, 100)
test_error = 1 / (0.1 * iterations) + 0.3 + np.random.normal(0, 0.05, 100)

plt.figure(figsize=(10, 6))
plt.plot(iterations, train_error, 'b-', label='训练误差')
plt.plot(iterations, test_error, 'r-', label='测试误差')
plt.xlabel('迭代次数')
plt.ylabel('误差')
plt.title('GBDT模型 - 学习曲线')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT学习曲线.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 桑基图
# 准备桑基图数据
source = []
target = []
value = []
label = features + ['预测结果']

# 创建连接关系
for i, imp in enumerate(importance):
    if imp > 0.5:  # 只显示重要性大于0.5的特征，避免图表过于复杂
        source.append(i)
        target.append(len(features))  # 目标节点为最后一个节点
        value.append(imp)

# 创建桑基图
fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = label,
        color = "blue"
    ),
    link = dict(
        source = source,
        target = target,
        value = value
    )
)])

# 更新布局
fig.update_layout(
    title_text="GBDT特征重要性桑基图",
    font_size=10,
    height=800
)

# 保存为HTML文件以保持交互性
pio.write_html(fig, r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT特征重要性桑基图.html')

# 10. 相关性热力图 (模拟)
# 生成随机相关性矩阵
np.random.seed(42)
corr_matrix = np.random.uniform(-1, 1, (len(features), len(features)))
# 确保对角线为1
np.fill_diagonal(corr_matrix, 1)
# 确保矩阵对称
corr_matrix = (corr_matrix + corr_matrix.T) / 2

# 创建相关性DataFrame
correlation_matrix = pd.DataFrame(corr_matrix, columns=features, index=features)

# 创建热力图
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix,
            annot=True,  # 显示数值
            fmt='.2f',   # 数值格式为两位小数
            cmap='coolwarm',  # 使用红蓝色图
            center=0,    # 将相关性0值设为白色
            square=True, # 保持方形
            linewidths=0.5,
            cbar_kws={"shrink": .5})

plt.title('GBDT模型 - 特征相关性热力图', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\GBDT特征相关性热力图.png', dpi=300, bbox_inches='tight')
plt.show()