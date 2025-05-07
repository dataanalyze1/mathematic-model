import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.io as pio

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建特征名称和重要性数据
features = [
    "创新研发专利数_个",
    "创新产业业务收入_亿元",
    "创新产品经费_亿元",
    "技术效率_万/人",
    "技术研发人数_万人",
    "技术生产_台",
    "能源强度_吨标准煤/万元_",
    "能源结构_吨标准煤/万元",
    "用水强度_立方米/万元",
    "废物利用率_%",
    "废水排放_吨/万元",
    "废气排放_吨/万元",
    "电子信息制造_万块",
    "电信业务通讯_亿元",
    "网络普及率_万个",
    "软件服务费用_亿元",
    "数字信息_公里",
    "电子商务_亿元"
]

importance = [
    0.600,  # 创新研发专利数
    0.100,  # 创新产业业务收入
    0.400,  # 创新产品经费
    0.200,  # 技术效率
    11.900,  # 技术研发人数
    2.900,  # 技术生产
    15.600,  # 能源强度
    0.000,  # 能源结构
    16.000,  # 用水强度
    0.100,  # 废物利用率
    17.900,  # 废水排放
    0.100,  # 废气排放
    23.000,  # 电子信息制造
    0.200,  # 电信业务通讯
    0.900,  # 网络普及率
    9.300,  # 软件服务费用
    0.200,  # 数字信息
    0.300   # 电子商务
]

# 创建模拟数据集
np.random.seed(42)  # 设置随机种子以确保结果可重复
n_samples = 100  # 样本数量

# 创建特征矩阵X，每个特征的值在0-1之间随机生成
X = np.random.rand(n_samples, len(features))
# 创建目标变量y，使用importance作为权重
y = np.dot(X, importance) + np.random.normal(0, 5, n_samples)  # 添加一些噪声

# 将数据转换为DataFrame以便于处理
df = pd.DataFrame(X, columns=features)
df['目标变量'] = y

# 数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化并训练贝叶斯岭回归模型
br = BayesianRidge(max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                   lambda_1=1e-6, lambda_2=1e-6, compute_score=True)
br.fit(X_train_scaled, y_train)

# 模型评估
y_pred_train = br.predict(X_train_scaled)
y_pred_test = br.predict(X_test_scaled)

# 计算各种评估指标
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# 交叉验证
cv_scores = cross_val_score(br, X_train_scaled, y_train, cv=5, scoring='r2')

# 打印评估结果
print("模型评估结果:")
print(f"训练集 MSE: {train_mse:.4f}")
print(f"测试集 MSE: {test_mse:.4f}")
print(f"训练集 MAE: {train_mae:.4f}")
print(f"测试集 MAE: {test_mae:.4f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")
print(f"5折交叉验证 R² 平均值: {np.mean(cv_scores):.4f}")
print(f"5折交叉验证 R² 标准差: {np.std(cv_scores):.4f}")

# 获取模型系数和特征重要性
coefficients = br.coef_
feature_importance = np.abs(coefficients)  # 使用系数的绝对值作为特征重要性

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    '特征': features,
    '系数': coefficients,
    '重要性': feature_importance,
    '真实重要性': importance
})
importance_df = importance_df.sort_values('重要性', ascending=False)

# 1. 特征重要性热力图
plt.figure(figsize=(16, 4))
# 创建自定义颜色映射 - 从浅到深
colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
cmap = LinearSegmentedColormap.from_list("custom_blue", colors)

# 创建热力图数据
heatmap_data = pd.DataFrame(importance_df['重要性'].values.reshape(1, -1),
                           columns=importance_df['特征'])

# 绘制热力图
ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmap,
                linewidths=1, linecolor='white', cbar_kws={'label': '特征重要性'})
plt.title('贝叶斯岭回归特征重要性系数矩阵', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks([])  # 隐藏y轴标签
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\特征重要性系数矩阵.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 水平条形图 - 特征重要性
plt.figure(figsize=(12, 10))
bars = plt.barh(importance_df['特征'], importance_df['重要性'], color='steelblue')

# 在条形上添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
             ha='left', va='center', fontsize=10)

plt.xlabel('重要性')
plt.title('贝叶斯岭回归模型 - 特征重要性排序')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\特征重要性条形图.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 预测值与真实值对比散点图 - 优化版
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_test, y_pred_test, alpha=0.7, c='steelblue', s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

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
plt.title('贝叶斯岭回归模型 - 预测值与真实值对比', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\预测值与真实值对比.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 残差分析图
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('贝叶斯岭回归模型 - 残差分析')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\残差分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 模型系数与置信区间图
plt.figure(figsize=(12, 8))
plt.errorbar(importance_df['特征'], importance_df['系数'],
            yerr=np.sqrt(np.diag(br.sigma_))[np.argsort(feature_importance)[::-1]],
            fmt='o', capsize=5)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel('特征')
plt.ylabel('系数值')
plt.title('贝叶斯岭回归模型 - 系数及其不确定性')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\系数及不确定性.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 三维柱状图 - 特征重要性 - 优化版
# 选择前10个最重要的特征
top_features = importance_df.head(10)

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
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\特征重要性三维柱形图.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 预测值与真实值的时间序列图
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, 'b-', label='真实值', alpha=0.7)
plt.plot(range(len(y_pred_test)), y_pred_test, 'r--', label='预测值', alpha=0.7)
plt.xlabel('样本索引')
plt.ylabel('值')
plt.title('贝叶斯岭回归模型 - 预测值与真实值时间序列对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\预测值与真实值时间序列.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 模型预测与真实重要性对比
plt.figure(figsize=(12, 8))
x = np.arange(len(features))
width = 0.35

plt.bar(x - width/2, importance_df['重要性'], width, label='模型预测重要性')
plt.bar(x + width/2, importance_df['真实重要性'], width, label='真实重要性')

plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('贝叶斯岭回归模型 - 预测重要性与真实重要性对比')
plt.xticks(x, importance_df['特征'], rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\预测重要性与真实重要性对比.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 模型学习曲线
plt.figure(figsize=(10, 6))
plt.plot(br.scores_)
plt.xlabel('迭代次数')
plt.ylabel('对数边际似然')
plt.title('贝叶斯岭回归模型 - 学习曲线')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\学习曲线.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 桑基图
# 准备桑基图数据
source = []
target = []
value = []
label = features + ['预测结果']

# 创建连接关系
for i, (feat, imp) in enumerate(zip(features, importance)):
    if imp > 0.1:  # 只显示重要性大于0.1的特征，避免图表过于复杂
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
    title_text="特征重要性桑基图",
    font_size=10,
    height=800
)

# 保存为HTML文件以保持交互性
pio.write_html(fig, r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\特征重要性桑基图.html')

# 11. 相关性热力图
# 计算特征之间的相关性
correlation_matrix = df[features].corr()

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

plt.title('特征相关性热力图', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r'c:\Users\JIMMYHE\PycharmProjects\统计模型\问卷数据\特征相关性热力图.png', dpi=300, bbox_inches='tight')
plt.show()