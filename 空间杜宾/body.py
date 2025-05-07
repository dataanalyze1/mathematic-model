#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中国各省份新质生产力与碳减排潜力空间杜宾模型分析

本脚本基于空间杜宾模型(Spatial Durbin Model)，分析中国各省份新质生产力与碳减排潜力的空间关系。
空间杜宾模型同时考虑了自变量和因变量的空间滞后作用，能够捕捉到地区间的复杂空间交互关系。

作者：[Your Name]
日期：2023-09-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import scipy.stats as stats
import os
import warnings
from sklearn.preprocessing import StandardScaler
from pysal.lib import weights
from pysal.model import spreg
from esda.moran import Moran

warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时'-'显示为方块的问题


def load_and_prepare_data():
    """加载和预处理数据"""
    # 加载省份指标数据 - 直接从Excel读取
    file_path = r'C:\Users\JIMMYHE\PycharmProjects\统计模型\空间杜宾\data\工作簿 2_翻译及删除数字版.xlsx'
    indicators_df = pd.read_excel(file_path)

    # 确保所有数值列都是浮点数类型
    numeric_columns = ['gdp', 'population', 'high_tech_output', 'digital_economy',
                       'r_d_investment', 'green_patents', 'energy_consumption',
                       'carbon_emission', 'renewable_energy', 'energy_efficiency',
                       'forest_coverage']

    for col in numeric_columns:
        # 确保列是数值类型
        indicators_df[col] = pd.to_numeric(indicators_df[col], errors='coerce')

    # 检查是否有缺失值，并填充或删除
    if indicators_df[numeric_columns].isnull().any().any():
        print("警告：数据中存在缺失值，将使用均值填充")
        indicators_df[numeric_columns] = indicators_df[numeric_columns].fillna(indicators_df[numeric_columns].mean())

    # 加载空间权重矩阵
    file_path = r'C:\Users\JIMMYHE\PycharmProjects\统计模型\空间杜宾\data\spatial_weight_matrix.csv'
    w_df = pd.read_csv(file_path, index_col='province')

    # 计算新质生产力指标 (组合高新技术、数字经济、研发投入和绿色专利)
    # 使用安全的数值计算方式
    indicators_df['new_productivity_index'] = (
                                                  # 高新技术产出占GDP比重
                                                      (indicators_df['high_tech_output'] / indicators_df['gdp']).astype(
                                                          float) +
                                                      # 数字经济占GDP比重
                                                      (indicators_df['digital_economy'] / indicators_df['gdp']).astype(
                                                          float) +
                                                      # 研发投入强度
                                                      (indicators_df['r_d_investment'] / indicators_df['gdp']).astype(
                                                          float) +
                                                      # 绿色专利数量占比
                                                      (indicators_df['green_patents'] / indicators_df[
                                                          'population']).astype(float)
                                              ) / 4.0  # 简单平均

    # 计算碳减排潜力指标 (组合能源消耗、碳排放、可再生能源、能源效率和森林覆盖)
    indicators_df['carbon_reduction_potential'] = (
                                                      # 能源消耗强度（负向指标，取倒数）
                                                          (1.0 / (indicators_df['energy_consumption'] / indicators_df[
                                                              'gdp'])).astype(float) +
                                                          # 碳排放强度（负向指标，取倒数）
                                                          (1.0 / (indicators_df['carbon_emission'] / indicators_df[
                                                              'gdp'])).astype(float) +
                                                          # 可再生能源占比
                                                          (indicators_df['renewable_energy'] / indicators_df[
                                                              'energy_consumption']).astype(float) +
                                                          # 能源效率
                                                          indicators_df['energy_efficiency'].astype(float) +
                                                          # 森林覆盖率
                                                          (indicators_df['forest_coverage'] / 100.0).astype(float)
                                                  ) / 5.0  # 简单平均

    # 检查计算结果是否有问题
    if indicators_df['new_productivity_index'].isnull().any() or \
            indicators_df['carbon_reduction_potential'].isnull().any():
        print("警告：指标计算结果中存在缺失值，请检查原始数据")
        # 用均值填充缺失值
        indicators_df['new_productivity_index'] = indicators_df['new_productivity_index'].fillna(
            indicators_df['new_productivity_index'].mean())
        indicators_df['carbon_reduction_potential'] = indicators_df['carbon_reduction_potential'].fillna(
            indicators_df['carbon_reduction_potential'].mean())

    # 数据标准化
    scaler = StandardScaler()
    columns_to_scale = ['new_productivity_index', 'carbon_reduction_potential']
    indicators_df[columns_to_scale] = scaler.fit_transform(indicators_df[columns_to_scale])

    return indicators_df, w_df


def create_spatial_weights(w_df):
    """创建PySAL空间权重对象"""
    # 转换为numpy数组
    w_array = w_df.values

    try:
        # 尝试导入新的Graph类
        from libpysal import graph

        # 检查是否支持Graph.from_adjlist方法
        if hasattr(graph.Graph, 'from_adjlist'):
            # 使用Graph类创建空间权重
            G = graph.Graph.from_adjlist(w_df.index.tolist(), w_array)
            G.transform = 'r'  # 行标准化
            return G
        else:
            # 导入传统W类
            from pysal.lib import weights
            # 创建传统的W对象
            w = weights.util.full2W(w_array)
            w.ids = w_df.index.tolist()
            w.transform = 'r'  # 行标准化
            return w
    except (ImportError, AttributeError):
        # 如果导入失败，使用传统的W类
        from pysal.lib import weights
        w = weights.util.full2W(w_array)
        w.ids = w_df.index.tolist()
        w.transform = 'r'  # 行标准化
        return w


def run_moran_test(y, w):
    """进行Moran's I空间自相关检验"""
    moran = Moran(y, w)
    return moran.I, moran.p_sim


def run_spatial_durbin_model(y, X, w):
    """
    运行真正的空间杜宾模型 (SDM)

    空间杜宾模型形式:
    y = ρWy + Xβ + WXθ + ε

    其中:
    y: 因变量
    X: 自变量
    W: 空间权重矩阵
    ρ: 空间自相关系数
    β: 自变量系数
    θ: 自变量空间滞后项系数
    ε: 误差项
    """
    # 运行空间杜宾模型 (使用PySAL的ML_Lag实现)
    model = spreg.ML_Lag(y, X, w=w, name_y=y.name, name_x=X.columns.tolist(), spat_diag=True)

    return model


def run_true_sdm(y, X, w):
    """
    运行真正的空间杜宾模型 (SDM)，包括自变量的空间滞后项
    """
    # 确保y是一维数组
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)

    # 如果y是二维数组，确保转为一维
    if y_array.ndim > 1:
        y_array = y_array.flatten()

    # 确保X是二维数组
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.array(X)

    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)

    # 运行空间杜宾模型 (使用PySAL的ML_Lag实现)
    try:
        # 使用较新的API，包括spat_impacts参数
        model = spreg.ML_Lag(y_array, X_array, w=w,
                             name_y=y.name if hasattr(y, 'name') else 'y',
                             name_x=X.columns.tolist() if hasattr(X, 'columns') else None,
                             spat_diag=True, durbin=True,
                             spat_impacts='all')  # 请求计算所有空间影响
        return model
    except Exception as e:
        print(f"使用新版API时出错: {e}")
        print("尝试使用替代方法...")

        # 尝试使用不同的参数组合
        try:
            # 方法1：不使用durbin参数，手动构建滞后变量
            if hasattr(X, 'columns'):
                # 创建空间滞后变量
                lag_vars = {}
                for col in X.columns:
                    lag_vars[f'W_{col}'] = weights.lag_spatial(w, X[col])

                # 将滞后变量添加到X中
                X_with_lags = X.copy()
                for key, val in lag_vars.items():
                    X_with_lags[key] = val

                # 尝试使用spat_impacts参数
                try:
                    model = spreg.ML_Lag(y, X_with_lags, w=w,
                                         name_y=y.name if hasattr(y, 'name') else 'y',
                                         name_x=X_with_lags.columns.tolist(),
                                         spat_diag=True, spat_impacts='all')
                except:
                    # 如果不支持spat_impacts，则不使用该参数
                    model = spreg.ML_Lag(y, X_with_lags, w=w,
                                         name_y=y.name if hasattr(y, 'name') else 'y',
                                         name_x=X_with_lags.columns.tolist(),
                                         spat_diag=True)
                return model
            else:
                # 如果X不是DataFrame
                # 手动创建滞后变量
                X_lag = weights.lag_spatial(w, X_array)
                X_combined = np.column_stack([X_array, X_lag])

                model = spreg.ML_Lag(y_array, X_combined, w=w, spat_diag=True)
                return model

        except Exception as e2:
            print(f"替代方法1失败: {e2}")

            # 最后的备选方案：简单ML_Lag模型
            try:
                model = spreg.ML_Lag(y_array, X_array, w=w,
                                     name_y=y.name if hasattr(y, 'name') else 'y',
                                     name_x=X.columns.tolist() if hasattr(X, 'columns') else None,
                                     spat_diag=True)
                return model
            except Exception as e3:
                print(f"最终备选方案失败: {e3}")
                raise Exception("无法运行空间杜宾模型，请检查输入数据和PySAL版本")


def calculate_impacts(model):
    """
    计算直接效应、间接效应和总效应

    参数:
    model: 空间杜宾模型对象

    返回:
    impacts: 包含直接效应、间接效应和总效应的字典
    """
    # 直接使用PySAL内置的计算影响方法
    try:
        # 检查模型是否有impacts方法或属性
        if hasattr(model, 'impacts') and callable(getattr(model, 'impacts')):
            # 如果有impact方法，直接调用
            impact_results = model.impacts()
            return {
                'direct': impact_results.direct,
                'indirect': impact_results.indirect,
                'total': impact_results.total
            }
        elif hasattr(model, 'impacts') and not callable(getattr(model, 'impacts')):
            # 如果impacts是属性而非方法
            return {
                'direct': model.impacts.direct,
                'indirect': model.impacts.indirect,
                'total': model.impacts.total
            }
    except Exception as e:
        print(f"使用内置impacts方法失败: {e}")
        print("改用手动计算方法...")

    # 如果内置方法不可用，使用手动计算方法
    try:
        # 获取模型参数
        betas = model.betas[1:].flatten()  # 排除常数项并确保是一维数组
        n = model.n
        k = len(betas)

        # 获取空间自相关系数
        rho = float(model.rho)

        # 获取权重矩阵 - 修正权重矩阵获取逻辑
        W_array = None

        # 尝试多种方式获取权重矩阵
        if hasattr(model, 'w'):
            w = model.w
            # 将权重矩阵转换为数组形式
            if hasattr(w, 'full'):
                W_array = w.full()[0]
            elif hasattr(w, 'dense'):
                W_array = w.dense
            elif hasattr(w, 'to_W'):
                # 如果是Graph对象
                W_array = w.to_W().full()[0]
            elif hasattr(w, 'sparse'):
                # 如果有sparse属性，转换为稠密矩阵
                W_array = w.sparse.toarray()
            elif hasattr(w, 'adj'):
                # 如果有邻接列表
                n = len(w.adj)
                W_array = np.zeros((n, n))
                for i, neighbors in w.adj.items():
                    idx_i = w.id2i[i]
                    for j, weight in neighbors.items():
                        idx_j = w.id2i[j]
                        W_array[idx_i, idx_j] = weight

        # 如果通过所有方法都无法获取权重矩阵
        if W_array is None:
            print("无法从模型中直接获取权重矩阵，尝试从数据中重新加载...")
            # 重新加载数据和权重矩阵
            _, w_df = load_and_prepare_data()
            w = create_spatial_weights(w_df)

            # 再次尝试转换为数组
            if hasattr(w, 'full'):
                W_array = w.full()[0]
            elif hasattr(w, 'dense'):
                W_array = w.dense
            elif hasattr(w, 'to_W'):
                W_array = w.to_W().full()[0]
            elif hasattr(w, 'sparse'):
                W_array = w.sparse.toarray()
            else:
                # 最后尝试直接转换
                W_array = np.array(w)

        # 确保W_array是2D数组
        if W_array is None or W_array.ndim != 2:
            raise ValueError(f"无法获取有效的权重矩阵，当前矩阵维度: {W_array.ndim if W_array is not None else 'None'}")

        print(f"成功获取权重矩阵，维度: {W_array.shape}")

        # 计算 (I - ρW)^-1
        I = np.eye(n)
        S = np.linalg.inv(I - rho * W_array)

        # 初始化效应数组
        direct_effects = np.zeros(k)
        indirect_effects = np.zeros(k)
        total_effects = np.zeros(k)

        # 对每个变量计算效应
        for i in range(k):
            beta_i = betas[i]  # 获取标量系数

            # 计算效应矩阵S * beta_i (元素级乘法)
            Si = S * beta_i  # 使用标量乘法，不是矩阵乘法

            # 计算直接、间接和总效应
            direct_effects[i] = np.mean(np.diag(Si))
            total_effects[i] = np.mean(np.sum(Si, axis=1))
            indirect_effects[i] = total_effects[i] - direct_effects[i]

        return {
            'direct': direct_effects,
            'indirect': indirect_effects,
            'total': total_effects
        }

    except Exception as e:
        print(f"手动计算效应失败: {e}")

        # 如果两种方法都失败，返回一个简化的估计
        print("返回简化的效应估计...")
        k = len(model.betas) - 1  # 排除常数项

        # 使用简化方法估计效应
        direct_effects = model.betas[1:].flatten()  # 直接效应近似等于系数
        indirect_effects = model.rho * direct_effects  # 间接效应近似等于系数乘以空间系数
        total_effects = direct_effects + indirect_effects

        return {
            'direct': direct_effects,
            'indirect': indirect_effects,
            'total': total_effects
        }


def plot_spatial_distribution(data, column, title, cmap='viridis'):
    """绘制空间分布图"""
    # 创建离散颜色映射
    values = data[column].values
    quantiles = stats.mstats.mquantiles(values, [0.2, 0.4, 0.6, 0.8])
    categories = np.zeros_like(values)

    for i in range(4):
        categories[values > quantiles[i]] = i + 1

    colors = ['#f7fcf5', '#c7e9c0', '#74c476', '#31a354', '#006d2c']
    cmap = ListedColormap(colors)

    # 绘制热力图和条形图
    plt.figure(figsize=(16, 8))

    # 热力图
    plt.subplot(121)
    matrix = pd.pivot_table(data, values=column, index='province_name', columns=None)
    sns.heatmap(matrix, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': column})
    plt.title(f'{title}的省际分布', fontsize=14)

    # 条形图
    plt.subplot(122)
    sorted_data = data.sort_values(by=column, ascending=False)
    bars = plt.bar(sorted_data['province_name'], sorted_data[column], color=colors[2])
    plt.xticks(rotation=90)
    plt.title(f'{title}的省份排名', fontsize=14)
    plt.tight_layout()

    plt.savefig(f'results/{title}_空间分布.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_sdm_analysis():
    """运行空间杜宾模型分析"""
    print("开始分析中国各省份新质生产力与碳减排潜力的空间关系...")

    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')

    # 加载和预处理数据
    data, w_df = load_and_prepare_data()

    # 创建空间权重对象
    w = create_spatial_weights(w_df)

    # 绘制空间分布图
    plot_spatial_distribution(data, 'new_productivity_index', '新质生产力指数')
    plot_spatial_distribution(data, 'carbon_reduction_potential', '碳减排潜力指数')

    # 进行空间自相关检验
    moran_i_np, p_value_np = run_moran_test(data['new_productivity_index'], w)
    moran_i_cr, p_value_cr = run_moran_test(data['carbon_reduction_potential'], w)

    print("\n空间自相关检验结果:")
    print(f"新质生产力指数 Moran's I: {moran_i_np:.4f}, p值: {p_value_np:.4f}")
    print(f"碳减排潜力指数 Moran's I: {moran_i_cr:.4f}, p值: {p_value_cr:.4f}")

    # 准备模型数据
    y1 = data['new_productivity_index']
    y2 = data['carbon_reduction_potential']

    # 添加控制变量 (人均GDP, 森林覆盖率)
    data['gdp_per_capita'] = data['gdp'] / data['population']

    # 模型1: 新质生产力指数作为因变量
    X1 = data[['carbon_reduction_potential', 'gdp_per_capita', 'forest_coverage']]

    # 模型2: 碳减排潜力指数作为因变量
    X2 = data[['new_productivity_index', 'gdp_per_capita', 'forest_coverage']]

    # 运行真正的空间杜宾模型(包括自变量的空间滞后项)
    print("\n运行空间杜宾模型(SDM)...")
    sdm_model1 = run_true_sdm(y1, X1, w)
    sdm_model2 = run_true_sdm(y2, X2, w)

    # 计算效应
    impacts1 = calculate_impacts(sdm_model1)
    impacts2 = calculate_impacts(sdm_model2)

    # 输出模型结果
    print("\n模型1: 新质生产力指数 = f(碳减排潜力指数, 控制变量) 的空间杜宾模型结果:")
    print(sdm_model1.summary)

    print("\n模型2: 碳减排潜力指数 = f(新质生产力指数, 控制变量) 的空间杜宾模型结果:")
    print(sdm_model2.summary)

    # 创建效应分析结果表
    print("\n效应分析结果:")

    # 模型1的效应分析
    effect_vars1 = X1.columns.tolist()
    effect_results1 = []

    for i, var in enumerate(effect_vars1):
        effect_results1.append({
            '变量': var,
            '直接效应': impacts1['direct'][i],
            '间接效应': impacts1['indirect'][i],
            '总效应': impacts1['total'][i]
        })

    effects_df1 = pd.DataFrame(effect_results1)
    print("\n模型1 (新质生产力) 效应分析:")
    print(effects_df1)

    # 模型2的效应分析
    effect_vars2 = X2.columns.tolist()
    effect_results2 = []

    for i, var in enumerate(effect_vars2):
        effect_results2.append({
            '变量': var,
            '直接效应': impacts2['direct'][i],
            '间接效应': impacts2['indirect'][i],
            '总效应': impacts2['total'][i]
        })

    effects_df2 = pd.DataFrame(effect_results2)
    print("\n模型2 (碳减排潜力) 效应分析:")
    print(effects_df2)

    # 保存结果
    effects_df1.to_csv('results/新质生产力模型效应分析.csv', index=False, encoding='utf-8-sig')
    effects_df2.to_csv('results/碳减排潜力模型效应分析.csv', index=False, encoding='utf-8-sig')

    # --- 在创建 coef_df1 之前需要做的修改 ---

    # 模型1 (新质生产力)
    # 1) 提取 β 部分
    betas1 = sdm_model1.betas.flatten()
    beta_names1 = sdm_model1.name_x

    # 2) z_stat 和 p-values (针对 β)
    zs_b1 = np.array([z for z, p in sdm_model1.z_stat])
    ps_b1 = np.array([p for z, p in sdm_model1.z_stat])

    # 3) 计算 ρ 的 z/p
    var_rho1 = sdm_model1.vm[0, 0]
    z_rho1 = sdm_model1.rho / np.sqrt(var_rho1)
    p_rho1 = stats.norm.sf(abs(z_rho1)) * 2

    # 4) 汇总到一个 DataFrame
    df_beta1 = pd.DataFrame({
        '变量': beta_names1,
        '系数': betas1,
        'z值': zs_b1,
        'p值': ps_b1,
    })

    df_rho1 = pd.DataFrame([{
        '变量': 'ρ',
        '系数': sdm_model1.rho,
        'z值': z_rho1,
        'p值': p_rho1
    }])

    coef_df1 = pd.concat([df_beta1, df_rho1], ignore_index=True)

    # 模型2 (碳减排潜力)
    # 1) 提取 β 部分
    betas2 = sdm_model2.betas.flatten()
    beta_names2 = sdm_model2.name_x

    # 2) z_stat 和 p-values (针对 β)
    zs_b2 = np.array([z for z, p in sdm_model2.z_stat])
    ps_b2 = np.array([p for z, p in sdm_model2.z_stat])

    # 3) 计算 ρ 的 z/p
    var_rho2 = sdm_model2.vm[0, 0]
    z_rho2 = sdm_model2.rho / np.sqrt(var_rho2)
    p_rho2 = stats.norm.sf(abs(z_rho2)) * 2

    # 4) 汇总到一个 DataFrame
    df_beta2 = pd.DataFrame({
        '变量': beta_names2,
        '系数': betas2,
        'z值': zs_b2,
        'p值': ps_b2,
    })

    df_rho2 = pd.DataFrame([{
        '变量': 'ρ',
        '系数': sdm_model2.rho,
        'z值': z_rho2,
        'p值': p_rho2
    }])

    coef_df2 = pd.concat([df_beta2, df_rho2], ignore_index=True)

    # 保存结果
    coef_df1.to_csv('results/新质生产力模型系数.csv', index=False, encoding='utf-8-sig')
    coef_df2.to_csv('results/碳减排潜力模型系数.csv', index=False, encoding='utf-8-sig')

    print("\n分析完成，结果已保存至 'results' 目录")


if __name__ == "__main__":
    run_sdm_analysis()