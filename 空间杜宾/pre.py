import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.preprocessing import MinMaxScaler
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
import mapclassify as mc


def load_provinces_data(geojson_path, data_path):
    """
    加载省份地理数据和指标数据并合并

    参数:
    geojson_path: 省份地理边界GeoJSON文件路径
    data_path: 省份指标数据CSV文件路径

    返回:
    provinces_gdf: 包含地理和指标数据的GeoDataFrame
    """
    # 加载省份地理数据
    provinces_gdf = gpd.read_file(geojson_path, encoding='utf-8')

    # 加载指标数据
    indicators_df = pd.read_csv(data_path, encoding='utf-8')

    # 合并数据
    provinces_gdf = provinces_gdf.merge(indicators_df, on='province_name')

    return provinces_gdf


def normalize_indicators(gdf, columns):
    """
    对指定列进行归一化处理

    参数:
    gdf: GeoDataFrame
    columns: 需要归一化的列名列表

    返回:
    normalized_gdf: 包含归一化列的GeoDataFrame
    """
    scaler = MinMaxScaler()
    normalized_gdf = gdf.copy()

    normalized_data = scaler.fit_transform(gdf[columns])

    for i, col in enumerate(columns):
        normalized_gdf[f'{col}_norm'] = normalized_data[:, i]

    return normalized_gdf


def calculate_composite_index(gdf, columns, weights=None):
    """
    计算综合指数

    参数:
    gdf: GeoDataFrame
    columns: 用于计算综合指数的列名列表
    weights: 各指标权重，默认为平均权重

    返回:
    index_gdf: 包含综合指数的GeoDataFrame
    """
    if weights is None:
        weights = [1 / len(columns)] * len(columns)

    index_gdf = gdf.copy()
    index_gdf['composite_index'] = 0

    for i, col in enumerate(columns):
        index_gdf['composite_index'] += index_gdf[col] * weights[i]

    return index_gdf


def plot_choropleth(gdf, column, title, cmap='YlGnBu', figsize=(12, 8), legend_title=None):
    """
    绘制分级统计地图

    参数:
    gdf: GeoDataFrame
    column: 用于绘图的列名
    title: 图表标题
    cmap: 颜色映射
    figsize: 图表尺寸
    legend_title: 图例标题
    """
    fig, ax = plt.subplots(1, figsize=figsize)

    # 使用分位数分类 - 修复 Quantiles 没有 lower 属性的问题
    try:
        # 方法1：使用字符串指定方案而不是对象
        gdf.plot(column=column,
                 ax=ax,
                 cmap=cmap,
                 scheme='Quantiles',
                 k=5,
                 legend=True,
                 edgecolor='white',
                 linewidth=0.5,
                 legend_kwds={'title': legend_title if legend_title else column})
    except Exception as e:
        print(f"使用scheme='Quantiles'方法出错: {e}")
        try:
            # 方法2：手动计算分位数并使用分类边界
            values = gdf[column].dropna().values
            bins = np.percentile(values, [0, 20, 40, 60, 80, 100])
            gdf.plot(column=column,
                     ax=ax,
                     cmap=cmap,
                     legend=True,
                     edgecolor='white',
                     linewidth=0.5,
                     legend_kwds={'title': legend_title if legend_title else column})
        except Exception as e2:
            print(f"手动计算分位数方法也失败: {e2}")
            # 方法3：最简单的方法，不使用分类
            gdf.plot(column=column,
                     ax=ax,
                     cmap=cmap,
                     legend=True,
                     edgecolor='white',
                     linewidth=0.5)

    ax.set_title(title, fontsize=15)
    ax.axis('off')

    plt.tight_layout()
    return fig, ax


def calculate_spatial_autocorrelation(gdf, column):
    """
    计算全局空间自相关 (Moran's I)

    参数:
    gdf: GeoDataFrame
    column: 待分析的列名

    返回:
    moran_i: Moran's I统计量
    p_value: p值
    """
    # 创建邻接矩阵
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'  # 行标准化

    # 计算全局Moran's I
    moran = Moran(gdf[column], w)

    return moran.I, moran.p_sim


def calculate_local_moran(gdf, column):
    """
    计算局部空间自相关 (Local Moran's I)

    参数:
    gdf: GeoDataFrame
    column: 待分析的列名

    返回:
    local_moran_gdf: 包含局部Moran's I结果的GeoDataFrame
    """
    # 创建邻接矩阵
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'  # 行标准化

    # 计算局部Moran's I
    local_moran = Moran_Local(gdf[column], w)

    # 添加结果到GeoDataFrame
    local_moran_gdf = gdf.copy()
    local_moran_gdf['local_moran_i'] = local_moran.Is
    local_moran_gdf['p_value'] = local_moran.p_sim
    local_moran_gdf['quadrant'] = local_moran.q

    return local_moran_gdf


def create_interactive_map(gdf, column, title, popup_columns=None):
    """
    创建交互式地图

    参数:
    gdf: GeoDataFrame
    column: 用于绘图的列名
    title: 图表标题
    popup_columns: 点击时显示的列名列表

    返回:
    m: Folium地图对象
    """
    # 确保数据不包含空值
    map_data = gdf.copy()
    if map_data[column].isnull().any():
        print(f"警告: '{column}'列中存在空值，使用均值填充")
        map_data[column] = map_data[column].fillna(map_data[column].mean())

    # 计算中心点
    center = [map_data.geometry.centroid.y.mean(), map_data.geometry.centroid.x.mean()]

    # 创建地图
    m = folium.Map(location=center, zoom_start=4, tiles='CartoDB positron')

    # 添加标题
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    # 创建颜色映射
    min_val = map_data[column].min()
    max_val = map_data[column].max()

    # 添加debug信息
    print(f"地图数据范围: {min_val} - {max_val}")
    print(f"数据样本: {map_data[['province_name', column]].head()}")

    # 添加choropleth图层
    choropleth = folium.Choropleth(
        geo_data=map_data.to_json(),
        name='choropleth',
        data=map_data,
        columns=['province_name', column],
        key_on='feature.properties.province_name',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=column
    ).add_to(m)

    # 添加悬停标签
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['province_name'], labels=False)
    )

    # 添加GeoJSON图层和弹出窗口
    style_function = lambda x: {'fillColor': '#ffffff',
                                'color': '#000000',
                                'fillOpacity': 0.1,
                                'weight': 0.5}

    highlight_function = lambda x: {'fillColor': '#000000',
                                    'color': '#000000',
                                    'fillOpacity': 0.25,
                                    'weight': 0.5}

    if popup_columns is None:
        popup_columns = ['province_name', column]

    # 确保popup_columns中的列都存在，且不含空值
    for col in popup_columns:
        if col not in map_data.columns:
            print(f"警告: 弹出窗口列'{col}'不存在，移除该列")
            popup_columns.remove(col)
        elif map_data[col].isnull().any() and col != 'province_name':
            print(f"警告: 弹出窗口列'{col}'存在空值，使用0填充")
            map_data[col] = map_data[col].fillna(0)

    # 如果列表为空，添加省份名称
    if not popup_columns:
        popup_columns = ['province_name']

    popup = folium.features.GeoJsonPopup(
        fields=popup_columns,
        aliases=popup_columns,
        localize=True,
        labels=True
    )

    tooltip = folium.features.GeoJsonTooltip(
        fields=['province_name'],
        aliases=['省份'],
        localize=True,
        sticky=False,
        labels=True
    )

    folium.features.GeoJson(
        map_data,
        style_function=style_function,
        highlight_function=highlight_function,
        popup=popup,
        tooltip=tooltip
    ).add_to(m)

    # 添加图层控制
    folium.LayerControl().add_to(m)

    return m