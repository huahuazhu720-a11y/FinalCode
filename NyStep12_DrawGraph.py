
"""
Generates PDFs verison of map&network picutre from the network graph. Can create undirected graphs, directed graphs, and road graphs from the saved network .pkl file.
"""


import geopandas as gpd
import json
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.colors as mcolors
import networkx as nx 
from matplotlib.patches import FancyArrowPatch

def run():
    #  这里控制画什么图，只能三选一
    un_directed_graph=False
    directed_Graph=False
    roude_graph=True
    roude_coler=True

    # 读取 shapefile
    districts = gpd.read_file(r"shapfile\merged_taxi_zones.shp").to_crs(epsg=4326)
    if os.path.exists("NewYorkCity.pickle"):
        with open("NewYorkCity.pickle", "rb") as f:
            graph = pickle.load(f)
    # 读取 nodes 和 edges 数据
    nodes = gpd.read_file("nodes_NewYorkCity.geojson")
    edges = gpd.read_file("edges_NewYorkCity.geojson")

    # 这里控制使用哪一年的网络图
    year=2022
    name=f"Graph_NewYork_Manhattan{year}"
    Net_work=pd.read_pickle(f"{name}.pkl")

    edges_data = Net_work.edges(data=True)
    max_congestion=0
    for u, v, data in edges_data:
        data['congestion']=int(data['Road_Capacity'])-int(data['driving_travelor']+data['taxi_travelor'])
        if data['congestion']>max_congestion:
            max_congestion=data['congestion']
    with open("Graph_NewYork_edge_traffic.json", "r") as f:
        edge_traffic_data = json.load(f)

    adjacent_districts = pd.read_pickle('adjacent_districts.pkl')
    # 创建 node KDTree 用于快速搜索
    node_coords = np.array([[geom.x, geom.y] for geom in nodes["geometry"]])
    tree = cKDTree(node_coords)

    # 绘图初始化
    fig, ax = plt.subplots(figsize=(15, 30))
    districts.plot(ax=ax, color="lightblue", edgecolor="black", alpha=0.5, linewidth=1, label="Taxi Zones",zorder=1 )
    # 下面的代码是用于突出显示曼哈顿地区的：

    specific_id=[114, 61, 119, 98, 59, 103, 90, 69]

    specific_districts = districts[districts['zone_id'].isin(specific_id)]
    # 3. 绘制符合条件的区域为红色
    specific_districts.plot(ax=ax, color="red", edgecolor="black", alpha=0.8, linewidth=1, label="ManHD CBD", zorder=2)
    # 上面的代码是用于突出显示曼哈顿地区的

    edges.plot(ax=ax, color="lightgray", linewidth=0.3, alpha=0.7, label="Road Network")

    # 计算区域中心点
    districts["center"] = districts.geometry.centroid
    # 遍历区域并查找最近的节点

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'traffic_cmap', ['green', 'red', 'black'], N=100
    )
    traffic_values = list(edge_traffic_data.values())  # 获取所有 traffic 的值
    norm = plt.Normalize(0, max_congestion)
    all_routes = []
    edge_colors = []
    for i, row in districts.iterrows():
        zone_id = int(row["zone_id"])
        try:
            if zone_id in specific_id:
                continue
        except Exception:
            pass
        node_coords = np.array([[geom.x, geom.y] for geom in nodes['geometry']])
        print(zone_id)  
        #----下面这些是从shpfile里面先计算中心点位置，再根据中心点位置确定路网node的位置，这里有生成的network图纸，所以不需要了·······
        # center = row["center"]
        # center_point = Point(center.x, center.y)    
        # tree = cKDTree(node_coords)   
        # nearest_distance, nearest_node_index = tree.query([center.x, center.y])  
        # nearest_node_coords = node_coords[nearest_node_index]
        #----上面这些不需要了·······  
        facecolor="green"    
        TotalPopulation=Net_work.nodes[zone_id].get("TotalPopulation", "属性不存在")    
        nearest_node_coords = node_coords[Net_work.nodes[zone_id].get("center_index", "属性不存在")]
        # 绘制圆（以最近的 `node` 作为圆心）
        radius = int(TotalPopulation)*0.001*0.00008
        min_radius = 0.0001  # 最小半径
        radius = max(radius, min_radius)
        circle = Point(nearest_node_coords).buffer(radius)  # 半径为 0.001 度
        gpd.GeoSeries([circle]).plot(ax=ax, facecolor=facecolor, linestyle="--",alpha=0.9,zorder=10)    
        ax.text(nearest_node_coords[0], nearest_node_coords[1], str(zone_id), fontsize=10, color='black', ha='center', va='center',zorder=20)
        records=set()
        # 遍历每个节点连接的点
        for i in adjacent_districts[zone_id]: 
            try:
                if zone_id in specific_id and i in specific_id: # 曼哈顿地区的不画
                    continue
            except Exception:
                pass
            flow=Net_work[zone_id][i].get("total_flow", False) # 跳过没有的
            if not flow: # 注意：：flow是0的，也会跳过
                continue

            try:
                edge_data = Net_work[zone_id][i].get("edge_indices", None)
                congestion = Net_work[zone_id][i].get("congestion", None)
                congestion = 0 if congestion is not None and congestion < 0 else congestion
            except Exception:
                print(f"{zone_id} is not connected with {i}")
                temp=(zone_id,i)
                records.add(temp)   
                continue 
            #~~~~下面的代码是用于只是画一条直线作为link的~~~~~~~~~~~~~~~~~~~~~~
            if un_directed_graph:
                temp=(zone_id,i)
                temp1=(i,zone_id)
                if (temp in records) or (temp1 in records):  
                    continue
                records.add(temp)    
            #~~~~上面的代码是用于只是画一条直线作为link的~~~~~~~~~~~~~~~~~~~~~~



            start_road_node_index = Net_work.nodes[zone_id]["center_index"]
            end_road_node_index = Net_work.nodes[i]["center_index"]
            start_coords = (nodes.loc[start_road_node_index, 'geometry'].x, nodes.loc[start_road_node_index, 'geometry'].y)
            end_coords = (nodes.loc[end_road_node_index, 'geometry'].x, nodes.loc[end_road_node_index, 'geometry'].y)

            # 创建双向箭头表示交通流方向
            if directed_Graph:
                arrow = FancyArrowPatch(
                    posA=start_coords, posB=end_coords,
                    arrowstyle="->", mutation_scale=10, color="brown", lw=1.2, zorder=10,
                    connectionstyle=f"arc3,rad=0.15"
                )
                ax.add_patch(arrow)
            if un_directed_graph:
                # 创建直线箭头表示交通流方向
                arrow = FancyArrowPatch(
                    posA=start_coords, posB=end_coords,
                    arrowstyle="-", mutation_scale=10, color="brown", lw=1.5, zorder=10,            
                )
                ax.add_patch(arrow)
                
            #================下面代码是把node的连接用路网绘制========================
            if roude_graph:
                for u, v, traffic_flow in edge_data:         
                    edge = edges[(edges['u'] == u) & (edges['v'] == v)]
                    if not edge.empty:
                        edge_geom = edge.geometry.values[0]
                        all_routes.append(edge_geom)
                        color = cmap(norm(congestion))  # 使用归一化后的值来选择颜色
                        edge_colors.append(color)
    if roude_graph and not roude_coler:
        gpd.GeoSeries(all_routes).plot(ax=ax, color='black', linewidth=0.8)
    if roude_graph and roude_coler:
        gpd.GeoSeries(all_routes).plot(ax=ax, color=edge_colors, linewidth=0.8)
    #================上面代码是把node的连接用路网绘制========================

    if directed_Graph:
        title="Directed Network Diagram of New York City"
        filename=f"Graph_Directed{name}.pdf"
    if un_directed_graph:
        title="Undirected Network Diagram of New York City"
        filename=f"Graph_Undirected{name}.pdf"
    if roude_graph and not roude_coler:
        title="Road Network Diagram of New York City"
        filename=f"Graph_Road{name}.pdf"

    if roude_graph and roude_coler:
        title="Road Network Diagram of New York City"
        filename=f"Graph_Road_colered{name}.pdf"

    plt.title(title)
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # {(30, 94), (32, 8), (30, 129), (37, 44), (41, 115), (21, 18), (41, 76)}
        