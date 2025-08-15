import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import osmnx as ox
import os
import pickle
import networkx as nx 
from database import Database
from scipy.spatial import cKDTree
import numpy as np
import pickle
import json
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from itertools import islice
import random
import requests
API_KEY = "AIzaSyBelwVdD3IN1nPYvpcw_19SyFfWBKkOPm8"
def get_free_flow_time(nodes, start_node, end_node, mode):
    try:
        # 获取起始点和终点的经纬度
        origin_lat = nodes.loc[start_node, 'y']
        origin_lon = nodes.loc[start_node, 'x']
        destination_lat = nodes.loc[end_node, 'y']
        destination_lon = nodes.loc[end_node, 'x'] 
        # 构建URL
        url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin_lat},{origin_lon}&destinations={destination_lat},{destination_lon}&mode={mode}&key={API_KEY}"
        
        # 发起请求
        response = requests.get(url)
        data = response.json()

        # 获取通勤时间（分钟）
        free_flow_time = data["rows"][0]["elements"][0]["duration"]["value"] / 60  # 单位：分钟
        return free_flow_time

    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        print(f"request failed: {e}")
        return False
# 读取并转换区域的 CRS
districts = gpd.read_file(r"shapfile\ManhattanCBD.shp")
districts = districts.to_crs(epsg=4326)
CBD_zone_ids = [114, 61, 119, 98, 59, 103, 90, 69]
# selected_ids = [
#    1,2,3,4
#     ]
# districts = districts[districts["zone_id"].isin(selected_ids)]

# print(districts.columns)
# 使用 place_name 获取 New York City 的路网数据
place_name = "New York City, New York, USA"

# 检查是否已有存储的 graph 文件
if os.path.exists("NewYorkCity.pickle"):
    with open("NewYorkCity.pickle", "rb") as f:
        graph = pickle.load(f)
else:
    graph = ox.graph_from_place(place_name, network_type="drive")
    with open("NewYorkCity.pickle", "wb") as f:
        pickle.dump(graph, f)

# 检查是否已有存储的 edges 文件
if os.path.exists("edges_NewYorkCity.geojson"):
    edges = gpd.read_file("edges_NewYorkCity.geojson")
else:
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges.to_file("edges_NewYorkCity.geojson", driver="GeoJSON")

# 检查是否已有存储的 nodes 文件
if os.path.exists("nodes_NewYorkCity.geojson"):
    nodes = gpd.read_file("nodes_NewYorkCity.geojson")
else:
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    nodes.to_file("nodes_NewYorkCity.geojson", driver="GeoJSON")
# graph=nx.DiGraph(graph)


# New York County	36061
#  Kings County	36047
# 	Queens County	36081
# 	Bronx County	36005
# 	Richmond County	36085
#  011402  ct 114.02

# 计算区域的中心点
districts.loc[:, "center"] = districts.geometry.centroid
fig, ax = plt.subplots(figsize=(15, 30))
# 绘制出租车区域
districts.plot(ax=ax, color="lightblue", edgecolor="black", alpha=0.8, linewidth=1, label="Taxi Zones")
# 绘制中心点
# districts["center"].plot(ax=ax, color="red", markersize=10, label="Centroids")
# 绘制道路网络
edges.plot(ax=ax, color="lightgray", linewidth=0.1, alpha=0.1, label="Road Network")

# 创建网络图new  graph
Graph_MhdCBD= nx.DiGraph()


# 连接到数据库
DB=Database("GreYelHir.db")
# 使用2022 年的数据来画图
year="2022"
query = f"""
            SELECT 
                *               
            FROM 
                ZoneSensusData_NewYork
            WHERE 
                year={year}           
                
        """
all_sensus_data = DB.sql_to_df(query)
all_sensus_data["zone_id"]=all_sensus_data["zone_id"].astype(float).astype(int)
all_sensus_data["S0801_C01_002E"]=all_sensus_data["S0801_C01_002E"].astype(float) #driving rate
all_sensus_data["S0801_C01_009E"]=all_sensus_data["S0801_C01_009E"].astype(float) #transit rate
all_sensus_data["S0801_C01_010E"]=all_sensus_data["S0801_C01_010E"].astype(float) #walk rate
all_sensus_data["S0801_C01_011E"]=all_sensus_data["S0801_C01_011E"].astype(float) #bike rate
all_sensus_data["private_car"]=all_sensus_data["S0801_C01_003E"]+all_sensus_data['S0801_C01_005E']/2+all_sensus_data['S0801_C01_006E']/3+all_sensus_data['S0801_C01_007E']/4

#计算在通勤人口中的driving_rate
all_sensus_data['driving_rate']=all_sensus_data['private_car'] / (all_sensus_data['private_car'] + all_sensus_data['S0801_C01_009E'] + all_sensus_data['S0801_C01_012E'])
# 计算在通勤人口中公共交通的rate
all_sensus_data['transit_rate']=all_sensus_data['S0801_C01_009E'] / (all_sensus_data['private_car'] + all_sensus_data['S0801_C01_009E'] + all_sensus_data['S0801_C01_012E'])
# 计算在通勤人口中taxi 的rate
all_sensus_data['taxi_rate']=all_sensus_data['S0801_C01_012E'] / (all_sensus_data['private_car'] + all_sensus_data['S0801_C01_009E'] + all_sensus_data['S0801_C01_012E'])


query = f"""
            SELECT 
                *               
            FROM 
                ZoneOutNodes
            WHERE 
                year={year}           
                
        """
Out_workface_all = DB.sql_to_df(query)
Out_workface_all['OutZone_id']=Out_workface_all['OutZone_id'].fillna(0).astype(int)
Out_workface = Out_workface_all[
    (Out_workface_all['zone_id'] != Out_workface_all['OutZone_id']) &
    (Out_workface_all['OutZone_id'].isin(CBD_zone_ids))
]# 将 OutZone_id 全部修改为 999


# 按 zone_id 分组，并对 count_of_jobs 求和
Out_workface = Out_workface.groupby("zone_id", as_index=False)["count_of_jobs"].sum()


Out_workface.loc[:, "OutZone_id"] = 999

# 避免 NaN 转换错误
Out_workface["zone_id"] = Out_workface["zone_id"].fillna(0).astype(int)

print(Out_workface.head())
query = f"""
            SELECT 
                *
            FROM 
                ZoneInNodes                 
            WHERE 
                year={year} 
           
        """
In_workface_all = DB.sql_to_df(query)
In_workface=In_workface_all[In_workface_all['zone_id']!=In_workface_all['InZone_id']]
# In_workface=In_workface.groupby('zone_id')['count_of_jobs'].sum().reset_index()
In_workface["InZone_id"]=In_workface["InZone_id"].astype(float).astype(int)
In_workface["zone_id"]=In_workface["zone_id"].astype(int)
# print(In_workface.head())

# 找到每个区域内最近的 `node` 并绘制圆

for index,row in districts.iterrows():    
    #数据库查询该节点属性  
    zone_id=int(row["zone_id"])
   
    if zone_id ==999:
        TotalPopulation = all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'S0801_C01_001E'].sum() if not all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'S0801_C01_001E'].empty else 0
        drive_rate = all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'driving_rate'].mean() if not all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'driving_rate'].empty else 0
        transit_rate = all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'transit_rate'].mean() if not all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'transit_rate'].empty else 0
        taxi_rate = all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'taxi_rate'].mean() if not all_sensus_data.loc[all_sensus_data['zone_id'].isin(CBD_zone_ids), 'taxi_rate'].empty else 0
        in_degree = In_workface.loc[In_workface['zone_id'].isin(CBD_zone_ids), 'count_of_jobs'].sum() if not In_workface.loc[In_workface['zone_id'].isin(CBD_zone_ids), 'count_of_jobs'].empty else 0
        out_degree = Out_workface.loc[Out_workface['zone_id'].isin(CBD_zone_ids), 'count_of_jobs'].sum() if not Out_workface.loc[Out_workface['zone_id'].isin(CBD_zone_ids), 'count_of_jobs'].empty else 0
    else:
        TotalPopulation = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == zone_id, 'S0801_C01_001E']), 0)
        try:
            drive_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == zone_id, 'driving_rate']), 0)
            transit_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == zone_id, 'transit_rate']), 0)
            taxi_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == zone_id, 'taxi_rate']), 0)
        except Exception:
            drive_rate=0
            transit_rate=0
            taxi_rate=0
        out_degree = Out_workface.loc[Out_workface['zone_id'] == zone_id, 'count_of_jobs'].sum() if not Out_workface.loc[Out_workface['zone_id'] == zone_id, 'count_of_jobs'].empty else 0
        in_degree = In_workface.loc[In_workface['zone_id'] == zone_id, 'count_of_jobs'].sum() if not In_workface.loc[In_workface['zone_id'] == zone_id, 'count_of_jobs'].empty else 0
    
    if out_degree < in_degree:
        nodetype="w"
        facecolor="blue"
    else:
        nodetype="R"
        facecolor="green"
    # 找到离中心点最近的道路node
    region=row.geometry
    nodes_in_districts = nodes[nodes.geometry.within(region)]
    center = row["center"]    
    center_point = Point(center.x, center.y)
    node_coords = np.array([[geom.x, geom.y] for geom in nodes['geometry']])
    tree = cKDTree(node_coords)
    # 查找 nearest node
    nearest_distance, nearest_node_index = tree.query([center.x, center.y])
    nearest_node_coords = node_coords[nearest_node_index]
    # 绘制圆（以最近的 `node` 作为圆心）
    radius = TotalPopulation*0.001*0.00008
    min_radius = 0.0001  # 最小半径
    radius = max(radius, min_radius)
    circle = Point(nearest_node_coords).buffer(radius)  # 半径为 0.001 度
    gpd.GeoSeries([circle]).plot(ax=ax, facecolor=facecolor, linestyle="--",zorder=10)    
    ax.text(nearest_node_coords[0], nearest_node_coords[1], str(zone_id), fontsize=10, color='red', ha='center', va='center',zorder=20)
    # 绘制最近的 `node`（圆心）
    # ax.plot(nearest_node_coords[0], nearest_node_coords[1], 'ro')  # 绘制节点点位

    print(f"""绘制node: {zone_id}""")
   
    #这里直接使用tract_id作为节点的名称
    Graph_MhdCBD.add_node(zone_id,
                         In_tarveler=in_degree,#从其他区到本区工作的总人口
                         Out_traveler=out_degree,#从本区出去工作的总人口
                         TotalPopulation=TotalPopulation,
                         drive_rate=drive_rate,
                         transit_rate=transit_rate,
                         taxi_rate=taxi_rate,
                         center_index=nearest_node_index, # node 节点的index，用这个再nodes里面访问node数据
                         Road_node=nodes.loc[nearest_node_index,'osmid'],
                         Type=nodetype)


# 统一查询数据库，避免在遍历中查询，提高性能
#定义统一绘制路径和方向的数组
all_routes = []
arrow_positions = []
nodes_not_in_OD_data=[]
all_edge_U_V_Taffic_dict={}
# 给每个node加加边，并画图
for node, graph_node_data in Graph_MhdCBD.nodes(data=True):
    print(f"""绘制 {node} 的edges""")
    if node==999:
        continue
    try:
        #检查node是否在inconnect里面有数据，如果没有，代表这个node里一个工作岗位都没有。再检查是否在outconnection    
        if node in Out_workface['zone_id'].values:
            connections= Out_workface.loc[Out_workface['zone_id'] == node]
            connection_list = list(zip(connections["OutZone_id"], connections["count_of_jobs"]))
            from_in_connection=False
            from_out_connection=True       
        else:
            nodes_not_in_OD_data.append(f"""this {node} has no edges""")
            continue
        # 开始连接节点
        print(len(connections))
        shortest_path_cache = {}

        for X_zone_id, count_of_jobs in connection_list:
            if from_out_connection:
                start_graph_node = node
                end_graph_node = int(X_zone_id)

            elif from_in_connection:
                print("from in connection~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                start_graph_node = int(X_zone_id)
                end_graph_node = node
            else:
                continue

            # 过滤无效情况
            if end_graph_node in {-1, start_graph_node} or start_graph_node in {-1,end_graph_node} or Graph_MhdCBD.has_edge(start_graph_node, end_graph_node):
                continue
            #获取这start node和end node的通勤总数
            total_flow = int(count_of_jobs)
            #获取交通模式的比例
            try:
                drive_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(start_graph_node), 'driving_rate']), 0)
                transit_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(start_graph_node), 'transit_rate']), 0)
                taxi_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(start_graph_node), 'taxi_rate']), 0)
            except Exception:
                drive_rate=0
                transit_rate=0
                taxi_rate=0
            driving_travelor=0
            transit_travelor=0
            taxi_travelor=0
            if total_flow>=3: 
                #计算不同的交通模式
                driving_travelor=int(float(total_flow)*float(drive_rate))#car 的数量
                transit_travelor=int(float(total_flow)*float(transit_rate))#乘公共交通的人口 的数量
                taxi_travelor=int(float(total_flow)*float(taxi_rate))#乘taxi的人口 的数量
                #调整这些交通模式
                values = [driving_travelor, transit_travelor, taxi_travelor]
                total_sum = sum(values)
                if total_sum > total_flow:
                    tep=total_sum-total_flow
                    max_value = max(values)
                    max_index = values.index(max_value)
                    values[max_index] -= tep
                    driving_travelor, transit_travelor, taxi_travelor = values
                elif total_sum < total_flow:
                    tep=total_flow-total_sum
                    min_value = min(values)
                    min_index = values.index(min_value)
                    values[min_index] += tep
                    driving_travelor, transit_travelor, taxi_travelor = values
            else:
                rates = [drive_rate, transit_rate, taxi_rate]
                modes = ['driving', 'transit', 'taxi']
                sorted_modes = [mode for _, mode in sorted(zip(rates, modes), reverse=True)]
                # 根据rates从大到小赋值
                for mode in sorted_modes[:int(total_flow)]:
                    if mode == 'driving':
                        driving_travelor = 1
                    elif mode == 'transit':
                        transit_travelor = 1
                    elif mode == 'taxi':
                        taxi_travelor = 1

            # 获取 road_node
            start_road_node = Graph_MhdCBD.nodes[start_graph_node]["Road_node"]
            start_road_node_index = Graph_MhdCBD.nodes[start_graph_node]["center_index"]
            end_road_node = Graph_MhdCBD.nodes[end_graph_node]["Road_node"]
            end_road_node_index = Graph_MhdCBD.nodes[end_graph_node]["center_index"]

            path_key = (start_road_node, end_road_node)

            # 先检查缓存
            if path_key in shortest_path_cache:
                route, route_length = shortest_path_cache[path_key]
            else:
                max_attempts = 5
                attempt = 0
                found_path = False

                while attempt < max_attempts:
                    try:
                        route = nx.shortest_path(graph, source=start_road_node, target=end_road_node, weight='length')
                        #获取路径长度，单位：米
                        route_length = nx.shortest_path_length(graph, source=start_road_node, target=end_road_node, weight='length')
                        shortest_path_cache[path_key] = (route, route_length)
                        found_path = True
                        break
                    except nx.NetworkXNoPath:
                        print(f"Attempt {attempt + 1}: No route between node {start_road_node} and node {end_road_node}, total_flow: {total_flow}")

                        # 尝试新的邻居
                        start_neighbors = list(graph.neighbors(start_road_node))
                        end_neighbors = list(graph.neighbors(end_road_node))

                        if start_neighbors:
                            start_road_node = min(start_neighbors, key=lambda n: graph.nodes[n].get('traffic_weight', 1))
                        if end_neighbors:
                            end_road_node = min(end_neighbors, key=lambda n: graph.nodes[n].get('traffic_weight', 1))
                        attempt += 1

                if not found_path:
                    print(f"Cannot find route from {start_road_node} to {end_road_node}, total_flow is {total_flow}.")
                    continue

            # 计算流量
            path_edges = [(route[i], route[i + 1], total_flow) for i in range(len(route) - 1)]
            for u, v, traffic in path_edges:
                all_edge_U_V_Taffic_dict[(u, v)] = all_edge_U_V_Taffic_dict.get((u, v), 0) + total_flow

            # 计算 free time
            
            freetime_dirve= get_free_flow_time(nodes,start_road_node_index,end_road_node_index,'driving')      
            if not freetime_dirve:
                freetime_dirve=(route_length/14.4)/60
            freetime_transit= get_free_flow_time(nodes,start_road_node_index,end_road_node_index,'transit')       
            if not freetime_transit:
                freetime_transit=(route_length/5.5)/60
            # 添加边
            Graph_MhdCBD.add_edge(start_graph_node, end_graph_node, total_flow=total_flow, 
                                edge_indices=path_edges, route_length=route_length, 
                                freetime_dirve=freetime_dirve,freetime_transit=freetime_transit,
                                driving_travelor=driving_travelor,
                                transit_travelor=transit_travelor,
                                taxi_travelor=taxi_travelor)
            start_coords = (nodes.loc[start_road_node_index, 'geometry'].x, nodes.loc[start_road_node_index, 'geometry'].y)
            end_coords = (nodes.loc[end_road_node_index, 'geometry'].x, nodes.loc[end_road_node_index, 'geometry'].y)

            # 创建直线箭头表示交通流方向
            arrow = FancyArrowPatch(
                posA=start_coords, posB=end_coords,
                arrowstyle="->", mutation_scale=10, color="blue", lw=1, zorder=1
            )

            # 添加箭头到图形
            ax.add_patch(arrow)
    except Exception as e :
        print(f"""绘制 {node} 的edges 失败：错误信息:{e}.""")
        continue
# 将转换后的字典保存为 JSON 文件
dict_to_save = {f"({v},{u})": traffic_value for (v, u), traffic_value in all_edge_U_V_Taffic_dict.items()}
with open('Graph_MhdCBD_edge_traffic.json', 'w') as json_file:
    json.dump(dict_to_save, json_file, indent=4)
print('edges 信息 保存成功！')
#保存网络图
with open('Graph_MhdCBD.pkl', 'wb') as f:
    pickle.dump(Graph_MhdCBD, f)
print('Graph_MhdCBD 保存成功！')


file_path = 'Graph_MhdCBD_nodes_not_in_OD_data.json'
# 将列表存储到 JSON 文件中
with open(file_path, 'w') as json_file:
    json.dump(nodes_not_in_OD_data, json_file)



# for mid_x, mid_y, dx, dy in arrow_positions:
#     arrow = FancyArrowPatch(
#         (mid_x, mid_y),                # 箭头起点
#     (mid_x + dx / 10, mid_y + dy / 10),  # 箭头终点
#     arrowstyle="fancy",               # 仅绘制箭头尖
#     color='red',                   # 箭头颜色
#     mutation_scale=3,  
#     linewidth=1,             # 箭头大小
#     alpha=0.9,                     # 透明度
#     zorder=5                       # 图层顺序
#     )
#     ax.add_patch(arrow)

plt.savefig('Graph_MhdCBD.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
