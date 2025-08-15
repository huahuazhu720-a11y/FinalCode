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

def show(districts):
    fig, ax = plt.subplots(figsize=(10, 10))
    districts.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.5)
    # 设置标题
    plt.title("NY Census Blocks")
    # 隐藏坐标轴
    plt.axis("off")
    # 显示图像
    plt.show()

# 读取并转换区域的 CRS
districts = gpd.read_file(r"shapfile\Demographic.shp")
districts = districts.to_crs(epsg=4326)


# 使用 place_name 获取 New York City 的路网数据
place_name = "Seattle, Washington, USA"


# 检查是否已有存储的 graph 文件
if os.path.exists("Seattle.pickle"):
    with open("Seattle.pickle", "rb") as f:
        graph = pickle.load(f)
else:
    graph = ox.graph_from_place(place_name, network_type="drive")
    with open("Seattle.pickle", "wb") as f:
        pickle.dump(graph, f)

# 检查是否已有存储的 edges 文件
if os.path.exists("edges_Seattle.geojson"):
    edges = gpd.read_file("edges_Seattle.geojson")
else:
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges.to_file("edges_Seattle.geojson", driver="GeoJSON")

# 检查是否已有存储的 nodes 文件
if os.path.exists("nodes_Seattle.geojson"):
    nodes = gpd.read_file("nodes_Seattle.geojson")
else:
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    nodes.to_file("nodes_Seattle.geojson", driver="GeoJSON")


# New York County	36061
#  Kings County	36047
# 	Queens County	36081
# 	Bronx County	36005
# 	Richmond County	36085
#  011402  ct 114.02
# 选择指定的区域
selected_ids = [
   "53033000500","53033000401","53033000300","53033000200","53033000100",
    "53033000402","53033000600","53033000700","53033000800","53033000900",
    "53033001500","53033001600","53033001400","53033001702","53033001701","53033001300","53033001800","53033001200",
    "53033001100","53033001900","53033002000","53033002100","53033002200",
    "53033003100","53033003000","53033002900","53033002800","53033002700","53033002600","53033002500","53033002400",
    "53033003800","53033003900","53033004000",
    "53033003200","53033003300","53033004700","53033003400","53033004800","53033003500","53033004900",
    "53033004600","53033003600","53033004500","53033005000","53033005400","53033005100","53033005200",
    "53033005301","53033004400","53033004301","53033004302","53033004200","53033005302","53033004100",
    "53033006100",
    "53033005700","53033005600","53033005801","53033005802","53033005900","53033006900","53033006000",
    "53033006800","53033007000","53033007100","53033006700","53033007200","53033008001","53033008002",
    "53033008100","53033006600","53033007300","53033008200","53033008300","53033008500","53033008400",
    "53033007402","53033007401","53033006500","53033007500","53033008600","53033008700","53033007900",
    "53033007600","53033006400","53033006200","53033007700","53033008800","53033007800","53033006300",
    "53033009600","53033009701","53033009702","53033009800","53033010500","53033010600","53033011600",
    "53033011500","53033011401","53033010702","53033010701","53033009900","53033010800","53033009300",
    "53033009200","53033009100","53033009000","53033009400","53033010002","53033010001","53033010402",
    "53033010401","53033011002","53033011001","53033011800","53033011101","53033011102","53033010200",
    "53033010300","53033010100","53033009500","53033008900",
    "53033012100", "53033012000","53033011402","53033011300","53033011200","53033010900","53033011700","53033011900",
    "53033001000"
    ]
filtered_districts = districts[districts["GEO_ID_TRT"].isin(selected_ids)]

# 计算区域的中心点
filtered_districts.loc[:, "center"] = filtered_districts.geometry.centroid

# 筛选出区域内的道路
roads_in_districts = edges[edges.geometry.within(filtered_districts.union_all())]

# 绘制地图
fig, ax = plt.subplots(figsize=(15, 30))
filtered_districts.plot(ax=ax, edgecolor="black", color="white", linewidth=3)
roads_in_districts.plot(ax=ax, edgecolor="darkgrey", linewidth=1)
# 创建网络图new  graph
Final_Graph= nx.DiGraph()


# 连接到数据库
DB=Database("Sensus.db")
# 使用2019 年的数据来画图
year="2019"
all_sensus_data = DB.execute_query("SELECT * FROM SensusData WHERE year=?", (year,))
# 转换为字典
all_sensus_data_dict = {row[1]: row for row in all_sensus_data}

query = f"""
            SELECT 
                S.tract,
                sum(I.number_of_jobs)                 
            FROM 
                SensusData AS S 
            JOIN 
                InNodes AS I 
            ON                
                I.Id_in_SensusData = S.id                
            WHERE 
                S.year = ? 
            Group by S.tract                
                
        """
In_workface = DB.execute_query(query, (2019,))
In_workface_dict={row[0]: row[1] for row in In_workface}
query = f"""
            SELECT 
                S.tract,                    
                sum(O.number_of_jobs)
            FROM 
                SensusData AS S                 
            JOIN 
                OutNodes AS O
            ON 
                O.Id_in_SensusData = S.id
            WHERE 
                S.year = ? 
            Group by S.tract                 
                
        """
out_workface = DB.execute_query(query, (2019,))
out_workface_dict={row[0]: row[1] for row in out_workface}



# 找到每个区域内最近的 `node` 并绘制圆
for count, (idx, row) in enumerate(filtered_districts.iterrows()):  
    #数据库查询该节点属性
    tract=row["GEO_ID_TRT"]
    
    tract_sensusData=all_sensus_data_dict[tract]
    if row['Between18a']:
        TotalPopulation=int(row['Between18a'])
    else:
        TotalPopulation=0
    # 判断是什么type
    if tract in In_workface_dict:
        in_degree=float(In_workface_dict[tract])
    else:
        in_degree=0
    if tract in In_workface_dict:
        out_degree=float(out_workface_dict[tract])
    else:
        out_degree=0
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
    radius = TotalPopulation*0.001*0.0004
    min_radius = 0.0001  # 最小半径
    radius = max(radius, min_radius)
    circle = Point(nearest_node_coords).buffer(radius)  # 半径为 0.001 度
    gpd.GeoSeries([circle]).plot(ax=ax, facecolor=facecolor, linestyle="--",zorder=10)    
    # 绘制最近的 `node`（圆心）
    # ax.plot(nearest_node_coords[0], nearest_node_coords[1], 'ro')  # 绘制节点点位

    print(f"""绘制node: {tract}""")
    #这里直接使用tract_id作为节点的名称
    Final_Graph.add_node(row["GEO_ID_TRT"],
                         districtIndex=idx,
                         TotalPopulation=TotalPopulation,
                         center_index=nearest_node_index, # node 节点的index
                         Road_node=nodes.loc[nearest_node_index],
                         Type=nodetype)

# 统一查询数据库，避免在遍历中查询，提高性能
tract_numbers_str = ",".join(f"'{tract}'" for tract in selected_ids)
query = f"""
            SELECT 
                S.tract,
                I.SensusTractNum, 
                I.number_of_jobs
            FROM 
                SensusData AS S 
            JOIN 
                InNodes AS I 
            ON 
                I.Id_in_SensusData = S.id
            WHERE 
                S.year = ?                 
                AND I.SensusTractNum IN ({tract_numbers_str})
        """
In_connection = DB.execute_query(query, (2019,))
in_connection_tract_dict = {}
for tract, sensus_tract, num_of_jobs in In_connection:
    if tract not in in_connection_tract_dict:
        in_connection_tract_dict[tract] = []
    in_connection_tract_dict[tract].append((sensus_tract, num_of_jobs))

query = f"""
            SELECT 
                S.tract,
                I.SensusTractNum, 
                I.number_of_jobs
            FROM 
                SensusData AS S 
            JOIN 
                OutNodes AS I 
            ON 
                I.Id_in_SensusData = S.id
            WHERE 
                S.year = ?                 
                AND I.SensusTractNum IN ({tract_numbers_str})
        """
Out_connection = DB.execute_query(query, (2019,))
Out_connection_tract_dict = {}
for tract, sensus_tract, num_of_jobs in Out_connection:
    if tract not in Out_connection_tract_dict:
        Out_connection_tract_dict[tract] = []
    Out_connection_tract_dict[tract].append((sensus_tract, num_of_jobs))




#定义统一绘制路径和方向的数组
all_routes = []
arrow_positions = []
nodes_not_in_OD_data=[]
all_edge_U_V_Taffic_dict={}
# 给每个node加加边，并画图
for node, data in Final_Graph.nodes(data=True):
    print(f"""绘制 {node} 的edges""")
    #检查node是否在inconnect里面有数据，如果没有，代表这个node里一个工作岗位都没有。再检查是否在outconnection
    if node in in_connection_tract_dict:
        connections= in_connection_tract_dict[node]
        from_in_connection=True
        from_out_connection=False
    elif node in Out_connection_tract_dict:
        connections= in_connection_tract_dict[node]
        from_in_connection=False
        from_out_connection=True
    else:
        nodes_not_in_OD_data.append(f"""this {node} has no edges""")
        continue
    # 开始连接节点
    for row in connections:  
        if from_in_connection:
            start_graph_node=str(row[0])
            end_graph_node=node
        else:
            start_graph_node=node
            end_graph_node=str(row[0])
        #如果在本区工作，或者已经有连接了
        if start_graph_node==end_graph_node or Final_Graph.has_edge(start_graph_node, end_graph_node):
            continue
        total_flow=float(row[1])
        # 获取 两点间道路node的信息
        start_road_node_index=Final_Graph.nodes[start_graph_node]["center_index"] #取得这个node 里面中心点，也就是离中心点最近的道路的node的index
        end_road_node_index=Final_Graph.nodes[end_graph_node]["center_index"] #取得这个node 里面中心点，也就是离中心点最近的道路的node的index        
        start_road_node = nodes.loc[start_road_node_index, 'osmid']  
        end_road_node =nodes.loc[end_road_node_index, 'osmid']#这里得到的index
        # 获取 起点和终点的路径信息
        route = nx.shortest_path(graph, source=start_road_node, target=end_road_node, weight='length')
        route_length = nx.shortest_path_length(graph, source=start_road_node, target=end_road_node, weight='length')
        path_edges = [] #存储了edge的起点终点和对应的交通流量
        for i in range(len(route) - 1):
            u, v, traffic = route[i], route[i + 1], row[1]  # Edge in the graph            
            path_edges.append((u, v, traffic))
            if (u, v) in all_edge_U_V_Taffic_dict:
                all_edge_U_V_Taffic_dict[(u, v)] += total_flow
            else:
                all_edge_U_V_Taffic_dict[(u, v)] = total_flow     
        
        
        # free time 的计算规则，如果距离超过5000米，则不考虑骑自行车的情况。
        #           自行车的速度按 17.5 km/h, 开车的速度按 52 km/h，公交车按照20km/h 计算
        freetime={}
        freetime['car_freetim']=(route_length/52000)*60
        freetime['bike_freetim']=(route_length/17500)*60
        freetime['pulicTraffic_freetim']=(route_length/20000)*60
        freetime['unit']="minute"
        Final_Graph.add_edge(start_graph_node, end_graph_node, total_flow=total_flow,edge_indices=path_edges,route_length=route_length,freetime=freetime)

# 将转换后的字典保存为 JSON 文件
dict_to_save = {f"({v},{u})": traffic_value for (v, u), traffic_value in all_edge_U_V_Taffic_dict.items()}
with open('edge_traffic.json', 'w') as json_file:
    json.dump(dict_to_save, json_file, indent=4)
print('edges 信息 保存成功！')
#保存网络图
with open('graph.pkl', 'wb') as f:
    pickle.dump(Final_Graph, f)
print('graph 保存成功！')


file_path = 'nodes_not_in_OD_data.json'
# 将列表存储到 JSON 文件中
with open(file_path, 'w') as json_file:
    json.dump(nodes_not_in_OD_data, json_file)



# 给每一段路根据道路traffic流量来使用颜色表示拥堵情况 以及画出箭头代表方向
cmap = mcolors.LinearSegmentedColormap.from_list(
    'traffic_cmap', ['green', 'red', 'black'], N=100
)
traffic_values = list(all_edge_U_V_Taffic_dict.values())  # 获取所有 traffic 的值
norm = plt.Normalize(min(traffic_values), max(traffic_values))
edge_colors = []
for (u, v), traffic_value in all_edge_U_V_Taffic_dict.items():
    edge = edges[(edges['u'] == u) & (edges['v'] == v)]
    if not edge.empty:
        edge_geom = edge.geometry.values[0]
        all_routes.append(edge_geom)
        x, y = edge_geom.xy               
        # 在每个部分的中点绘制箭头
        mid_index = len(x) // 3
        for i in range(len(x) - 1):
            mid_x = (x[mid_index] + x[mid_index + 1]) / 2  # 中点 x 坐标
            mid_y = (y[mid_index] + y[mid_index + 1]) / 2  # 中点 y 坐标
            dx = x[mid_index + 1] - x[mid_index]          # 方向向量 dx
            dy = y[mid_index + 1] - y[mid_index]          # 方向向量 dy
            arrow_positions.append((mid_x, mid_y, dx, dy))
        color = cmap(norm(traffic_value))  # 使用归一化后的值来选择颜色
        edge_colors.append(color)
        



#这里才来绘制边和箭头
gpd.GeoSeries(all_routes).plot(ax=ax, color=edge_colors, linewidth=1)
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
plt.title("Selected Districts with Circles around Nearest Nodes")
plt.savefig('Geograph.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
