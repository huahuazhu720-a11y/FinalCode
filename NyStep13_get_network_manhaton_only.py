"""
step13
get network graph only the traffice flow related to Manhaton 
"""
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
from networkx import ego_graph
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from itertools import islice
import pandas as pd
import requests
def get_total_degree(node, G):
    return G.in_degree(node) + G.out_degree(node)

def get_free_flow_time(nodes, start_node, end_node, mode):
    API_KEY = os.getenv('Google_Map_API_KEY')
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


def run():
    # 读取并转换区域的 CRS
    districts = gpd.read_file(r"shapfile\merged_taxi_zones.shp").to_crs(epsg=4326)
    # selected_ids = [
    #    1,2,3,4
    #     ]
    # districts = districts[districts["zone_id"].isin(selected_ids)]

    # print(districts.columns)
    # 使用 place_name 获取 New York City 的路网数据
    place_name = "New York City, New York, USA"
    adjacent_districts = pd.read_pickle('adjacent_districts.pkl')
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
    Net_work_Old=pd.read_pickle('Graph_NewYork_2022.pkl') 

    # New York County	36061
    #  Kings County	36047
    # 	Queens County	36081
    # 	Bronx County	36005
    # 	Richmond County	36085
    #  011402  ct 114.02



    # 创建网络图new  graph
    Graph_NewYork= nx.DiGraph()


    # 连接到数据库
    DB=Database("GreYelHir.db")
    # 使用2022 年的数据来画图
    years=["2022"]
    for year in years:
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

        # 只是选取曼哈顿地区的
        query = f"""
                    SELECT 
                        *               
                    FROM 
                        ZoneOutNodes
                    WHERE 
                        year={year}    
                        AND   OutZone_id IN (114, 61, 119, 98, 59, 103, 90, 69)      
                        
                """
        Out_workface_all = DB.sql_to_df(query)
        Out_workface=Out_workface_all[Out_workface_all['zone_id']!=Out_workface_all['OutZone_id']]
        # Out_workface=Out_workface.groupby('zone_id')['count_of_jobs'].sum().reset_index()
        Out_workface["zone_id"]=Out_workface["zone_id"].astype(int)
        Out_workface["OutZone_id"]=Out_workface["OutZone_id"].astype(int)
        # print(Out_workface.head())
        query = f"""
                    SELECT 
                        *
                    FROM 
                        ZoneInNodes                 
                    WHERE 
                        year={year} 
                        AND zone_id IN (114, 61, 119, 98, 59, 103, 90, 69) 
                
                """
        In_workface_all = DB.sql_to_df(query)
        In_workface=In_workface_all[In_workface_all['zone_id']!=In_workface_all['InZone_id']]
        # In_workface=In_workface.groupby('zone_id')['count_of_jobs'].sum().reset_index()
        In_workface["InZone_id"]=In_workface["InZone_id"].astype(float).astype(int)
        In_workface["zone_id"]=In_workface["zone_id"].astype(int)
        # print(In_workface.head())

        # 找到每个区域内最近的 `node` 并绘制圆
        for zone_id in range(130):    
            #数据库查询该节点属性  
            zone_id+=1
            TotalPopulation = all_sensus_data.loc[all_sensus_data['zone_id'] == zone_id, 'S0801_C01_001E'].iloc[0] if not all_sensus_data.loc[all_sensus_data['zone_id'] == zone_id, 'S0801_C01_001E'].empty else 0
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
            facecolor="green" # 不体现type
            

            nearest_node_index=Net_work_Old.nodes[zone_id].get('center_index', None)
            # 绘制圆（以最近的 `node` 作为圆心）
            print(f"""处理{year}: {zone_id}""")
        
            #这里使用zone_id作为节点的名称
            Graph_NewYork.add_node(zone_id,
                                In_tarveler=in_degree,#从其他区到本区工作的总人口
                                Out_traveler=out_degree,#从本区出去工作的总人口
                                TotalPopulation=TotalPopulation,
                                drive_rate=drive_rate,
                                transit_rate=transit_rate,
                                taxi_rate=taxi_rate,
                                center_index=nearest_node_index, # node 节点的index
                                Road_node=nodes.loc[nearest_node_index,'osmid'],
                                Type=nodetype)


            # 遍历每个节点连接的点
        #定义统一绘制路径和方向的数组
        all_edge_U_V_Taffic_dict={}
        # 给每个node加加边，并画图
        for node, graph_node_data in Graph_NewYork.nodes(data=True):
            print(f"""处理{year} {node} 的edges""")
            try:
                    
                for i in adjacent_districts[node]:                
                    start_graph_node = node
                    end_graph_node = i
                    #获取这start node和end node的通勤总数
                    if True:
                        total_flow=0
                        driving_travelor=0
                        transit_travelor=0
                        taxi_travelor=0             
                    
                    
                    start_road_node = Graph_NewYork.nodes[start_graph_node]["Road_node"]
                    start_road_node_index = Graph_NewYork.nodes[start_graph_node]["center_index"]
                    end_road_node = Graph_NewYork.nodes[end_graph_node]["Road_node"]
                    end_road_node_index = Graph_NewYork.nodes[end_graph_node]["center_index"]
                    # 先检查原来的图有没有记录
                    if Net_work_Old.has_edge(start_graph_node, end_graph_node): 
                        route_length=Net_work_Old[start_graph_node][end_graph_node]['route_length']
                        path_edges=Net_work_Old[start_graph_node][end_graph_node].get("edge_indices", None)
                        freetime_dirve=Net_work_Old[start_graph_node][end_graph_node].get("freetime_dirve", None)
                        freetime_transit=Net_work_Old[start_graph_node][end_graph_node].get("freetime_transit", None)
                        taxi_cost=Net_work_Old[start_graph_node][end_graph_node].get("taxi_cost", None)
                        dirving_cost=Net_work_Old[start_graph_node][end_graph_node].get("dirving_cost", None)
                        transit_cost=Net_work_Old[start_graph_node][end_graph_node].get("transit_cost", None)
                    else:                    
                        max_attempts = 5
                        attempt = 0
                        found_path = False

                        while attempt < max_attempts:
                            try:
                                route = nx.shortest_path(graph, source=start_road_node, target=end_road_node, weight='length')
                                #获取路径长度，单位：米
                                route_length = nx.shortest_path_length(graph, source=start_road_node, target=end_road_node, weight='length')
                                route_length= round(route_length/1609.34,2)
                                dirving_cost=round((route_length/24.17)*2.995,2)
                                transit_cost=round(3.3,2)
                                taxi_cost=round(route_length*6.128,2)
                                found_path = True
                                break
                            except nx.NetworkXNoPath:
                                print(f"Attempt {attempt + 1}: No route between node {start_road_node} and node {end_road_node}, total_flow: {total_flow}")

                                
                                start_neighbors = list(graph.neighbors(start_road_node))
                                end_neighbors = list(graph.neighbors(end_road_node))

                                if start_neighbors:
                                    start_road_node = min(start_neighbors, key=lambda n: graph.nodes[n].get('traffic_weight', 1))
                            
                                if end_neighbors:
                                    end_road_node = min(end_neighbors, key=lambda n: graph.nodes[n].get('traffic_weight', 1))
                                
                                attempt += 1

                        if not found_path:
                                print(f"Cannot find route from start_road_node: {start_road_node} in start_graph_node {start_graph_node} to end_road_node: {end_road_node} in graph node :{end_graph_node}, total_flow is {total_flow}.")
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
                    Graph_NewYork.add_edge(start_graph_node, end_graph_node, total_flow=total_flow, 
                                        edge_indices=path_edges, route_length=route_length, 
                                        freetime_dirve=freetime_dirve,freetime_transit=freetime_transit,
                                        driving_travelor=driving_travelor,
                                        transit_travelor=transit_travelor,
                                        taxi_travelor=taxi_travelor,
                                        taxi_cost=taxi_cost,
                                        dirving_cost=dirving_cost,
                                        transit_cost=transit_cost)               
                        

            except Exception as e :
                print(f"""绘制 {node} 的edges 失败：错误信息:{e}.""")
                continue
        
        # 更新各个edge的workflow，各个交通模式的数量
        for node, graph_node_data in Graph_NewYork.nodes(data=True):
            print(f"update workflow and traffic model for the edge that out from {node}.")
            if node in Out_workface['zone_id'].values:
                connections= Out_workface.loc[Out_workface['zone_id'] == node]
                connection_list = list(zip(connections["OutZone_id"], connections["count_of_jobs"]))
                from_out=True
                try:#获取该点出发的交通模式的比例 
                    drive_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(node), 'driving_rate']), 0)
                    transit_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(node), 'transit_rate']), 0)
                    taxi_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(node), 'taxi_rate']), 0)
                except Exception:
                    drive_rate=0
                    transit_rate=0
                    taxi_rate=0
            elif node in In_workface['zone_id'].values:
                connections= In_workface.loc[In_workface['zone_id'] == node]
                connection_list = list(zip(connections["InZone_id"], connections["count_of_jobs"]))
                from_out=False            
            else:
                continue
            # 开始连接节点      
            total=len(connection_list)
            counter=0
            for row in connection_list:
                counter+=1
                # print(f"deal {counter} connections out of {total} on node {node}.")
                if from_out:
                    start_graph_node=node
                    end_graph_node=row[0]
                else:
                    start_graph_node=row[0]
                    end_graph_node=node
                    try:#获取该点出发的交通模式的比例 
                        drive_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(start_graph_node), 'driving_rate']), 0)
                        transit_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(start_graph_node), 'transit_rate']), 0)
                        taxi_rate = next(iter(all_sensus_data.loc[all_sensus_data['zone_id'] == int(start_graph_node), 'taxi_rate']), 0)
                    except Exception:
                        drive_rate=0
                        transit_rate=0
                        taxi_rate=0

                if start_graph_node<=0 or end_graph_node<=0:
                    continue
                total_flow = int(row[1])
                        
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
                
                
                try:
                    path = nx.shortest_path(Graph_NewYork, source=start_graph_node, target=end_graph_node, weight="route_length")
                except Exception:
                    # 如果这个记录没有路径可以到达，就放弃
                    continue

                for u, v in zip(path, path[1:]):
                    Graph_NewYork[u][v]['total_flow'] += total_flow
                    Graph_NewYork[u][v]['driving_travelor'] += driving_travelor
                    Graph_NewYork[u][v]['transit_travelor'] += transit_travelor
                    Graph_NewYork[u][v]['taxi_travelor'] += taxi_travelor 
        # 把freetime_dirve，freetime_transit，taxi_cost 保留2位小数
        for u, v, data in Graph_NewYork.edges(data=True):        
            data['freetime_dirve'] = np.round(data['freetime_dirve'], 2)
            data['freetime_transit'] = np.round(data['freetime_transit'], 2)
            data['taxi_cost'] = np.round(data['taxi_cost'], 2)
                







        net_work_name=f"Graph_NewYork_Manhattan{year}.pkl"
        #保存网络图
        with open(net_work_name, 'wb') as f:
            pickle.dump(Graph_NewYork, f)
        print(f"Graph_NewYork_Manhattan{year} 保存成功！")



