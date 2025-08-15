"""
step10 Calculate the traffic capacity of each road for every year (2022 used as an example) and update the network graph accordingly.
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




def calculate_road_capacity(edge_tuples, edges):
    total_capacity = 0
    total_length = 0
    highway_capacity = {
        'motorway': 2200,
        'primary': 1900,
        'secondary': 1600,
        'tertiary': 1500,
        'residential': 800,
        'living_street': 400,
        'motorway_link': 2200,
        'primary_link': 1900,
        'secondary_link': 1600,
        'tertiary_link': 1500,
        'unclassified': 1200,
        'busway': 1200,
        'trunk': 2100,
        'trunk_link': 2100
    }
    speed_limits_adjustment = {
        "30 mph": 0.5, 
        "40 mph": 0.6,
        "50 mph": 0.7,  # 50 mph -> 80.47 km/h
        "60 mph": 0.8,  # 60 mph -> 96.56 km/h
        "70 mph": 0.9,  # 70 mph -> 112.65 km/h
        "80 mph": 1.0,   # 80 mph -> 128.75 km/h
        "90 mph": 1.1   # 80 mph -> 128.75 km/h
    }
    for u, v, _ in edge_tuples:
        # 查找对应的 edge
        edge = edges[(edges['u'] == u) & (edges['v'] == v) & (edges['key'] == 0)]
        
        if not edge.empty:
            try:
                # 获取每条边的通行能力
                highway_type = edge['highway'].fillna(0).iloc[0]
                
                lanes = edge['lanes'].iloc[0]
          
                length = edge['length'].fillna(0).iloc[0]      
                maxspeed = edge['maxspeed'].iloc[0]          
                # 计算这条边的通行能力：capacity_per_lane * lanes
                capacity_per_lane = highway_capacity.get(highway_type, 1600)  # 默认值为 secondary
                maxspeed_factor = speed_limits_adjustment.get(maxspeed, 0.7)  # 默认值为城市道路按0.7算
                edge_capacity = float(capacity_per_lane) * float(lanes)  * float(maxspeed_factor)     
                # 计算加权通行能力
                total_capacity += edge_capacity * length
                total_length+=length
            except Exception:
                continue
          
    
    # 返回加权平均通行能力
    if total_length > 0:
        return round(total_capacity / total_length,2)
    else:
        return 0  # 防止除以0的情况
def run():
    with open("NewYorkCity.pickle", "rb") as f:
        graph = pickle.load(f)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges = edges.reset_index()
    edges['lanes'] = edges['lanes'].apply(lambda x: max(map(int, x)) if isinstance(x, list) else x)
    edges['lanes'] = edges['lanes'].fillna(1)
    unique_values = edges['lanes'].unique()
    print(unique_values)
    # years=["2011","2012","2013","2014","2015","2016","2017","2018","2019",
    #         "2020","2021","2022"]
    years=["2022"]
    for year in years:
        Graph_NewYork=pd.read_pickle(f'Graph_NewYork_Manhattan{year}.pkl') 
        for u, v, data in Graph_NewYork.edges(data=True):
            path_edges=Graph_NewYork[u][v].get("edge_indices", None)
            a=calculate_road_capacity(path_edges,edges)
            data['Road_Capacity'] = a
        net_work_name=f"Graph_NewYork_Manhattan{year}.pkl"
        #保存网络图
        with open(net_work_name, 'wb') as f:
            pickle.dump(Graph_NewYork, f)
        print(f"Graph_NewYork_Manhattan{year} 保存成功！")

if __name__ == "__main__":
    run()
