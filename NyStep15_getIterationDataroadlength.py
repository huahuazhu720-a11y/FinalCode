"""
some functions for Temporary tasks
"""

import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import numpy as np
import pickle
from database import Database
import pandas as pd
import matplotlib.colors as mcolors
import networkx as nx 
from matplotlib.patches import FancyArrowPatch
import sympy as sp
import numpy as np
import pandas as pd

def single_link_Travel_time(free_flow_travel_time,traffic_flow,capacity,kappa):
    """
    参数:
    traffic_flow: 是一个list,第一个是pulic_flow，第二个是vehicle_flow
    free_flow_travel_time: 是一个list,第一个是public_traveltime，第二个是vehicle_raveltime
    capacity: 道路容量常数
    kappa: 固定常数
    返回更新的:
    (public_traveltime,vehicle_raveltime)
    """
    pulic_flow=traffic_flow[0]
    vehicle_flow=traffic_flow[1]
    public_traveltime=free_flow_travel_time[0]
    vehicle_raveltime=free_flow_travel_time[1]+kappa* sp.log(1 + vehicle_flow/capacity)
    return np.array(public_traveltime,vehicle_raveltime)

def get_iteration_matrix(year):
    data=pd.read_pickle(f"Graph_NewYork_{year}.pkl")
    edge_data = []
    num_links = data.number_of_edges()
    print(num_links)
    for u, v, data in data.edges(data=True):
        denominator = data['driving_travelor'] + data['taxi_travelor']
        denominator = 2 if denominator == 0 else denominator 
        edge_data.append({
            'start_node': int(u),
            'end_node': int(v),
            'route_length':float(data['route_length'])
            
            })
    edges_df = pd.DataFrame(edge_data)
    edges_df.to_csv(f"network_links_data_{year}.csv")


if __name__ == "__main__":



   
    get_iteration_matrix(2022)


