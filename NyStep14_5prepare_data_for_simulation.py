
"""
step 14_5
prepare data for simulation
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

def prepare_Data_for_manhattan_simulation():
    name=f"Graph_NewYork_Manhattan2022"
    Graph_NewYork_Manhattan=pd.read_pickle(f'{name}.pkl') 
    specific_nodes = [114, 61, 119, 98, 59, 103, 90, 69]
    name=f"Graph_NewYork_2022"
    Graph_NewYork_old=pd.read_pickle(f'{name}.pkl') 
    edges_to_remove = []
    index=1
    for u, v, data in Graph_NewYork_old.edges(data=True):
        if u in specific_nodes:
            Manhaon_total_flow=False
        else:
            Manhaon_total_flow=Graph_NewYork_Manhattan[u][v]['total_flow']
        if Manhaon_total_flow:
            Graph_NewYork_old[u][v]['total_flow_Manhattan']=Manhaon_total_flow
            Graph_NewYork_old[u][v]['fixed_flow_without_Manhattan']= Graph_NewYork_old[u][v]['driving_travelor']+Graph_NewYork_old[u][v]['taxi_travelor']-Graph_NewYork_Manhattan[u][v]['driving_travelor']-Graph_NewYork_Manhattan[u][v]['taxi_travelor']
            Graph_NewYork_old[u][v]['driving_travelor_Manhattan']= Graph_NewYork_Manhattan[u][v]['driving_travelor']+Graph_NewYork_Manhattan[u][v]['taxi_travelor']
            Graph_NewYork_old[u][v]['transit_travelor_Manhattan']= Graph_NewYork_Manhattan[u][v]['transit_travelor']
            Graph_NewYork_old[u][v]['link_index']=index
            Graph_NewYork_old[u][v]['fixed_flow_without_Manhattan_PT']=Graph_NewYork_old[u][v]['transit_travelor']-Graph_NewYork_Manhattan[u][v]['transit_travelor']
            index+=1        
        else:
            Graph_NewYork_old[u][v]['total_flow_Manhattan']=0
            Graph_NewYork_old[u][v]['fixed_flow_without_Manhattan']= Graph_NewYork_old[u][v]['driving_travelor']+Graph_NewYork_old[u][v]['taxi_travelor']
            Graph_NewYork_old[u][v]['driving_travelor_Manhattan']= 0
            Graph_NewYork_old[u][v]['transit_travelor_Manhattan']= 0
            Graph_NewYork_old[u][v]['fixed_flow_without_Manhattan_PT']=Graph_NewYork_old[u][v]['transit_travelor']
            Graph_NewYork_old[u][v]['link_index']=222
    Graph_NewYork_old.remove_edges_from(edges_to_remove)     
    for node, data in Graph_NewYork_old.nodes(data=True):

        Out_traveler_in_manhatton = Graph_NewYork_Manhattan.nodes.get(node, {}).get('Out_traveler', 0)
        data['Out_traveler_Manhattan'] = Out_traveler_in_manhatton

    net_work_name=f"2022_for_manhattan_simulation.pkl"
   
    with open(net_work_name, 'wb') as f:
        pickle.dump(Graph_NewYork_old, f)
    print(f"{net_work_name} 保存成功！")

def run():
    prepare_Data_for_manhattan_simulation() 
    name=f"2022_for_manhattan_simulation"
    Graph_NewYork_old=pd.read_pickle(f'{name}.pkl') 
    edges_data = []

    for u, v, data in Graph_NewYork_old.edges(data=True):
        row = {'source': u, 'target': v}
        row.update(data)  # 把所有属性加入
        edges_data.append(row)

    # 转换为 DataFrame
    edges_df = pd.DataFrame(edges_data)

    # 保存为 CSV
    edges_df.to_csv('2022_for_manhattan_simulation.csv', index=False)

if __name__ == "__main__":  
    run()
    



    

        

    