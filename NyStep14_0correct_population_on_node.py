
"""
Correct the population on each node, make sure the population is only related the traffice flow to Manhaton
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
def run():
    years=["2022"]
    specific_id=[114, 61, 119, 98, 59, 103, 90, 69]
    for year in years:
        name=f"Graph_NewYork_Manhattan{year}"
        Graph_NewYork=pd.read_pickle(f'{name}.pkl') 
        DB=Database("GreYelHir.db")
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
        Out_workface=Out_workface.groupby('zone_id')['count_of_jobs'].sum().reset_index()
        Out_workface["zone_id"]=Out_workface["zone_id"].astype(int)


        

        for node, graph_node_data in Graph_NewYork.nodes(data=True):
            if node in specific_id:
                 graph_node_data["Out_traveler"] =0
            else:
                 Out_traveler = Out_workface[Out_workface['zone_id'] == node].get('count_of_jobs', pd.Series([0])).iloc[0]
                      
                 graph_node_data["Out_traveler"] =Out_traveler
        net_work_name=f"{name}.pkl"
        #保存网络图
        with open(net_work_name, 'wb') as f:
            pickle.dump(Graph_NewYork, f)
        print(f"{net_work_name} 保存成功！")

if __name__ == "__main__":
    run()
    

