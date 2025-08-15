
"""
step11
Includes several functions:
correct_workflow – Mainly corrects data from the initial version. history version, just keep it here, not used anymore
get_holidays – Retrieves holidays for a given year.history version, just keep it here, not used anymore
calculate_taxi_cost – Calculates the cost per mile for taxis; very important.history version, just keep it here, not used anymore
get_edge_to_csv – Exports the edges of a graph to a CSV file.history version, just keep it here, not used anymore
add_Taxi_Cost – Adds the calculated taxi costs to the network.history version, just keep it here, not used anymore
Prepare_data_for_getting_parameters_multiprocessing – Prepares data for subsequent parameter estimation.
"""




import numpy as np
import pickle
import holidays
import pandas as pd

import networkx as nx 

from database import Database
import logging  
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
logging.basicConfig(
    filename="error_log.log",
    level=logging.INFO,  # INFO 可以记录 INFO 及更高的级别（ERROR, CRITICAL）
    format="%(asctime)s - %(levelname)s - Year: %(year)s, ID: %(id)s, Error: %(error_message)s, CustomMessage: %(custom_message)s"
)

def correct_workflow():
    DB=Database("GreYelHir.db")
    Graph_NewYork_New=pd.read_pickle('Graph_NewYork_New.pkl')
    Net_work_Old=pd.read_pickle('Graph_NewYork.pkl')
    query = f"""
                SELECT 
                    *               
                FROM 
                    ZoneOutNodes
                WHERE 
                    year=2022          
                    
            """
    Out_workface_all = DB.sql_to_df(query)
    Out_workface=Out_workface_all[Out_workface_all['zone_id']!=Out_workface_all['OutZone_id']]
    # Out_workface=Out_workface.groupby('zone_id')['count_of_jobs'].sum().reset_index()
    Out_workface["zone_id"]=Out_workface["zone_id"].astype(int)
    Out_workface["OutZone_id"]=Out_workface["OutZone_id"].astype(int)
    Out_workface["count_of_jobs"]=Out_workface["count_of_jobs"].astype(int)
    Out_workface_array = Out_workface.to_numpy()  # 将 DataFrame 转换为 NumPy 数组，提高性能
    nx.set_edge_attributes(Graph_NewYork_New, 0, 'total_flow')
    nx.set_edge_attributes(Graph_NewYork_New, 0, 'driving_travelor')
    nx.set_edge_attributes(Graph_NewYork_New, 0, 'transit_travelor')
    nx.set_edge_attributes(Graph_NewYork_New, 0, 'taxi_travelor')
    total=len(Out_workface_array)
    counter=0
    for row in Out_workface_array:
        start_node=row[2]
        end_node=row[3]
        print(f"deal {counter} out of {total}")
        counter+=1
        if start_node<=0 or end_node<=0:
            continue
        try:
            path = nx.shortest_path(Graph_NewYork_New, source=start_node, target=end_node, weight="route_length")
        except Exception:
            continue
        for u, v in zip(path, path[1:]):
            if Graph_NewYork_New.has_edge(u, v):
                a=Graph_NewYork_New[u][v]['total_flow']
                Graph_NewYork_New[u][v]['total_flow'] += row[4]

                Graph_NewYork_New[u][v]['driving_travelor'] += Net_work_Old[start_node][end_node]['driving_travelor']
                Graph_NewYork_New[u][v]['transit_travelor'] += Net_work_Old[start_node][end_node]['transit_travelor']
                Graph_NewYork_New[u][v]['taxi_travelor'] += Net_work_Old[start_node][end_node]['taxi_travelor']
            else:
                logging.info(f"{u} and {v} is not connected", )        
    with open('Graph_NewYork_corrected_Flow.pkl', 'wb') as f:
        pickle.dump(Graph_NewYork_New, f)
    print('Graph_NewYork_corrected_Flow 保存成功！')
def get_holidays(year):
    # 获取指定年份的美国法定节假日
    us_holidays = holidays.US(years=[year])
    
    # 格式化为 'YYYY-MM-DD' 的列表
    holiday_list = [date.strftime('%Y-%m-%d') for date in sorted(us_holidays.keys())]
    
    return holiday_list
def calculate_taxi_cost():
    DB=Database("GreYelHir.db")
    year=2022
    holiday_list = get_holidays(year)
    weekdays = pd.date_range(start=f"""{year}-01-01""", end=f"""{year}-12-31""", freq='B')  # 生成所有工作日（去除周末）
    valid_days = weekdays[~weekdays.isin(pd.to_datetime(holiday_list))]  # 去除节假日
    valid_days_count = len(valid_days) 
    holiday_condition = "AND tpep_pickup_datetime NOT IN (" + ",".join([f"'{date}'" for date in holiday_list]) + ")"
    query = f"""
    SELECT 
    COUNT(*) AS total_records,
    SUM(trip_distance) AS trip_distance_sum,
    SUM(total_amount) AS total_amount_sum,
    SUM(total_amount) / NULLIF(SUM(trip_distance), 0) AS average_cost_per_mile
    From
    (SELECT tpep_pickup_datetime, tpep_dropoff_datetime, trip_distance, total_amount
    FROM YellowT
    WHERE strftime('%Y', tpep_pickup_datetime) = '{year}'
    AND strftime('%Y', tpep_dropoff_datetime) = '{year}'
    AND strftime('%w', tpep_pickup_datetime) NOT IN ('0', '6')
    AND strftime('%w', tpep_dropoff_datetime) NOT IN ('0', '6') 
    {holiday_condition}) AS subquery
    """
    chunk1 = DB.sql_to_df(query)

    holiday_condition = "AND lpep_pickup_datetime NOT IN (" + ",".join([f"'{date}'" for date in holiday_list]) + ")"
    query = f"""
    SELECT 
    COUNT(*) AS total_records,
    SUM(trip_distance) AS trip_distance_sum,
    SUM(total_amount) AS total_amount_sum,
    SUM(total_amount) / NULLIF(SUM(trip_distance), 0) AS average_cost_per_mile
    FROM
    (SELECT lpep_pickup_datetime, lpep_dropoff_datetime, trip_distance, total_amount
    FROM GreenT
    WHERE strftime('%Y', lpep_pickup_datetime) = '{year}'
    AND strftime('%Y', lpep_dropoff_datetime) = '{year}'
    AND strftime('%w', lpep_pickup_datetime) NOT IN ('0', '6')
    AND strftime('%w', lpep_dropoff_datetime) NOT IN ('0', '6') 
    {holiday_condition}) as sunquery
    """
    chunk2 = DB.sql_to_df(query)
    df_merged = pd.concat([chunk1, chunk2], ignore_index=True)
    # 计算 YellowAndGreen 的合计值
    yellow_and_green = df_merged.sum(numeric_only=True).to_frame().T

    # 纵向合并 YellowAndGreen 统计行
    df_merged = pd.concat([df_merged, yellow_and_green], ignore_index=True)
    df_merged.index = ["YellowT", "GreenT", "YellowAndGreen"]
    df_merged.loc["YellowAndGreen", "average_cost_per_mile"] = df_merged.loc["YellowAndGreen", "total_amount_sum"]/df_merged.loc["YellowAndGreen", "trip_distance_sum"]
    
    df_merged[["trip_distance_sum", "total_amount_sum"]] = df_merged[
            ["trip_distance_sum", "total_amount_sum"]
        ].applymap(lambda x: f"{x:.2e}" if isinstance(x, (int, float)) else x)
    
    print(df_merged)
    df_merged.to_csv('calulated_taxi_cost_detail.csv', index=True)
    return df_merged
def add_Taxi_Cost():
    Graph_NewYork_New=pd.read_pickle('Graph_NewYork_corrected_Flow.pkl')
    greeT=pd.read_pickle('Mean_pickup_count_Green.pkl')
    yellowT=pd.read_pickle('Mean_pickup_count_yellow.pkl')
    num_edges = Graph_NewYork_New.number_of_edges()
    counter=0
    for u, v in Graph_NewYork_New.edges:
        print(f"deal with {counter} out of {num_edges}")
        counter+=1
        greeT_cost = next(iter(greeT.loc[
            (greeT['PULocationID'] == u) & (greeT['DOLocationID'] == v), 'mean_cost'
        ]), 0)
        yellowT_cost = next(iter(yellowT.loc[
            (yellowT['PULocationID'] == u) & (yellowT['DOLocationID'] == v), 'mean_cost'
        ]), 0)
        taxi_cost=((greeT_cost+yellowT_cost)/2)
        
        Graph_NewYork_New[u][v]['taxi_cost'] = round(taxi_cost, 2)
        route_length=(Graph_NewYork_New[u][v]['route_length']/1609.34)        
        Graph_NewYork_New[u][v]['route_length']=round(route_length,2)
        dirving_cost=(route_length/24.17)*2.995
        Graph_NewYork_New[u][v]['dirving_cost']=round(dirving_cost,2)
        Graph_NewYork_New[u][v]['transit_cost']=3.3
        if taxi_cost==0:
            Graph_NewYork_New[u][v]['taxi_cost'] = Graph_NewYork_New[u][v]['route_length']*6.128
    with open('Graph_NewYork_final_version.pkl', 'wb') as f:
        pickle.dump(Graph_NewYork_New, f)

def get_edge_to_csv(graph):
    data=pd.read_pickle(graph)
    edge_data = []
    num_links = data.number_of_edges()
    print(num_links)
    for u, v, data in data.edges(data=True):
        edge_data.append({
            'start_node': u,
            'end_node': v,
            'total_flow':data['total_flow'],
            'route_length':data['route_length'],
            'freetime_dirve':data['freetime_dirve'],
            'freetime_transit':data['freetime_transit'],
            'driving_travelor':data['driving_travelor'],
            'transit_travelor':data['transit_travelor'],
            'taxi_travelor':data['taxi_travelor'],
            'taxi_cost':data['taxi_cost'],
            'dirving_cost':data['dirving_cost'],
            'transit_cost':data['transit_cost'],
            'Road_Capacity':data['Road_Capacity'],
            })
    edges_df = pd.DataFrame(edge_data)
    filename_without_ext = graph.removesuffix(".pkl")
    # 导出为 CSV 文件
    edges_df.to_csv(f'{filename_without_ext}.csv', index=False) 

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
            'VE_travel_time':float(data['freetime_dirve']),
            'PT_travel_time':float(data['freetime_transit']),
            'VE_FreeF_travel_time':float(data['freetime_dirve']),
            'PT_FreeF_travel_time':float(data['freetime_transit']),
            'VE_travelor':int(data['driving_travelor']+data['taxi_travelor']),
            'PT_travelor':int(data['transit_travelor']),       
            'VE_fixed_cost':(float(data['dirving_cost'])*float(data['driving_travelor'])+data['taxi_travelor']*data['taxi_cost'])/denominator,
            'PT_fixed_cost':float(data['transit_cost']),
            'Road_Capacity':float(data['Road_Capacity']),
            'VE_traffic_flow':0,
            'PT_traffic_flow':0,
            'VE_travle_cost':0,
            'PT_travle_cost':0,
            })
    edges_df = pd.DataFrame(edge_data)
    edges_df.to_pickle(f"network_links_data_{year}.pkl")


def process_year(year):
    """ 处理单个年份的数据，并返回分组后的列表 """
    print(f"Processing {year}...")  # 方便查看进度
    
    data = pd.read_pickle(f"Graph_NewYork_{year}.pkl")  

    group1, group2, group3, group4, group5 = [], [], [], [], []

    for u, v, data in data.edges(data=True):
        denominator = data['driving_travelor'] + data['taxi_travelor']
        denominator = 2 if denominator == 0 else denominator  

        VE_travelor = int(data['driving_travelor'] + data['taxi_travelor'])
        capacity = float(data['Road_Capacity'])
        a=VE_travelor+int(data['transit_travelor']) 
        if a>0:
            temp = {
                'Year': year,
                'Start_node': int(u),
                'End_node': int(v),
                'VE_FreeF_travel_time': float(data['freetime_dirve']),
                'PT_FreeF_travel_time': float(data['freetime_transit']),
                'Toal_Flow': int(data['total_flow']),
                'VE_travelor': VE_travelor,
                'PT_travelor': int(data['transit_travelor']),
                # 'VE_fixed_cost': (float(data['dirving_cost']) * float(data['driving_travelor']) + 
                #                   data['taxi_travelor'] * data['taxi_cost']) / denominator,
                'VE_fixed_cost': float(data['dirving_cost']),
                'PT_fixed_cost': float(data['transit_cost']),
                'Road_Capacity': capacity,
            }

            # 分类存入不同的组
            if VE_travelor <= capacity * 0.5:
                group1.append(temp)
            elif capacity * 0.5 < VE_travelor <= capacity:
                group2.append(temp)
            elif capacity < VE_travelor <= capacity * 1.5:
                group3.append(temp)
            elif capacity * 1.5 < VE_travelor <= capacity * 2:
                group4.append(temp)
            else:
                group5.append(temp)

    return group1, group2, group3, group4, group5

def Prepare_data_for_getting_parameters_multiprocessing():
    """ 使用 multiprocessing 并行处理多个年份的数据 """
    years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", 
             "2017", "2018", "2019", "2022"]
    
    # 获取 CPU 核心数
    cores=multiprocessing.cpu_count()
    print(f"CPU:{cores}")
    num_workers = min(len(years), cores)  # 不超过 CPU 核心数

    # 使用多进程池并行处理多个年份
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_year, years)

 
    all_groups = [[], [], [], [], []]

    # 合并每个年份的数据
    for i in range(len(results)):
        for j in range(5):
            all_groups[j].extend(results[i][j]) 

    # 存储数据
    for i in range(1, 6):
        pd.DataFrame(all_groups[i-1]).to_pickle(f"training_data_group{i}.pkl")
        pd.DataFrame(all_groups[i-1]).to_csv(f"training_data_group{i}.csv")
        df1=pd.DataFrame(all_groups[i-1])   
        columns_to_normalize = [ "VE_FreeF_travel_time","PT_FreeF_travel_time","Toal_Flow"]
        scaler = MinMaxScaler(feature_range=(1, 2))
        df1[columns_to_normalize] = scaler.fit_transform(df1[columns_to_normalize])
        df1.loc[:,['VE_fixed_cost','PT_fixed_cost']]=df1[['VE_fixed_cost','PT_fixed_cost']].div(df1['VE_fixed_cost']+df1['PT_fixed_cost'],axis=0)
        # df1[['VE_travelor', 'PT_travelor']] = df1[['Toal_Flow', 'VE_travelor', 'PT_travelor']].apply(lambda row: [
        #     row['Toal_Flow'] * row['VE_travelor'] / (row['VE_travelor'] + row['PT_travelor']),
        #     row['Toal_Flow'] * row['PT_travelor'] / (row['VE_travelor'] + row['PT_travelor'])
        # ], axis=1, result_type='expand')
        
        df1.to_csv(f"normalized_training_data_group{i}.csv")
        df1.to_pickle(f"normalized_training_data_group{i}.pkl")
        

    print("Data processing completed!")    


def compare_graph_edges(file_2022, file_2021):
    # 加载 2022 和 2021 的图
    with open(file_2022, "rb") as f:
        G_2022 = pickle.load(f)
    
    with open(file_2021, "rb") as f:
        G_2021 = pickle.load(f)

    # 获取所有的边集合
    edges_2022 = set(G_2022.edges())
    edges_2021 = set(G_2021.edges())

    # 找出 2022 年新增的边 (存在于 2022, 但不存在于 2021)
    added_edges = edges_2022 - edges_2021

    # 找出 2021 年被删除的边 (存在于 2021, 但不存在于 2022)
    removed_edges = edges_2021 - edges_2022

    return added_edges, removed_edges
if __name__ == "__main__":
    print('start!')
    Prepare_data_for_getting_parameters_multiprocessing()

    # calculate_taxi_cost()
    # correct_workflow()
    # add_Taxi_Cost()
    # years=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019",
    #     "2020","2021","2022"]
    # for year in years:
        # get_edge_to_csv(f"Graph_NewYork_{year}.pkl")
    # Net_work_Old=pd.read_pickle('Graph_NewYork_final_version.pkl') 
