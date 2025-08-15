"""
step 16
simulation only the traffic flow related to manhatton
"""

import sympy as sp
import numpy as np
import pandas as pd
import warnings
from datetime import date
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
warnings.simplefilter(action='ignore', category=DeprecationWarning)
def softmax_flow(row, lamda, cost_column):
    
    # 计算起点和终点的lambda差异
    start = int(row['start_node'])
    end = int(row['end_node'])    
    # 计算每条路径的 "重要性" （cost + lambda 差异）

    exponent = lamda[start][0] - lamda[end][0] - row[cost_column]    

    exp_values = np.exp(exponent) -1
    # exp_values = max(0,(np.exp(exponent) -1))
    return exp_values
def get_exponent(row,lamda, cost_column):
    
    # 计算起点和终点的lambda差异
    start = int(row['start_node'])
    end = int(row['end_node'])    
    # 计算每条路径的 "重要性" （cost + lambda 差异）

    exponent = abs(lamda[start]) + abs(lamda[end]) - row[cost_column]    

    # exp_values = max(0,(np.exp(exponent) -1))
    return exponent
def compute_flow(df, lamda, cost_column):
    def process_group(group):
        start_node = group['start_node'].iloc[0]

        # 1. clip负值为0，归一化 cost
        costs = group[cost_column].clip(lower=0)
        total_cost = costs.sum() or 1  # 防止除以0
        norm_costs = costs / total_cost

        lamda_total = []
        for i, row in group.iterrows():
            start = int(row['start_node'])
            end = int(row['end_node'])
            lamda_total.append(lamda[start][0])
            lamda_total.append(lamda[end][0])
        lamda_total = np.sum(lamda_total) or 1
        # 2. 计算 exponent
        exponents = []
        for i, row in group.iterrows():
            start = int(row['start_node'])
            end = int(row['end_node'])
            exponent = lamda[start][0]/lamda_total - lamda[end][0]/lamda_total - norm_costs.loc[i]
            exponents.append(exponent)
        if len(exponents)>1:
            # 3. 标准化 exponent 到 [1, 2]
            min_exp = min(exponents)
            max_exp = max(exponents)
            range_exp = max_exp - min_exp or 1
            scaled_exponents = [1 + (e - min_exp) / range_exp for e in exponents]
        else:
            scaled_exponents = exponents

        # 4. 计算最终 flow 值
        return pd.Series([np.exp(e) - 1 for e in scaled_exponents], index=group.index)

    # 按 start_node 分组并计算 flow
    flow = df.groupby('start_node').apply(lambda g: process_group(g))
    return flow.sort_index()  # 确保和原 df 对齐
def update_Lambda(df,lamda):
    netGraph = df.copy()
    netGraph.loc[netGraph['end_node'].isin(end_nodes), 'end_node'] = 999
    # 构建原始有向图
    G = nx.DiGraph()    
    for row in netGraph.itertuples(index=False):
        G.add_edge(
            row.start_node,
            row.end_node,
            VE_traffic_flow=row.VE_traffic_flow,
            PT_traffic_flow=row.PT_traffic_flow,
            VE_travle_cost=row.VE_travle_cost,
            PT_travle_cost=row.PT_travle_cost
        )
    G_reverse = G.reverse()
    visited = set()
    queue = deque([999])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            if node==999:
                lamda[59][0]=0
                lamda[114][0]=0
                lamda[61][0]=0
                lamda[119][0]=0
                lamda[98][0]=0
                lamda[103][0]=0
                lamda[90][0]=0
                lamda[69][0]=0
            else:            
                out_nodes=G.neighbors(node)
                lamda_value=99999
                for neighbor in out_nodes:
                    if neighbor==999:
                        lamda_end_node=0
                    else:
                        lamda_end_node=lamda[neighbor][0]
                    VE_travle_cost = G.edges[node, neighbor]['VE_travle_cost']
                    VE_traffic_flow = G.edges[node, neighbor]['VE_traffic_flow']
                    VE_temp_value=np.log(1+VE_traffic_flow)+VE_travle_cost

                    PT_travle_cost = G.edges[node, neighbor]['PT_travle_cost']
                    PT_traffic_flow = G.edges[node, neighbor]['PT_traffic_flow']
                    PT_temp_value=np.log(1+PT_traffic_flow)+PT_travle_cost

                    min_one_from_edge=min([PT_temp_value,VE_temp_value])
                    min_lamda_value=min_one_from_edge+lamda_end_node
                    if min_lamda_value<lamda_value:
                        lamda_value=min_lamda_value
                lamda[node][0]=lamda_value
            for neighbor in G_reverse.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)
    scaler = MinMaxScaler()
    lamda[:, 0:1] = scaler.fit_transform(lamda[:, 0:1])
    return lamda
def find_need_wait(G):
    need_wait = np.zeros(num_nodes)
    for i in range(num_nodes):
        if i in G:
            need_wait[i] = G.in_degree(i)
        
    return need_wait

def normalize_group(group):
    
    # 平移到最小值为1
    ve_min=group['VE_in_exp'].min()
    PT_min=group['PT_in_exp'].min()
    all_min=min(ve_min,PT_min)
    group['VE_in_exp'] = group['VE_in_exp'] + (10 - all_min)
    group['PT_in_exp'] = group['PT_in_exp'] + (10 - all_min)
    sum_=group['VE_in_exp'].sum()+group['PT_in_exp'].sum()
    # 归一化（按组内总和）
    group['VE_in_exp'] =1+(group['VE_in_exp'] / sum_)*8
    group['PT_in_exp'] =1+ (group['PT_in_exp'] / sum_)*8
    
    return group
def global_minmax_scale_columns(df, columns, feature_range=(0, 1)):

    a, b = feature_range
    combined = df[columns].values.flatten()
    global_min = combined.min()
    global_max = combined.max()
    
    # 避免除以 0 的情况
    if global_max == global_min:
        raise ValueError("选中的列中所有值都相同，无法进行缩放。")

    def scale_series(s):
        return (s - global_min) / (global_max - global_min) * (b - a) + a

    df_scaled = df.copy()
    for col in columns:
        df_scaled[col] = scale_series(df_scaled[col])

    return df_scaled

scale = lambda x, min_val, max_val, a=0, b=1: (x - min_val) / (max_val - min_val) * (b - a) + a
def run():
    debug=False
    great=False 
    congestion=True
    VE_value_time=0.3495
    PT_value_time=0.1356
    step1 = 0.5
    step2 = 0.3
    threshold=0.0001
    end_nodes = [114,59, 61, 119, 98, 103, 90, 69]
    #读取网络图
    name=f"2022_for_manhattan_simulation"
    Graph_NewYork=pd.read_pickle(f'{name}.pkl') 
    nodes=Graph_NewYork.nodes  
    num_nodes = len(nodes) + 1  
    df = pd.read_pickle("iterationdata04102025.pkl") 
    if debug:   
        df = df[
                (df['start_node'].isin([33, 122])) &
                (df['end_node'].isin([122, 77, 126, 15]))
            ]
    df['kappa_origin'] = pd.cut(
                (df['VE_travelor']+df['fixed_flow_without_Manhattan'])/df['Road_Capacity'], 
                bins=[-float('inf'), 0.5, 1, 1.5, 2,float('inf')], 
                labels=[0.0497, 0.1082, 0.1667, 0.2252,0.2837]  
            ).astype(float)


    VE_travel_time_prev_loss=0
    lambda_prev_loss=0
    norm_prev_loss=1
    
    if congestion:
        df["congestion_price"]=15
    start_nodes = df[~df['start_node'].isin(df['end_node'])]['start_node'].unique().tolist()
    print(start_nodes)

    # 初始化 lamda
    lamda = np.zeros((131, 3)) 
    # # 初始化flow
    # for index, value in enumerate(lamda):
    #     if index == 0 or index == 11 or index in end_nodes:
    #         value[0]=0
    #         continue
    #     min_dist = float('inf')
    #     for target in end_nodes:
    #         try:
    #             dist = nx.shortest_path_length(Graph_NewYork, source=index, target=target, weight='route_length')
    #             if dist < min_dist:
    #                 min_dist = dist
    #         except nx.NetworkXNoPath:
    #             continue          
    #     value[0]=min_dist
    # scaler = MinMaxScaler(feature_range=(1, 3))
    # lamda[:, 0] = scaler.fit_transform(lamda[:, 0].reshape(-1, 1)).flatten()

    df['VE_traffic_flow'] = df['VE_travelor'].astype(float)
    df['PT_traffic_flow'] =df['PT_travelor'].astype(float)
    
    
    # 获取标准化的od_demand list
    Original_OD_demand = np.zeros(num_nodes)  
    for node in nodes:
        Original_OD_demand[node] = nodes[node].get('Out_traveler_Manhattan', 0) 
    # Original_OD_demand=Original_OD_demand/12
    scaler = MinMaxScaler()
    scaled_demand = scaler.fit_transform(Original_OD_demand.reshape(-1, 1))
    # scaled_demand = scaled_demand.ravel() 
    scaled_demand = Original_OD_demand
    # 启动迭代
    if debug:
        print(f"step000 VE_traffic_flow:{df[['start_node', 'end_node', 'VE_traffic_flow']].to_numpy().tolist()}")
        print(f"step000 PT_traffic_flow:{df[['start_node', 'end_node', 'PT_traffic_flow']].to_numpy().tolist()}")
    for i in range(2500): 
        if congestion:             
            df['VE_travle_cost'] = VE_value_time * df['VE_travel_time'] + df['VE_fixed_cost']+df["congestion_price"]
        else:
            df['VE_travle_cost'] = VE_value_time * df['VE_travel_time'] + df['VE_fixed_cost']
        df['PT_travle_cost'] = PT_value_time * df['PT_travel_time'] + df['PT_fixed_cost']
        if debug:
            print(f"step{i} VE_travle_cost:{df[['start_node', 'end_node', 'VE_travle_cost']].to_numpy().tolist()}")
            print(f"step{i} PT_travle_cost:{df[['start_node', 'end_node', 'PT_travle_cost']].to_numpy().tolist()}")
        scaler = MinMaxScaler(feature_range=(0, 10))
        normalize_lamda = scaler.fit_transform(lamda[:, 0].reshape(-1, 1)).flatten()      
        # normalize_lamda = lamda[:, 0]  
        
        df['VE_in_exp']=df.apply(lambda row:get_exponent(row,normalize_lamda,'VE_travle_cost'),axis=1)
        df['PT_in_exp']=df.apply(lambda row:get_exponent(row,normalize_lamda,'PT_travle_cost'),axis=1)
        
        df = df.groupby('start_node', group_keys=False).apply(normalize_group)

        # combined = df[['VE_in_exp', 'PT_in_exp']].values.flatten()
        # global_min = combined.min()
        # global_max = combined.max()
        # df['VE_in_exp'] = scale(df['VE_in_exp'], global_min, global_max, 1, 9)
        # df['PT_in_exp'] = scale(df['PT_in_exp'], global_min, global_max, 1, 9)

        if debug:             
            print(f"step{i} VE_in_exp:{df[['start_node', 'end_node', 'VE_in_exp']].to_numpy().tolist()}")
            print(f"step{i} PT_in_exp:{df[['start_node', 'end_node', 'PT_in_exp']].to_numpy().tolist()}")
        
        # df = df.groupby('start_node', group_keys=False).apply(normalize_group)
   
        # lamda=update_Lambda(df,lamda)
        # 更新VE_traffic_flow    
        df['VE_traffic_flow'] =np.exp(df['VE_in_exp'])-1       
        # 更新PT_traffic_flow       
        df['PT_traffic_flow']=np.exp(df['PT_in_exp']) -1  

        
        
        # mask = (df['start_node'] == 33) & (df['end_node']==122)  
        # selected = df[mask]    
        # A = selected['VE_traffic_flow'].sum() + selected['PT_traffic_flow'].sum()                  
        # df.loc[mask, 'VE_traffic_flow'] = df.loc[mask, 'VE_traffic_flow'] *5068 /A
        # df.loc[mask, 'PT_traffic_flow'] = df.loc[mask, 'PT_traffic_flow'] *5068 /A
        # out_nodes=[77,126,15]
        # mask = (df['start_node'] == 122) & (df['end_node'].isin(out_nodes))  
        # selected = df[mask]    
        # A = selected['VE_traffic_flow'].sum() + selected['PT_traffic_flow'].sum()                  
        # df.loc[mask, 'VE_traffic_flow'] = df.loc[mask, 'VE_traffic_flow'] *17797 /A
        # df.loc[mask, 'PT_traffic_flow'] = df.loc[mask, 'PT_traffic_flow'] *17797 /A

        if debug:
            print(f"step{i} VE_traffic_flow:{df[['start_node', 'end_node', 'VE_traffic_flow']].to_numpy().tolist()}")
            print(f"step{i} PT_traffic_flow:{df[['start_node', 'end_node', 'PT_traffic_flow']].to_numpy().tolist()}") 
        # 更新lambda
        for index, value in enumerate(lamda):
            if index==0 or index==11 or index in end_nodes[1:]:
                continue
            if debug:
                if index not in [33,122,15,126,77]:
                    continue
            a=index in df['start_node'].tolist()
            b=index not in df['end_node'].tolist()

            if a and b:
                filtered_row = df.loc[df['start_node'] == index].iloc[0]
                total_flow=filtered_row['VE_traffic_flow'] +filtered_row['PT_traffic_flow']
                # demand_on_node=nodes.get(index, {}).get('Out_traveler_Manhattan', 0)*50000/958283
                demand_on_node=scaled_demand[index]

                # gradient =(total_flow-demand_on_node) /(total_flow+2+threshold)
                # gradient = np.clip(gradient, -10, 10)
                gradient=total_flow-demand_on_node           
                
                value[1]-=gradient*step1
                value[2]=total_flow-demand_on_node
                if debug:
                    print(f"step{i} lambda on node {index}:{value[1]}") 
            elif index in end_nodes:
                # flow_sum=df.loc[df['end_node'].isin(end_nodes), 'VE_traffic_flow'].sum()+df.loc[df['end_node'].isin(end_nodes), 'PT_traffic_flow'].sum()
                # Numerator= 958283-flow_sum
                # Denominator=flow_sum+16
                # gradient=Numerator/Denominator
                value[1]=0          
                lamda[59][1]=value[1]
                lamda[61][1]=value[1]
                lamda[119][1]=value[1]
                lamda[98][1]=value[1]
                lamda[103][1]=value[1]
                lamda[90][1]=value[1]
                lamda[69][1]=value[1]
                value[2]=0          
                lamda[59][2]=value[2]
                lamda[61][2]=value[2]
                lamda[119][2]=value[2]
                lamda[98][2]=value[2]
                lamda[103][2]=value[2]
                lamda[90][2]=value[2]
                lamda[69][2]=value[2]
            else:
                count = df[(df['end_node'] == index) | (df['start_node'] == index)].shape[0]
                flow_out_sum=df.loc[df['start_node']==index, 'VE_traffic_flow'].sum()+df.loc[df['start_node']==index, 'PT_traffic_flow'].sum()
                flow_in_sum=df.loc[df['end_node']==index, 'VE_traffic_flow'].sum()+df.loc[df['end_node']==index, 'PT_traffic_flow'].sum()
                # demand_on_node=nodes.get(index, {}).get('Out_traveler_Manhattan', 0)*50000/958283
                demand_on_node=scaled_demand[index]
                Numerator= flow_out_sum-flow_in_sum-demand_on_node

                Denominator=flow_out_sum+flow_in_sum+count*2+threshold
                # gradient=Numerator /Denominator
                # gradient = np.clip(gradient, -10, 10)
                gradient=Numerator
                value[1]-=gradient*step1 
                value[2]=Numerator  
                if debug:
                    print(f"step{i} lambda on node {index}:{value[1]}")    
        # 按总共的flow数量来0-1标准化flow，因为是整个网络图level的
        # flow_scaler=MinMaxScaler()
        # df[['VE_traffic_flow', 'PT_traffic_flow']] = flow_scaler.fit_transform(df[['VE_traffic_flow', 'PT_traffic_flow']])

        
        # mask = df['end_node'].isin(end_nodes)
        # total_flow = df.loc[mask, 'VE_traffic_flow'].sum() + df.loc[mask, 'PT_traffic_flow'].sum()
        # print(f"8links total flow:{total_flow}")
        # 更新 VE_travel_time

        # Max_=df['VE_traffic_flow'].max()
        # scale_factor_VE_last = Max_ / 11000
        # df['VE_traffic_flow']=df['VE_traffic_flow']/scale_factor_VE_last

        # Max_=df['PT_traffic_flow'].max()
        # scale_factor_PT_last = Max_ / 11000
        # df['PT_traffic_flow']=df['PT_traffic_flow']/scale_factor_PT_last
        T=df['VE_traffic_flow'].sum()+df['PT_traffic_flow'].sum()
        # df = global_minmax_scale_columns(df, ['VE_traffic_flow', 'PT_traffic_flow'], feature_range=(0, 1))
        # df['VE_traffic_flow'] =df['VE_traffic_flow']*3800901 /T    26603
        df['VE_traffic_flow'] =df['VE_traffic_flow']*3800901 /T   
              
        df['PT_traffic_flow']=df['PT_traffic_flow']*3800901/T
        df['kappa'] = pd.cut(
                (df['VE_traffic_flow']+df['fixed_flow_without_Manhattan']) / df['Road_Capacity'], 
                bins=[-float('inf'), 0.5, 1, 1.5, 2,float('inf')], 
                labels=[0.0497, 0.1082, 0.1667, 0.2252,0.2837]  
            ).astype(float)
        if congestion:
            df['VE_Travel_time_gradient'] = (df['VE_travel_time']-df['kappa']*(df['VE_traffic_flow']+df['fixed_flow_without_Manhattan']) - df['VE_FreeF_travel_time'])/(1+df['kappa']*VE_value_time*(1+df['VE_traffic_flow']))
        else:
            df['VE_Travel_time_gradient'] = (df['VE_travel_time']-df['kappa']*(df['VE_traffic_flow']+df['fixed_flow_without_Manhattan']) - df['VE_FreeF_travel_time'])/(1+2*df['kappa']*VE_value_time*(1+df['VE_traffic_flow']))
        df['VE_travel_time_new']=df['VE_travel_time']-df['VE_Travel_time_gradient']*step2
        if debug:
            print(f"step{i} VE_travel_time_new:{df[['start_node', 'end_node', 'VE_travel_time_new']].to_numpy().tolist()}") 
        VE_travel_time_CHANGE = np.abs(df['VE_travel_time_new'] - df['VE_travel_time']).mean()  
        lambda_change = np.mean(np.abs(lamda[:, 1] - lamda[:, 0]))
        third_column = lamda[:, 2] 
        gradient= np.linalg.norm(third_column)/130
        # print(gradient)
        # mask_all = (df['end_node'].isin(end_nodes))
        # selected = df[mask_all]
        # a=selected['VE_traffic_flow'].sum()+selected['PT_traffic_flow'].sum()
        # print(a)
        # a=selected['VE_travelor'].sum()+selected['PT_travelor'].sum()
        # print(a)
        df["VE_Optimal_cost"]=VE_value_time* ((df['VE_traffic_flow']+ df['fixed_flow_without_Manhattan'])*df['kappa'] +df['VE_FreeF_travel_time'])+df['VE_fixed_cost']+(1+df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])*np.log((1+df['VE_traffic_flow'])+df['fixed_flow_without_Manhattan'])-(df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])
        df["PT_Optimal_cost"]=PT_value_time* (df['PT_FreeF_travel_time'])+df['PT_fixed_cost']+(1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])*np.log((1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT']))-(df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])
        totalcost=df["VE_Optimal_cost"].sum()+df["PT_Optimal_cost"].sum()
        # print(totalcost)
        


        
        if VE_travel_time_CHANGE < threshold and lambda_change< threshold:
            print(f'great!!!!!!!!!!!!!!!!!!!!!!!!!!1```````````````````````````end at{i}```````')
            # 结合真实的网络图来还原真的的flow
            netGraph = df.copy()
            # 构建原始有向图
            G = nx.DiGraph()    
            for row in netGraph.itertuples(index=False):
                G.add_edge(
                    row.start_node,
                    row.end_node
                )
            G_reverse = G.reverse()
            visited = set()
            node_list = list(G.nodes)
            queue = deque(node_list)
            accumulated_OD_demand = np.zeros(num_nodes)

            need_wait=find_need_wait(G)
            while queue:
                node = queue.popleft()            
                if not need_wait[node]:            
                    total_OD_demand=Original_OD_demand[node] + accumulated_OD_demand[node]     
                    out_nodes = list(G.neighbors(node))
                    visited.add(node)
                    if len(out_nodes)==1:
                        out_node=out_nodes[0]
                        accumulated_OD_demand[out_node] += total_OD_demand
                        # 第一步：筛选出符合条件的行
                        mask = (df['start_node'] == node) & (df['end_node']==out_node)
                        selected = df[mask]
                        # 第二步：计算 VE_traffic_flow 和 PT_traffic_flow 的和
                        A = selected['VE_traffic_flow'].sum() + selected['PT_traffic_flow'].sum()
                        if A !=0:
                            # 第三步：更新这两列的值
                            df.loc[mask, 'VE_traffic_flow'] = df.loc[mask, 'VE_traffic_flow'] * total_OD_demand / A
                            df.loc[mask, 'PT_traffic_flow'] = df.loc[mask, 'PT_traffic_flow'] * total_OD_demand / A
                        else:
                            df.loc[mask, 'VE_traffic_flow'] = 0
                            df.loc[mask, 'PT_traffic_flow'] = 0    
                        need_wait[out_node]-=1
                    else:
                        mask_all = (df['start_node'] == node) & (df['end_node'].isin(out_nodes))
                        selected = df[mask_all]
                        # 第二步：计算 VE_traffic_flow 和 PT_traffic_flow 的和
                        A = selected['VE_traffic_flow'].sum() + selected['PT_traffic_flow'].sum()
                        for out_node in out_nodes:
                            mask = (df['start_node'] == node) & (df['end_node']==out_node)                        
                            df.loc[mask, 'VE_traffic_flow'] = df.loc[mask, 'VE_traffic_flow'] * total_OD_demand / A
                            df.loc[mask, 'PT_traffic_flow'] = df.loc[mask, 'PT_traffic_flow'] * total_OD_demand / A
                            ve = float(df.loc[mask, 'VE_traffic_flow'].iloc[0])
                            pt = float(df.loc[mask, 'PT_traffic_flow'].iloc[0])
                            accumulated_OD_demand[out_node] += ve + pt
                            need_wait[out_node]-=1
                    # for neighbor in G.neighbors(node):
                    #     if neighbor not in visited:
                    #         queue.append(neighbor)
                else:
                    queue.append(node)
            df["VE_Optimal_cost"]=VE_value_time* ((df['VE_traffic_flow']+ df['fixed_flow_without_Manhattan'])*df['kappa'] +df['VE_FreeF_travel_time'])+df['VE_fixed_cost']+(1+df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])*np.log((1+df['VE_traffic_flow'])+df['fixed_flow_without_Manhattan'])-(df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])
            df["PT_Optimal_cost"]=PT_value_time* (df['PT_FreeF_travel_time'])+df['PT_fixed_cost']+(1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])*np.log((1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT']))-(df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])
            df["VE_Original_cost"]=VE_value_time* ((df['VE_travelor']+ df['fixed_flow_without_Manhattan'])*df['kappa_origin'] +df['VE_FreeF_travel_time'])+df['VE_fixed_cost']+(1+df['VE_travelor']+df['fixed_flow_without_Manhattan'])*np.log((1+df['VE_travelor'])+df['fixed_flow_without_Manhattan'])-(df['VE_travelor']+df['fixed_flow_without_Manhattan'])
            df["PT_Original_cost"]=PT_value_time* (df['PT_FreeF_travel_time'])+df['PT_fixed_cost']+(1+df['PT_travelor']+df['fixed_flow_without_Manhattan_PT'])*np.log((1+df['PT_travelor']+df['fixed_flow_without_Manhattan_PT']))-(df['PT_travelor']+df['fixed_flow_without_Manhattan_PT'])
            df["VE_perturbation_new"]=(1+df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])*np.log((1+df['VE_traffic_flow'])+df['fixed_flow_without_Manhattan'])-(df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])
            df["PT_perturbation_new"]=(1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])*np.log((1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT']))-(df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])  
            df["VE_perturbation_origin"]=(1+df['VE_travelor']+df['fixed_flow_without_Manhattan'])*np.log((1+df['VE_travelor'])+df['fixed_flow_without_Manhattan'])-(df['VE_travelor']+df['fixed_flow_without_Manhattan'])
            df["PT_perturbation_origin"]=(1+df['PT_travelor']+df['fixed_flow_without_Manhattan_PT'])*np.log((1+df['PT_travelor']+df['fixed_flow_without_Manhattan_PT']))-(df['PT_travelor']+df['fixed_flow_without_Manhattan_PT'])
           
            # set flows back to the normal scale
            # df['VE_travelor']=df['VE_travelor']*scale_factor_max_VE
            # df['PT_travelor']=df['PT_travelor']*scale_factor_max_PT            
            # df['VE_traffic_flow']=df['VE_traffic_flow']*scale_factor_VE_last
            # df['PT_traffic_flow']=df['PT_traffic_flow']*scale_factor_PT_last
            
            
            df['VE_travel_time_original']=(df['VE_travelor']+ df['fixed_flow_without_Manhattan'])*df['kappa_origin'] +df['VE_FreeF_travel_time'] 
            df['VE_travel_time']=(df['VE_traffic_flow']+ df['fixed_flow_without_Manhattan'])*df['kappa'] +df['VE_FreeF_travel_time']
            
            mask_all = (df['end_node'].isin(end_nodes))
            selected = df[mask_all]
            a=selected['VE_traffic_flow'].sum()+selected['PT_traffic_flow'].sum()
            print(a)
            a=selected['VE_travelor'].sum()+selected['PT_travelor'].sum()
            print(a)
            # clean temp field
            df = df.drop(columns=['VE_travel_time_new','VE_Travel_time_gradient','type'])

            new_cost=df['VE_Optimal_cost'].sum()+df['PT_Optimal_cost'].sum()
            origin_cost=df['VE_Original_cost'].sum()+df['PT_Original_cost'].sum()
            
            print(f"Optimal system cost: PT_cost_new+VE_cost_new:{new_cost}")

            print(f"Original system cost: PT_cost_origin+VE_cost_origin : {origin_cost}")
            
            print(f"all links total flow:{df['VE_traffic_flow'].sum()}")

            if new_cost>origin_cost:
                print(f"Optimal system cost larger than Original system cost new_cost-origin_cost={new_cost-origin_cost}")
            else:
                print(f"Optimal system cost less than Original system cost,new_cost-origin_cost=  {new_cost-origin_cost}")            
        
            great=True
            lamda = pd.DataFrame(lamda)
            lamda.to_csv(f"solved_lamda_manhattan_convergence{date.today()}.csv",index=False)      
            df.to_csv(f"solved_flow_manhattan_convergence{date.today()}.csv", index=False)
            break       
             
        # if gradient -norm_prev_loss < 0 and ((norm_prev_loss-gradient)/norm_prev_loss)<0.005:
        #     step1 *= 1.05      
        # elif gradient -norm_prev_loss > 0 :  
        #     step1 *= 0.95
        # norm_prev_loss=gradient

        if VE_travel_time_CHANGE > VE_travel_time_prev_loss:
            step2 *= 0.85
            VE_travel_time_prev_loss=VE_travel_time_CHANGE
        else:  # 误差下降
            step2 *= 1.05
            VE_travel_time_prev_loss=VE_travel_time_CHANGE
        if lambda_change>lambda_prev_loss:
            step1 *= 0.85
            lambda_prev_loss=lambda_change
        else:  # 误差下降
            step1 *= 1.05
            lambda_prev_loss=lambda_change
        

        

        # print(f"VE_travel_time_CHANGE``````````````````````:{VE_travel_time_CHANGE}")
        # print(f"lamda_CHANGE:{VE_travel_time_CHANGE}")

        # 把新值赋给老值
        df['VE_travel_time']=df['VE_travel_time_new']
        lamda[:, 0] = lamda[:, 1]

    if not great:
        print("did not get there!")
        lamda = pd.DataFrame(lamda)
        lamda.to_csv(f"solved_lamda_manhattan_not_convergence_{date.today()}.csv",index=False)

        df.to_csv(f"solved_flow_manhattan_not_convergence_{date.today()}.csv", index=False)
    




if __name__ == "__main__":
    run()

