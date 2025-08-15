"""
step 17
simulation without conjestion pricing
"""
import sympy as sp
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
warnings.simplefilter(action='ignore', category=DeprecationWarning)
def softmax_flow(row, lamda, cost_column):
    # 计算起点和终点的lambda差异
    start = int(row['start_node'])
    end = int(row['end_node'])    
    # 计算每条路径的 "重要性" （cost + lambda 差异）
    exponent = lamda[start][0] - lamda[end][0] - row[cost_column]    
    # softmax 归一化
    exp_values = np.exp(exponent)    
    return exp_values

def run():
    great=False 
    VE_value_time=0.3495
    PT_value_time=0.1356
    step1 = 0.05
    step2 = 0.05

    threshold=1e-6
    df = pd.read_pickle("iterationdata04102025.pkl")
    df['VE_traffic_flow'] = df['VE_traffic_flow'].astype(float)
    df['PT_traffic_flow'] = df['PT_traffic_flow'].astype(float)
    VE_travel_time_prev_loss=0
    lambda_prev_loss=0
    df['VE_fixed_cost']=9+df['VE_fixed_cost']
    lamda=[[-1, 0.0, 0.0]]*131  #0号位置不用
    lamda = np.array(lamda).reshape(131, 3) 
    name=f"2022_for_manhattan_simulation"
    Graph_NewYork=pd.read_pickle(f'{name}.pkl') 
    nodes=Graph_NewYork.nodes  
    num_nodes = len(nodes) + 1  # 防止编号不连续可以用 len(nodes) 替代
    # 初始化 list
    out_traveler_list = [0] * num_nodes

    # 遍历所有节点，填入对应的值
    for node in nodes:
        out_traveler_list[node] = nodes[node].get('Out_traveler_Manhattan', 0)
    X = np.array(out_traveler_list).reshape(-1, 1)

    # 标准化
    scaler = MinMaxScaler()
    od_demand = scaler.fit_transform(X)
    od_demand_list = od_demand.flatten().tolist()
    od_demand_list = np.exp(od_demand_list)
    # od_demand_list = X.flatten().tolist()   # 不标准化demand

    end_nodes = [98, 114, 61, 103, 90, 69]
    end_nodes = [114,59, 61, 119, 98, 103, 90, 69]
    for index, value in enumerate(lamda):
        if index == 0 or index == 11 or index in end_nodes:
            value[0]=0
            continue
        min_dist = float('inf')
        for target in end_nodes:
            try:
                dist = nx.shortest_path_length(Graph_NewYork, source=index, target=target, weight='route_length')
                if dist < min_dist:
                    min_dist = dist
            except nx.NetworkXNoPath:
                continue          
        value[0]=min_dist
    scaler = MinMaxScaler()
    lamda[:, 0] = scaler.fit_transform(lamda[:, 0].reshape(-1, 1)).flatten()

    for i in range(500):       
            
        df['VE_travle_cost'] = VE_value_time * df['VE_travel_time'] + df['VE_fixed_cost']
        df['PT_travle_cost'] = PT_value_time * df['PT_travel_time'] + df['PT_fixed_cost']
        scaler = MinMaxScaler()
        df[['VE_travle_cost', 'PT_travle_cost']] = scaler.fit_transform(df[['VE_travle_cost', 'PT_travle_cost']])
   
        df['VE_traffic_flow'] = np.maximum(
            0,
            df.apply(lambda row: softmax_flow(row, lamda, 'VE_travle_cost'), axis=1)
        )
        T1 = df['VE_traffic_flow'].sum()

        # 将 VE_traffic_flow 列的每个值除以 T
        df['VE_traffic_flow'] = df['VE_traffic_flow'] / T1
       
        df['PT_traffic_flow']=np.maximum(
            0,
            df.apply(lambda row: softmax_flow(row, lamda, 'PT_travle_cost'), axis=1)
        )
        T2 = df['PT_traffic_flow'].sum()
        # 将 PT_traffic_flow 列的每个值除以 T
        df['PT_traffic_flow'] = df['PT_traffic_flow'] / T2
        # 更新lambda
        for index, value in enumerate(lamda):
            if index==0 or index==11 or index in end_nodes[1:]:
                continue
            a=index in df['start_node'].tolist()
            b=index not in df['end_node'].tolist()

            if a and b:
                filtered_row = df.loc[df['start_node'] == index].iloc[0]
                total_flow=filtered_row['VE_traffic_flow'] +filtered_row['PT_traffic_flow']
                # demand_on_node=nodes.get(index, {}).get('Out_traveler_Manhattan', 0)*50000/958283
                demand_on_node=od_demand_list[index]

                gradient =(total_flow-demand_on_node)    /(total_flow+2+threshold)
                gradient = np.clip(gradient, -10, 10)

                # gradient=total_flow-demand_on_node
                
                
                value[1]-=gradient*step1
                value[2]=total_flow-demand_on_node
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
                demand_on_node=od_demand_list[index]
                Numerator= flow_out_sum-flow_in_sum-demand_on_node

                Denominator=flow_out_sum+flow_in_sum+count*2+threshold
                gradient=Numerator /Denominator
                gradient = np.clip(gradient, -10, 10)

                # gradient=Numerator


                value[1]-=gradient*step1  
                value[2]=Numerator
            
        




        # 更新 VE_travel_time
        total_sum = df['VE_traffic_flow'].sum() + df['PT_traffic_flow'].sum()
        df['kappa'] = pd.cut(
                (df['VE_traffic_flow']*3800901/total_sum+df['fixed_flow_without_Manhattan']) / df['Road_Capacity'], 
                bins=[-float('inf'), 0.5, 1, 1.5, 2,float('inf')], 
                labels=[0.0497, 0.1082, 0.1667, 0.2252,0.2837]  
            ).astype(float)

        df['VE_Travel_time_gradient'] = (df['VE_travel_time']-df['kappa']*(df['VE_traffic_flow']+df['fixed_flow_without_Manhattan']) - df['VE_FreeF_travel_time'])/(1+df['kappa']*VE_value_time*(1+df['VE_traffic_flow']))
        df['VE_travel_time_new']=df['VE_travel_time']-df['VE_Travel_time_gradient']*step2

        # 计算lambda和VE traveltime的差距    
        lambda_change = np.mean(np.abs(lamda[:, 1] - lamda[:, 0]))
        VE_travel_time_CHANGE = np.abs(df['VE_travel_time_new'] - df['VE_travel_time']).mean()    
        third_column = lamda[1:, 2]
 
        result = norm = np.linalg.norm(third_column)

        if VE_travel_time_CHANGE <=threshold and VE_travel_time_CHANGE<=threshold:
            print(f'great!!!!!!!!!!!!!!!!!!!!!!!!!!1```````````````````````````end at{i}```````')
            lamda = pd.DataFrame(lamda)
            T=df['PT_traffic_flow'].sum()+df['VE_traffic_flow'].sum()
            df['PT_traffic_flow']=3800901*df['PT_traffic_flow']/T
            df['VE_traffic_flow']=3800901*df['VE_traffic_flow']/T
            df["VE_cost_linK"]=VE_value_time* ((df['VE_traffic_flow']+ df['fixed_flow_without_Manhattan'])*df['kappa'] +df['VE_FreeF_travel_time'])+df['VE_fixed_cost']+(1+df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])*np.log((1+df['VE_traffic_flow'])+df['fixed_flow_without_Manhattan'])-(df['VE_traffic_flow']+df['fixed_flow_without_Manhattan'])
            df["PT_cost_linK"]=PT_value_time* (df['PT_FreeF_travel_time'])+df['PT_fixed_cost']+(1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])*np.log((1+df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT']))-(df['PT_traffic_flow']+df['fixed_flow_without_Manhattan_PT'])
            
            df['kappa_origin'] = pd.cut(
                (df['VE_travelor']+df['fixed_flow_without_Manhattan'])/df['Road_Capacity'], 
                bins=[-float('inf'), 0.5, 1, 1.5, 2,float('inf')], 
                labels=[0.0497, 0.1082, 0.1667, 0.2252,0.2837]  
            ).astype(float)
            df["VE_cost_linK_origin"]=VE_value_time* ((df['VE_travelor']+ df['fixed_flow_without_Manhattan'])*df['kappa_origin'] +df['VE_FreeF_travel_time'])+df['VE_fixed_cost']+(1+df['VE_travelor']+df['fixed_flow_without_Manhattan'])*np.log((1+df['VE_travelor'])+df['fixed_flow_without_Manhattan'])-(df['VE_travelor']+df['fixed_flow_without_Manhattan'])
            df["PT_cost_linK_origin"]=PT_value_time* (df['PT_FreeF_travel_time'])+df['PT_fixed_cost']+(1+df['PT_travelor']+df['fixed_flow_without_Manhattan_PT'])*np.log((1+df['PT_travelor']+df['fixed_flow_without_Manhattan_PT']))-(df['PT_travelor']+df['fixed_flow_without_Manhattan_PT'])
            new_cost=df['VE_cost_linK'].sum()+df['PT_cost_linK'].sum()
            origin_cost=df['VE_cost_linK_origin'].sum()+df['PT_cost_linK_origin'].sum()
            print(f"sum of VE_cost_new:{df['VE_cost_linK'].sum()}")
            print(f"sum of PT_cost_new:{df['PT_cost_linK'].sum()}")
            print(f"sum of PT_cost_new+VE_cost_new:{new_cost}")
            print(f"sum of VE_cost_origin:{df['VE_cost_linK_origin'].sum()}")
            print(f"sum of PT_cost_origin:{df['PT_cost_linK_origin'].sum()}")
            print(f"sum of PT_cost_origin+VE_cost_origin : {origin_cost}")
            
            if new_cost>origin_cost:
                print(f"new cost bigger than origin cost new_cost-origin_cost={new_cost-origin_cost}")
            else:
                print(f"new cost less than origin cost,new_cost-origin_cost=  {new_cost-origin_cost}")
            lamda.to_csv("solved_lamda_manhattan_convergence_congestionP.csv",index=False)
            df.to_pickle("solved_flow_manhattan_convergence_congestionP.pkl")
            df.to_csv("solved_flow_manhattan_convergence_congestionP.csv", index=False)
            great=True
            break       
             
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
        # if lambda_change<=1e-2 and result > 1:
        #     step1=1
        #     step2=1

        

        print(f"VE_travel_time_CHANGE:{VE_travel_time_CHANGE}")
        print(f"lambda_change:{lambda_change}")
        print(f"gradient:{result}")

        # 把新值赋给老值
        df['VE_travel_time']=df['VE_travel_time_new']
        lamda[:, 0] = lamda[:, 1]

    if not great:
        print("did not get there!")
        lamda = pd.DataFrame(lamda)
        lamda.to_csv("solved_lamda_manhattan_not_convergence_congestionP.csv",index=False)
        df.to_pickle("solved_flow_manhattan_not_convergence_congestionP.pkl")
        df.to_csv("solved_flow_manhattan_not_convergence_congestionP.csv", index=False)

if __name__ == "__main__":
    run()
    
    






