import sympy as sp
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
'''
最先写的模拟模型运行的代码
'''

def weighted_avg1(group):
    return (group['VE_travelor'] * group['VE_fixed_cost']).sum() / group['VE_travelor'].sum()
def weighted_avg2(group):
    return (group['PT_travelor'] * group['PT_fixed_cost']).sum() / group['PT_travelor'].sum()
def VE_node_cost_gradient(group):
    return (group['VE_traffic_flow'].sum()-group['VE_travelor'].sum() )  / (group['VE_traffic_flow'].sum() +group['VE_traffic_flow'].count())
def PT_node_cost_gradient(group):
    return ( group['PT_traffic_flow'].sum()-group['PT_travelor'].sum())/ (group['PT_traffic_flow'].sum() +group['PT_traffic_flow'].count())

if __name__ == "__main__":
    great=False
    kappa=0.06
    VE_value_time=0.3
    PT_value_time=0.3
    step1 = 0.01
    step2 = 0.01
    step3 = 0.01
    step4 = 0.01
    step5 = 0.01
    threshold=1e-6
    df = pd.read_pickle("matrix.pkl")
    VE_travel_time_prev_loss=0
    VE_Start_node_cost_prev_loss=0
    PT_Start_node_cost_prev_loss=0
    VE_End_node_cost_prev_loss=0
    PT_End_node_cost_prev_loss=0
    # 计算start_node_VE_cost 初始值
    result = df.groupby('start_node', group_keys=False).apply(weighted_avg1).reset_index(name='weighted_VE_cost')
    result = result.sort_values(by='start_node')
    Start_node_VE_cost = dict(zip(result['start_node'], result['weighted_VE_cost']))
    df['Start_node_VE_cost'] = df['start_node'].map(Start_node_VE_cost)
    # print(Start_node_VE_cost) 

    # 计算start_node_PT_cost 初始值
    Start_node_PT_cost = {key: 3.3 for key in Start_node_VE_cost.keys()}
    df['Start_node_PT_cost'] = df['start_node'].map(Start_node_PT_cost)
    # print(Start_node_PT_cost)

    # 计算end_node_VE_cost 初始值
    result = df.groupby('end_node', group_keys=False).apply(weighted_avg1).reset_index(name='weighted_VE_cost')
    result = result.sort_values(by='end_node')
    End_node_VE_cost = dict(zip(result['end_node'], result['weighted_VE_cost']))
    df['End_node_VE_cost'] = df['end_node'].map(End_node_VE_cost)
    # print(End_node_VE_cost)

    # 计算end_node_PT_cost 初始值        
    End_node_PT_cost = {key: 3.3 for key in End_node_VE_cost.keys()}
    df['End_node_PT_cost'] = df['end_node'].map(End_node_PT_cost)
    print(df.columns)
    print(df.head())
    for i in range(1000):  
        df['VE_travle_cost'] = VE_value_time * df['VE_travel_time'] + df['VE_fixed_cost']
        df['PT_travle_cost'] = PT_value_time * df['PT_travel_time'] + df['PT_fixed_cost']
        df['VE_traffic_flow'] = np.maximum(
                0, 
                np.exp(-df['End_node_VE_cost'] - df['Start_node_VE_cost'] - df['VE_travle_cost'])
            )
        df['PT_traffic_flow'] = np.maximum(
                0, 
                np.exp(-df['End_node_PT_cost'] - df['Start_node_PT_cost'] - df['PT_travle_cost'])
            )
        df['VE_Travel_time_gradient'] =(df['VE_FreeF_travel_time']+kappa*df['VE_traffic_flow']-df['VE_travel_time'])/((kappa*(-VE_value_time*(df['VE_traffic_flow']+1)))-1)
        df['PT_Travel_time_gradient'] =0

        result = df.groupby('start_node', group_keys=False).apply(VE_node_cost_gradient).reset_index(name='temp') 
        Start_node_VE_cost_gradient = dict(zip(result['start_node'], result['temp']))   
        df['Start_node_VE_cost_gradient']=df['start_node'].map(Start_node_VE_cost_gradient)

        result = df.groupby('end_node', group_keys=False).apply(VE_node_cost_gradient).reset_index(name='temp') 
        End_node_VE_cost_gradient = dict(zip(result['end_node'], result['temp']))   
        df['End_node_VE_cost_gradient']=df['end_node'].map(End_node_VE_cost_gradient)


        result = df.groupby('start_node', group_keys=False).apply(PT_node_cost_gradient).reset_index(name='temp') 
        Start_node_PT_cost_gradient = dict(zip(result['start_node'], result['temp']))   
        df['Start_node_PT_cost_gradient']=df['start_node'].map(Start_node_PT_cost_gradient)

        result = df.groupby('end_node', group_keys=False).apply(PT_node_cost_gradient).reset_index(name='temp') 
        End_node_PT_cost_gradient = dict(zip(result['end_node'], result['temp']))   
        df['End_node_PT_cost_gradient']=df['end_node'].map(End_node_PT_cost_gradient)
        # print(df.head())
        #计算新值
        df['VE_travel_time_new']=df['VE_travel_time']-df['VE_Travel_time_gradient']*step1
        df['PT_travel_time_new']=df['PT_travel_time']-df['PT_Travel_time_gradient']

        df['Start_node_VE_cost_new']=df['Start_node_VE_cost']-df['Start_node_VE_cost_gradient']*step2
        df['Start_node_PT_cost_new']=df['Start_node_PT_cost']-df['Start_node_PT_cost_gradient']*step3
        df['End_node_VE_cost_new']=df['End_node_VE_cost']-df['End_node_VE_cost_gradient']*step4
        # print(f"End_node_VE_cost:{df['End_node_VE_cost'][1]}")
        # print(f"End_node_VE_cost_new:{df['End_node_VE_cost_new'][1]}")
      
        df['End_node_PT_cost_new']=df['End_node_PT_cost']-df['End_node_PT_cost_gradient']*step5 #0.0001
        # print(df['End_node_PT_cost_new'][1])
        # 判断梯度的变化决定是否停止迭代        
        VE_travel_time_CHANGE = np.abs(df['VE_travel_time_new'] - df['VE_travel_time']).mean()
        PT_travel_time_CHANGE = np.abs(df['PT_travel_time_new'] - df['PT_travel_time']).mean()
        VE_Start_node_cost_CHANGE = np.abs(df['Start_node_VE_cost_new']- df['Start_node_VE_cost']).mean()
        PT_Start_node_cost_CHANGE = np.abs(df['Start_node_PT_cost_new']- df['Start_node_PT_cost']).mean()
        VE_End_node_cost_CHANGE = np.abs(df['End_node_VE_cost_new']- df['End_node_VE_cost']).mean()
        PT_End_node_cost_CHANGE = np.abs(df['End_node_PT_cost_new']- df['End_node_PT_cost']).mean()
        if VE_travel_time_CHANGE <=threshold and PT_travel_time_CHANGE<=threshold and VE_Start_node_cost_CHANGE<=threshold\
            and PT_Start_node_cost_CHANGE<=threshold and VE_End_node_cost_CHANGE<=threshold and PT_End_node_cost_CHANGE<=threshold:
            print(f'great!!!!!!!!!!!!!!!!!!!!!!!!!!1```````````````````````````end at{i}```````')
            name="matrix_done_oooooooo"
            df.to_pickle(f"{name}.pkl")
            df.to_csv(f"{name}.csv", index=False)
            great=True
            break      
             
        if VE_travel_time_CHANGE > VE_travel_time_prev_loss:
            step1 *= 0.85
            VE_travel_time_prev_loss=VE_travel_time_CHANGE
        else:  # 误差下降
            step1 *= 1.05
            VE_travel_time_prev_loss=VE_travel_time_CHANGE

        if VE_Start_node_cost_CHANGE>VE_Start_node_cost_prev_loss:
            step2 *= 0.85
            VE_Start_node_cost_prev_loss=VE_Start_node_cost_CHANGE
        else:  # 误差下降
            step2 *= 1.05
            VE_Start_node_cost_prev_loss=VE_Start_node_cost_CHANGE
        if PT_Start_node_cost_CHANGE>PT_Start_node_cost_prev_loss:
            step3 *= 0.85
            PT_Start_node_cost_prev_loss=PT_Start_node_cost_CHANGE
        else:  # 误差下降
            step3 *= 1.05   
            PT_Start_node_cost_prev_loss=PT_Start_node_cost_CHANGE

        if VE_End_node_cost_CHANGE>VE_End_node_cost_prev_loss:
            step4 *= 0.85
            VE_End_node_cost_prev_loss=VE_End_node_cost_CHANGE
        else:  # 误差下降
            step4 *= 1.05       
            VE_End_node_cost_prev_loss=VE_End_node_cost_CHANGE
        
        if PT_End_node_cost_CHANGE>PT_End_node_cost_prev_loss:
            step5 *= 0.85
            PT_End_node_cost_prev_loss=PT_End_node_cost_CHANGE
        else:  # 误差下降
            step5 *= 1.05  
            PT_End_node_cost_prev_loss=PT_End_node_cost_CHANGE

        print(f"VE_travel_time_CHANGE:{VE_travel_time_CHANGE}")
        print(f"PT_travel_time_CHANGE:{PT_travel_time_CHANGE}")
        print(f"VE_Start_node_cost_CHANGE:{VE_Start_node_cost_CHANGE}")
        print(f"PT_Start_node_cost_CHANGE:{PT_Start_node_cost_CHANGE}")
        print(f"VE_End_node_cost_CHANGE:{VE_End_node_cost_CHANGE}")
        print(f"PT_End_node_cost_CHANGE:{PT_End_node_cost_CHANGE}")
        # 把新值赋给老值
        df['VE_travel_time']=df['VE_travel_time_new']
        df['PT_travel_time']=df['PT_travel_time_new']

        df['Start_node_VE_cost']=df['Start_node_VE_cost_new']
        df['Start_node_PT_cost']=df['Start_node_PT_cost_new']

        df['End_node_VE_cost']=df['End_node_VE_cost_new']
        df['End_node_PT_cost']=df['End_node_PT_cost_new']
    if not great:
        print("did not get there!")
        name="matrix_done_oooooooo"
        df.to_pickle(f"{name}.pkl")
        df.to_csv(f"{name}.csv", index=False)
    






