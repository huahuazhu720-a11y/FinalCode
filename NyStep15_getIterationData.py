
"""
step 15
get iteration data
"""

import pandas as pd



def run():
    name=f"2022_for_manhattan_simulation"
    Graph_NewYork=pd.read_pickle(f'{name}.pkl') 
    nodes=Graph_NewYork.nodes   

    edge_data = []
    # this is not nesseray, just used it to identify the group type,it not used any place
    data1111 = pd.read_pickle("manhattan_edges_data04102025.pkl")
    for u, v, data in Graph_NewYork.edges(data=True):
        if data['link_index'] ==222: # exclude the link inside of manhaton
            continue
        denominator = data['driving_travelor'] + data['taxi_travelor']
        denominator = 2 if denominator == 0 else denominator 
        
        val = data1111[data1111['link_index'] == data['link_index']]['type'].iloc[0]
        type1 = 1 if val == 3 else val
        edge_data.append({
            'start_node': int(u),
            'end_node': int(v),
            'VE_travel_time':float(data['freetime_dirve']),
            'PT_travel_time':float(data['freetime_transit']),
            'VE_FreeF_travel_time':float(data['freetime_dirve']),
            'PT_FreeF_travel_time':float(data['freetime_transit']),
            'VE_travelor':int(data['driving_travelor_Manhattan']),
            'PT_travelor':int(data['transit_travelor_Manhattan']),       
            'PT_VE_total_travelor':int(data['total_flow_Manhattan']),       
            'VE_fixed_cost':(float(data['dirving_cost'])*float(data['driving_travelor'])+data['taxi_travelor']*data['taxi_cost'])/denominator,
            'PT_fixed_cost':float(data['transit_cost']),
            'Road_Capacity':float(data['Road_Capacity']),
            'fixed_flow_without_Manhattan':float(data['fixed_flow_without_Manhattan']),
            'fixed_flow_without_Manhattan_PT':float(data['fixed_flow_without_Manhattan_PT']),
            'VE_traffic_flow':0,
            'PT_traffic_flow':0,
            'VE_travle_cost':0,
            'PT_travle_cost':0,
            'type':int(type1)
            })
    edges_df = pd.DataFrame(edge_data)
    edges_df.to_pickle(f"iterationdata04102025.pkl")


if __name__ == "__main__":   
    run()
    

