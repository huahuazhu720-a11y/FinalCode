
import pandas as pd
import numpy as np


A=pd.read_csv("L_starrrrrrrrrrrr.csv")
B=pd.read_pickle("matrix.pkl")
selected_columns = B[['VE_travelor', 'PT_travelor']]
i=0
C=pd.read_pickle("solved_equilibrium.pkl")
C=C['l_1']
for index, row in selected_columns.iterrows():
    selected_columns.at[index, 'VE_travelor'] = C[i][0]
    selected_columns.at[index, 'PT_travelor'] = C[i][1]
    i += 1
print(selected_columns.head(2))
repeat_times = len(A) // len(selected_columns) + 1
B_repeated = pd.DataFrame(
    np.tile(selected_columns.values, (repeat_times, 1)),  # 按行重复 B
    columns=selected_columns.columns
).iloc[:len(A)]
D= pd.concat([A, B_repeated], axis=1)
D["distance"] = np.sqrt((D["1"] - D["VE_travelor"])**2 + (D["2"] - D["PT_travelor"])**2)
mean_distance = D.groupby("0")["distance"].mean()
mean_distance_df = pd.DataFrame(mean_distance).reset_index()
mean_distance_df.columns = ['0', 'mean']
min_row = mean_distance_df.loc[mean_distance_df['mean'].idxmin()]
print(min_row)
pioint=min_row['0']
print(pioint)
mean_distance_df.to_csv("_ssssssssssssss.csv", index=False)
delta_1 = 0.01
delta_2 = 0.01
delta_4 = 0.05
nu_vehicles_range = np.arange(0.308, 0.462 + delta_1, delta_1)
nu_transit_range = np.arange(0.246, 0.37 + delta_2, delta_2)
eta_range = np.arange(0.1, 0.8 + delta_4, delta_4)  
eta_range = np.round(eta_range, 3) 
i=1

for nu_vehicles in nu_vehicles_range:
    for nu_transit in nu_transit_range:
        for eta in eta_range:
            if i==int(pioint):
                print(f"theta: {i},nu_vehicles: {nu_vehicles}, nu_transit: {nu_transit},eta: {eta}")
                i+=1
                break
            i+=1
            
