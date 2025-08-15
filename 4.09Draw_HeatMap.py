import matplotlib.pyplot as plt
import seaborn as sns
from database import Database
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
net_work_name=f"2022_for_manhattan_simulation.pkl"
Graph_NewYork=pd.read_pickle(net_work_name) 
edge_data = []
for u, v, data in Graph_NewYork.edges(data=True):
    row = {"u": u, "v": v}
    row.update(data)
    edge_data.append(row)

# 转换为 DataFrame
edges_df = pd.DataFrame(edge_data)

edges_df['u'] = pd.to_numeric(edges_df['u'], errors='coerce')
edges_df['v'] = pd.to_numeric(edges_df['v'], errors='coerce')

sorted_zone_ids = sorted(edges_df['u'].dropna().unique())
sorted_outzone_ids = sorted(edges_df['v'].dropna().unique())
heatmap_data = edges_df.pivot(index='u', columns='v', values='transit_travelor_Manhattan')
heatmap_data = heatmap_data.reindex(index=sorted_zone_ids, columns=sorted_outzone_ids)



# 设置画图尺寸（可选）
plt.figure(figsize=(24, 16))
# colors = ["#90EE90", "#00FF00", "#0000FF", "#FF0000"] 
# cmap = mcolors.LinearSegmentedColormap.from_list("CustomGreenBlue", colors, N=256)
# 绘制热力图
original_cmap = plt.cm.get_cmap('YlGnBu')
new_cmap = LinearSegmentedColormap.from_list("cut_YlGnBu", original_cmap(np.linspace(0.2, 1.0, 256)))

ax = sns.heatmap(heatmap_data, fmt=".0f", cmap=new_cmap)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
tick_labels =[1,10,20,30,40,50,60,70,80,90,100,110,120,130]
xtick_pos = [heatmap_data.columns.get_loc(i) for i in tick_labels]
ytick_pos = [heatmap_data.index.get_loc(i) for i in tick_labels]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(tick_labels, rotation=0, fontsize=6)
# plt.xticks(rotation=90, ha='left') 
ax.set_yticks(ytick_pos)
ax.set_yticklabels(tick_labels, fontsize=6)
# 添加标题和标签（可选）
plt.xlabel('End Node')
plt.ylabel('Start Node')
plt.show()



