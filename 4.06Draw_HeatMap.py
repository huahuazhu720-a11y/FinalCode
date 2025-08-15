import matplotlib.pyplot as plt
import seaborn as sns
from database import Database
import pandas as pd
import matplotlib.colors as mcolors
DB=Database("GreYelHir.db")
query = f"""
            SELECT 
                *              
            FROM 
                ZoneOutNodes
            WHERE 
                year=2022
            And 
            OutZone_id >0
            AND OutZone_id !=zone_id
        """
Out_workface_all = DB.sql_to_df(query)
Out_workface_all['zone_id'] = pd.to_numeric(Out_workface_all['zone_id'], errors='coerce')
Out_workface_all['OutZone_id'] = pd.to_numeric(Out_workface_all['OutZone_id'], errors='coerce')

sorted_zone_ids = sorted(Out_workface_all['zone_id'].dropna().unique())
sorted_outzone_ids = sorted(Out_workface_all['OutZone_id'].dropna().unique())
heatmap_data = Out_workface_all.pivot(index='zone_id', columns='OutZone_id', values='count_of_jobs')
heatmap_data = heatmap_data.reindex(index=sorted_zone_ids, columns=sorted_outzone_ids)
# 设置画图尺寸（可选）
plt.figure(figsize=(24, 16))
# colors = ["#90EE90", "#00FF00", "#0000FF", "#FF0000"] 
# cmap = mcolors.LinearSegmentedColormap.from_list("CustomGreenBlue", colors, N=256)
# 绘制热力图
ax = sns.heatmap(heatmap_data, fmt=".0f", cmap="YlGnBu")
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
tick_labels =[1,10,20,30,40,50,60,70,80,90,100,110,120,130]
xtick_pos = [heatmap_data.columns.get_loc(i)+1 for i in tick_labels]
ytick_pos = [heatmap_data.index.get_loc(i)+1 for i in tick_labels]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_pos, rotation=0, fontsize=6)
# plt.xticks(rotation=90, ha='left') 
ax.set_yticks(ytick_pos)
ax.set_yticklabels(ytick_pos, fontsize=6)
# 添加标题和标签（可选）
plt.xlabel('End Node')
plt.ylabel('Start Node')
plt.show()
plt.savefig("Heatmap.pdf", format='pdf', dpi=300, bbox_inches='tight')
Out_workface_all.to_csv("OD_2022.csv")

