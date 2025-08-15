import geopandas as gpd
import matplotlib.pyplot as plt
# 读取 merged_taxi_zones.shp
shp_path = r"shapfile\merged_taxi_zones.shp"
gdf = gpd.read_file(shp_path)

# 需要合并的 zone_id 列表
merge_zone_ids = [114, 61, 119, 98, 59, 103, 90, 69]

# 创建一个新列 'new_zone_id'，默认等于原始 zone_id
gdf["new_zone_id"] = gdf["zone_id"]

# 将待合并的区域标记为 "ManhattanCBD"
gdf.loc[gdf["zone_id"].isin(merge_zone_ids), "new_zone_id"] = "ManhattanCBD"

# 按 new_zone_id 进行合并
gdf_merged = gdf.dissolve(by="new_zone_id").reset_index()

# 重新分配 zone_id：
# - 如果是合并后的 "ManhattanCBD"，就赋一个新的 ID，比如 999
# - 其他区域保持原 zone_id 不变
gdf_merged["zone_id"] = gdf_merged["new_zone_id"].apply(lambda x: 999 if x == "ManhattanCBD" else int(x))

# 选择需要的列
gdf_merged = gdf_merged[["zone_id", "geometry"]]

# 保存新的 shapefile
output_shp_path = r"shapfile\ManhattanCBD.shp"
gdf_merged.to_file(output_shp_path, driver="ESRI Shapefile")

fig, ax = plt.subplots(figsize=(10, 10))
gdf_merged.boundary.plot(ax=ax, color="black", linewidth=1)

# Add zone numbers at centroids
for idx, row in gdf_merged.iterrows():
    centroid = row.geometry.centroid
    ax.text(centroid.x, centroid.y, str(row["zone_id"]), fontsize=8, ha='center', va='center', color="black")

plt.title("Merged Taxi Zones with Number Labels (Same Boroughs)")
plt.axis("off")
plt.show()
print("合并完成，新文件已保存为:", output_shp_path)