"""
step4
Overlay the merged taxi zone shapefile with the census block shapefile to identify the relationship between each merged zone and the census blocks.
Result: grouped_taxi_zones.pkl, showing the mapping between merged zones and census block IDs.
"""
import geopandas as gpd
import pandas as pd
def run():
    # 读取 Taxi Zones 和 Census Blocks 数据
    taxi_zones = gpd.read_file(r"shapfile\merged_taxi_zones.shp")
    census_blocks = gpd.read_file(r"shapfile\NYcensus_blocks.shp")

    # 确保两者使用相同的坐标系
    taxi_zones = taxi_zones.to_crs(epsg=4326)
    census_blocks = census_blocks.to_crs(epsg=4326)

    # **Step 1: 先尝试用 `within` 进行匹配**
    merged_within = gpd.sjoin(taxi_zones, census_blocks, how="left", predicate="within")

    # **Step 2: 计算 Census Block 的中心点**

    census_blocks["centroid"] = census_blocks.geometry.centroid

    # **Step 3: 对 `within` 没有匹配上的 Block，使用 `centroid` 归属**
    unmatched_blocks = census_blocks[~census_blocks["geoid"].isin(merged_within["geoid"])]

    # **Step 4: 用 `centroid` 匹配**
    merged_centroid = gpd.sjoin(taxi_zones, unmatched_blocks.set_geometry("centroid"), how="left", predicate="intersects")

    # **Step 5: 合并 `within` 和 `centroid` 的结果**
    merged = pd.concat([merged_within, merged_centroid], ignore_index=True).drop_duplicates()


    # 只保留需要的列
    merged = merged[["zone_id", "borough", "geoid"]]

    # 按照 zone_id 分组，获取每个 Taxi Zone 关联的唯一 Census Block
    grouped = merged.groupby("zone_id").agg({"geoid": list, "borough": "first"}).reset_index()
    grouped.to_pickle("grouped_taxi_zones.pkl")
    # 保存为 CSV
    # grouped.to_csv("taxi_zone_to_census_block_v1.csv", index=False)

    print("Mapping saved: taxi_zone_to_census_block_v1.csv")
if __name__ == "__main__":
    run()