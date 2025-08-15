"""
step6 find adjacent nodes—only neighboring areas with a connecting path can be linked.
Result: adjacent_districts, a dictionary where each key is a node ID and the value is a list of node IDs it can connect to.
"""

import geopandas as gpd
import pickle
def run():
    # 读取 NYC Taxi Zones Shapefile
    districts = gpd.read_file("shapfile/merged_taxi_zones.shp").to_crs(epsg=4326)

    # 创建存储邻接区的字典
    adjacent_districts = {zone: set() for zone in districts["zone_id"]}

    # 计算邻接关系（允许边界有重叠）
    for idx, district in districts.iterrows():
        neighbors = districts[districts.geometry.intersects(district.geometry)]
        for _, neighbor in neighbors.iterrows():
            if neighbor["zone_id"] != district["zone_id"]:  # 排除自身
                adjacent_districts[district["zone_id"]].add(neighbor["zone_id"])
            match district["zone_id"]:
                case 109:
                    adjacent_districts[district["zone_id"]].add(3)
                case 3:
                    adjacent_districts[district["zone_id"]].add(109)
                case 8:
                    adjacent_districts[district["zone_id"]].add(32)
                case 32:
                    adjacent_districts[district["zone_id"]].add(8)
                case 23:
                    adjacent_districts[district["zone_id"]].add(94)
                case 94:
                    adjacent_districts[district["zone_id"]].add(23)
                    adjacent_districts[district["zone_id"]].add(26)
                    adjacent_districts[district["zone_id"]].discard(30)
                case 108:
                    adjacent_districts[district["zone_id"]].add(69)
                case 69:
                    adjacent_districts[district["zone_id"]].add(108)
                case 87:
                    adjacent_districts[district["zone_id"]].add(90)
                case 90:
                    adjacent_districts[district["zone_id"]].add(87)
                case 80:
                    adjacent_districts[district["zone_id"]].add(103)
                case 103:
                    adjacent_districts[district["zone_id"]].add(80)
                case 31:
                    adjacent_districts[district["zone_id"]].add(98)
                case 98:
                    adjacent_districts[district["zone_id"]].add(31)
                case 115:
                    adjacent_districts[district["zone_id"]].add(111)
                    adjacent_districts[district["zone_id"]].add(41)
                case 111:
                    adjacent_districts[district["zone_id"]].add(115)
                case 41:
                    adjacent_districts[district["zone_id"]].add(115)
                    adjacent_districts[district["zone_id"]].discard(76)
                case 44:
                    adjacent_districts[district["zone_id"]].add(4)
                    adjacent_districts[district["zone_id"]].add(55)
                case 4:
                    adjacent_districts[district["zone_id"]].add(44)
                case 55:
                    adjacent_districts[district["zone_id"]].add(44)
                case 26:
                    adjacent_districts[district["zone_id"]].add(94)
                case 76:
                    adjacent_districts[district["zone_id"]].discard(41)
                case 19:# need to be veryfy
                    adjacent_districts[district["zone_id"]].add(30)
                case 30:
                    adjacent_districts[district["zone_id"]].add(19)
                    adjacent_districts[district["zone_id"]].discard(94)
                
            

    # 输出结果
    for zone, neighbors in adjacent_districts.items():
        print(f"Zone {zone} -> Adjacent Zones: {sorted(neighbors)}")
    with open("adjacent_districts.pkl", "wb") as f:
        pickle.dump(adjacent_districts, f)

if __name__ == "__main__":  
    run()  