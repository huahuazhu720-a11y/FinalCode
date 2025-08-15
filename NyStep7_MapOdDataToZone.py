"""
step 7
Map the original OD data (originally between Census blocks) to the newly defined zones.
Results: two tables—ZoneOutNodes and ZoneInNodes.
zone_id represents a node.
OutZone_id in ZoneOutNodes represents destinations from zone_id.
InZone_id in ZoneInNodes represents origins going to zone_id.
count_of_jobs indicates the number of people traveling between the OD pair.
"""
import requests
import numpy as np
from database import Database
import pandas as pd
import logging  
import re


def opencsv(file_name): 
  
    # 加载 CSV 文件
    try:
        # 尝试使用默认的 engine (c)
        df = pd.read_csv(file_name)
        return df
    except Exception as e:
        # 记录错误信息
        logging.error("Failed to read CSV file.", extra={"file name":file_name,"error_message": str(e), "custom_message": str(e)})
        
        # 尝试使用 python 引擎读取
        try:
            df = pd.read_csv(file_name, engine='python')
            return df
        except Exception as inner_e:
            # 如果 python 引擎也失败，记录错误信息并重新抛出异常
            logging.error("Failed to read CSV file with python engine.", extra={"file name":file_name, "error_message": str(inner_e), "custom_message": str(inner_e)})
            raise

def run():
    DB=Database("GreYelHir.db")
    OutColumnsDic = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT', 
        'year':'TEXT NOT NULL', 
        'zone_id': 'TEXT NOT NULL',  
        'OutZone_id': 'TEXT NOT NULL',
        'count_of_jobs': 'NUMERIC NOT NULL',        
        }

    DB.create_table("ZoneOutNodes",OutColumnsDic)
    OutColumns=[
        'year','zone_id','OutZone_id','count_of_jobs'
    ]

    InlumnsDic = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT', 
        'year':'TEXT NOT NULL', 
        'zone_id': 'TEXT NOT NULL',  
        'InZone_id': 'TEXT NOT NULL',
        'count_of_jobs': 'NUMERIC NOT NULL',        
        }
        
    DB.create_table("ZoneInNodes",InlumnsDic)
    InColumns=[
        'year','zone_id','InZone_id','count_of_jobs'
    ]

    grouped_df = pd.read_pickle("grouped_taxi_zones.pkl")
    grouped_df['geoid'] = grouped_df['geoid'].apply(
        lambda x: [str(item) for item in x if pd.notna(item)]
    )
    years=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019",
            "2020","2021","2022"]
    prefixes = ["36061", "36047", "36081", "36005", "36085"]
    for year in years:    
        file_name = f"data\\ny_od_main_JT01_{year}.csv"
        df=opencsv(file_name)
        df = df[df['w_geocode'].astype(str).str.startswith(tuple(prefixes)) & 
                df['h_geocode'].astype(str).str.startswith(tuple(prefixes))]
        df['w_geocode'] = df['w_geocode'].astype(str)
        df['h_geocode'] = df['h_geocode'].astype(str)
        geoid_to_zone = grouped_df.explode('geoid')[['geoid', 'zone_id']]
        geoid_to_zone['geoid'] = geoid_to_zone['geoid'].astype(str)
        # 创建映射字典
        geoid_to_zone_dict = dict(zip(geoid_to_zone['geoid'], geoid_to_zone['zone_id']))

        # 添加 w_zone_id 和 h_zone_id 列
        df['w_zone_id'] = df['w_geocode'].map(geoid_to_zone_dict)
        df['h_zone_id'] = df['h_geocode'].map(geoid_to_zone_dict)

        # 处理可能的 NaN 值（如果有些 geocode 没有匹配到 zone_id）
        df.fillna(-1, inplace=True)  # 或者用其他默认值
        zone_id_list = grouped_df['zone_id'].tolist()
        for zone_id in zone_id_list:
            result = df[df['h_zone_id'] == zone_id].groupby('w_zone_id')['S000'].sum().reset_index()
            result["year"] = year
            result["zone_id"] = zone_id
            result = result[["year", "zone_id", "w_zone_id", "S000"]] 
            result["w_zone_id"] = result["w_zone_id"].astype(int)
            result_list = [tuple(row) for row in result.itertuples(index=False, name=None)]
            DB.insert_many('ZoneOutNodes',OutColumns,result_list)

            result1 = df[df['w_zone_id'] == zone_id].groupby('h_zone_id')['S000'].sum().reset_index()
            result1["year"] = year
            result1["zone_id"] = zone_id
            result1 = result1[["year", "zone_id", "h_zone_id", "S000"]] 
            result1_list = [tuple(row) for row in result1.itertuples(index=False, name=None)]
            DB.insert_many('ZoneInNodes',InColumns,result1_list)
            print(f"done year: {year} zone id:{zone_id}")

if __name__ == "__main__":
    run()
    
