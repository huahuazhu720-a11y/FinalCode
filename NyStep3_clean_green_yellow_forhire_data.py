'''
Download the 2022 monthly data (January to December) for Green Taxi, Yellow Taxi, and For-Hire Vehicles.
After downloading, clean the data by:
Keeping only rows with complete data.
Further filtering rows where pickup_datetime and dropoff_datetime are both between 6:00 AM and 10:00 AM.
Create two datasets:
taxi: containing Green and Yellow Taxi data.
for_hire_vehicles: containing For-Hire Vehicles data.
'''
import pandas as pd
import numpy as np
import os
from database import Database
import logging  

logging.basicConfig(
    filename="error_log.log",
    level=logging.INFO,  # INFO 可以记录 INFO 及更高的级别（ERROR, CRITICAL）
    format="%(asctime)s - %(levelname)s - Year: %(year)s, ID: %(id)s, Error: %(error_message)s, CustomMessage: %(custom_message)s"
)
def setupdatabase():
    DB=Database("GreYelHir.db")
    OutColumnsDic = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',  
        'lpep_pickup_datetime': 'TEXT NOT NULL',  
        'lpep_dropoff_datetime': 'TEXT NOT NULL',
        'PULocationID': 'TEXT NOT NULL',        
        'DOLocationID': 'TEXT NOT NULL',        
        'passenger_count': 'TEXT NOT NULL',        
        'trip_distance': 'TEXT NOT NULL',        
        'payment_type': 'TEXT NOT NULL',        
        'congestion_surcharge': 'TEXT NOT NULL',        
        'total_amount': 'TEXT NOT NULL'            
        }
        
    DB.create_table("GreenT",OutColumnsDic)

    OutColumnsDic = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',  
        'tpep_pickup_datetime': 'TEXT NOT NULL',  
        'tpep_dropoff_datetime': 'TEXT NOT NULL',
        'PULocationID': 'TEXT NOT NULL',        
        'DOLocationID': 'TEXT NOT NULL',        
        'passenger_count': 'TEXT NOT NULL',        
        'trip_distance': 'TEXT NOT NULL',        
        'payment_type': 'TEXT NOT NULL',        
        'congestion_surcharge': 'TEXT NOT NULL',        
        'total_amount': 'TEXT NOT NULL',        
        'airport_fee': 'TEXT NOT NULL',        
        }
        
    DB.create_table("YellowT",OutColumnsDic)

    OutColumnsDic = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',      
        'PUlocationID': 'TEXT NOT NULL',        
        'DOlocationID': 'TEXT NOT NULL',
        'pickup_datetime': 'TEXT NOT NULL',  
        'dropOff_datetime': 'TEXT NOT NULL'
        }
        
    DB.create_table("ForHire",OutColumnsDic)
    print("database setup successfully")
    return DB
def dealwithGreenT(DB):
    year = 2022
    columns_to_read = ["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID","passenger_count","trip_distance","payment_type","congestion_surcharge","total_amount"]
    for month in range(1, 13):  # 从 1 月到 12 月        
        file_name = os.path.join("data", f"green_tripdata_{year}-{month:02d}.parquet")
        # 读取数据
        df = pd.read_parquet(file_name, columns=columns_to_read)

        # 确保空字符串替换为 NaN
        df["PULocationID"] = df["PULocationID"].replace("", np.nan)
        df["DOLocationID"] = df["DOLocationID"].replace("", np.nan)
        df["lpep_pickup_datetime"] = df["lpep_pickup_datetime"].replace("", np.nan)
        df["lpep_dropoff_datetime"] = df["lpep_dropoff_datetime"].replace("", np.nan)
        df["passenger_count"] = df["passenger_count"].replace("", np.nan)
        df["trip_distance"] = df["trip_distance"].replace("", np.nan)
        df["payment_type"] = df["payment_type"].replace("", np.nan)
        df["total_amount"] = df["total_amount"].replace("", np.nan)
        df["congestion_surcharge"] = df["congestion_surcharge"].replace("", 0)
        

        # 过滤掉缺失值
        df = df.dropna(subset=columns_to_read)

        # 解析时间格式（避免错误）
        df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

        # 过滤早晨 6-9 点的数据
        df_filtered = df[
            (df["lpep_pickup_datetime"].dt.hour >= 6) & (df["lpep_pickup_datetime"].dt.hour <= 10) &
            (df["lpep_dropoff_datetime"].dt.hour >= 6) & (df["lpep_dropoff_datetime"].dt.hour <= 10)
        ].copy()

    
        df_filtered["PULocationID"] = df_filtered["PULocationID"].fillna("-1").astype(str)  # 确保是字符串
        df_filtered["DOLocationID"] = df_filtered["DOLocationID"].fillna("-1").astype(str)  # 确保是字符串
        df_filtered["lpep_pickup_datetime"] = df_filtered["lpep_pickup_datetime"].astype(str)  # 转换为字符串
        df_filtered["lpep_dropoff_datetime"] = df_filtered["lpep_dropoff_datetime"].astype(str)  # 转换为字符串

        # 转换数据为 tuple
        records_to_insert = [tuple(row) for row in df_filtered.itertuples(index=False, name=None)]

        # 避免空数据插入 SQL 时报错
        if records_to_insert:
            batch_size = 10000
            for i in range(0, len(records_to_insert), batch_size):
                DB.insert_many('GreenT', columns_to_read, records_to_insert[i:i+batch_size])


def dealwithForHire(DB):
    year = 2022
    columns_to_read = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    for month in range(1, 13):  # 从 1 月到 12 月        
        file_name = os.path.join("data", f"fhv_tripdata_{year}-{month:02d}.parquet")
        # 读取数据
        df = pd.read_parquet(file_name, columns=columns_to_read)

        # 确保空字符串替换为 NaN
        df["PUlocationID"] = df["PUlocationID"].replace("", np.nan)
        df["DOlocationID"] = df["DOlocationID"].replace("", np.nan)

        # 过滤掉缺失值
        df = df.dropna(subset=columns_to_read)

        # 解析时间格式（避免错误）
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        df["dropOff_datetime"] = pd.to_datetime(df["dropOff_datetime"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

        # 过滤早晨 6-9 点的数据
        df_filtered = df[
            (df["pickup_datetime"].dt.hour >= 6) & (df["pickup_datetime"].dt.hour <= 10) &
            (df["dropOff_datetime"].dt.hour >= 6) & (df["dropOff_datetime"].dt.hour <= 10)
        ].copy()

    
        df_filtered["PUlocationID"] = df_filtered["PUlocationID"].fillna("-1").astype(str)  # 确保是字符串
        df_filtered["DOlocationID"] = df_filtered["DOlocationID"].fillna("-1").astype(str)  # 确保是字符串
        df_filtered["pickup_datetime"] = df_filtered["pickup_datetime"].astype(str)  # 转换为字符串
        df_filtered["dropOff_datetime"] = df_filtered["dropOff_datetime"].astype(str)  # 转换为字符串

        # 转换数据为 tuple
        records_to_insert = [tuple(row) for row in df_filtered.itertuples(index=False, name=None)]

        # 避免空数据插入 SQL 时报错
        if records_to_insert:
            batch_size = 10000
            for i in range(0, len(records_to_insert), batch_size):
                DB.insert_many('ForHire', columns_to_read, records_to_insert[i:i+batch_size])



def dealwithYellowT(DB):
    year = 2022
    columns_to_read = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID","passenger_count","trip_distance","payment_type","congestion_surcharge","total_amount","airport_fee"]
    for month in range(1, 13):  # 从 1 月到 12 月        
        file_name = os.path.join("data", f"yellow_tripdata_{year}-{month:02d}.parquet")
        # 读取数据
        df = pd.read_parquet(file_name, columns=columns_to_read)

        # 确保空字符串替换为 NaN
        df["PULocationID"] = df["PULocationID"].replace("", np.nan)
        df["DOLocationID"] = df["DOLocationID"].replace("", np.nan)
        df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].replace("", np.nan)
        df["tpep_dropoff_datetime"] = df["tpep_dropoff_datetime"].replace("", np.nan)
        df["passenger_count"] = df["passenger_count"].replace("", np.nan)
        df["trip_distance"] = df["trip_distance"].replace("", np.nan)
        df["payment_type"] = df["payment_type"].replace("", np.nan)
        df["total_amount"] = df["total_amount"].replace("", np.nan)
        df["congestion_surcharge"] = df["congestion_surcharge"].replace("", 0)
        df["airport_fee"] = df["airport_fee"].replace("", 0)
        

        # 过滤掉缺失值
        df = df.dropna(subset=columns_to_read)

        # 解析时间格式（避免错误）
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

        # 过滤早晨 6-9 点的数据
        df_filtered = df[
            (df["tpep_pickup_datetime"].dt.hour >= 6) & (df["tpep_pickup_datetime"].dt.hour <= 10) &
            (df["tpep_dropoff_datetime"].dt.hour >= 6) & (df["tpep_dropoff_datetime"].dt.hour <= 10)
        ].copy()

    
        df_filtered["PULocationID"] = df_filtered["PULocationID"].fillna("-1").astype(str)  # 确保是字符串
        df_filtered["DOLocationID"] = df_filtered["DOLocationID"].fillna("-1").astype(str)  # 确保是字符串
        df_filtered["tpep_pickup_datetime"] = df_filtered["tpep_pickup_datetime"].astype(str)  # 转换为字符串
        df_filtered["tpep_dropoff_datetime"] = df_filtered["tpep_dropoff_datetime"].astype(str)  # 转换为字符串

        # 转换数据为 tuple
        records_to_insert = [tuple(row) for row in df_filtered.itertuples(index=False, name=None)]

        # 避免空数据插入 SQL 时报错
        if records_to_insert:
            batch_size = 10000
            for i in range(0, len(records_to_insert), batch_size):
                DB.insert_many('YellowT', columns_to_read, records_to_insert[i:i+batch_size])
def run():
    DB = setupdatabase()     
    dealwithForHire(DB)
    print("done with dealwithForHire")
    dealwithGreenT(DB)
    print("done with dealwithGreenT")
    dealwithYellowT(DB)
    print("done with dealwithYellowT")
if __name__ == "__main__":
    run()
    



