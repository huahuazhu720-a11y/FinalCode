
"""
step5  Map the taxi trip data (originally between 200+ zones) to the new dataset showing trips between the 130 merged zones.
""" 
import requests
from database import Database
import pandas as pd
import logging  
import holidays
logging.basicConfig(
    filename="error_log.log",
    level=logging.INFO,  # INFO 可以记录 INFO 及更高的级别（ERROR, CRITICAL）
    format="%(asctime)s - %(levelname)s - Year: %(year)s, ID: %(id)s, Error: %(error_message)s, CustomMessage: %(custom_message)s"
)
logging.info("This is an informational message.", extra={"year": 2025, "id": 1234, "error_message": "No error", "custom_message": "Info message"})

def get_holidays(year):
    # 获取指定年份的美国法定节假日
    us_holidays = holidays.US(years=[year])
    
    # 格式化为 'YYYY-MM-DD' 的列表
    holiday_list = [date.strftime('%Y-%m-%d') for date in sorted(us_holidays.keys())]
    
    return holiday_list

def map_yellow_to_Od(year,daily=False):
    DB=Database("GreYelHir.db")   
    print(DB.execute_query("SELECT name FROM sqlite_master WHERE type='table'"))
    cursor = DB.connect().cursor()
    cursor.execute("PRAGMA table_info(YellowT)")
    columns = [row for row in cursor.fetchall()]
    print(columns)  # 输出列名列表
    grouped_df = pd.read_pickle("merged_taxi_zones.pkl")
    Origin_to_new = grouped_df.explode('original_OBJECTID')[['original_OBJECTID', 'zone_id']]
    Origin_to_new['original_OBJECTID'] = Origin_to_new['original_OBJECTID'].astype(str)
    # 创建映射字典
    Origin_to_new_dict = dict(zip(Origin_to_new['original_OBJECTID'], Origin_to_new['zone_id']))
    holiday_list = get_holidays(year)
    weekdays = pd.date_range(start=f"""{year}-01-01""", end=f"""{year}-12-31""", freq='B')  # 生成所有工作日（去除周末）
    valid_days = weekdays[~weekdays.isin(pd.to_datetime(holiday_list))]  # 去除节假日
    valid_days_count = len(valid_days) 
    holiday_condition = "AND tpep_pickup_datetime NOT IN (" + ",".join([f"'{date}'" for date in holiday_list]) + ")"

    # 查询语句，排除周末和节假日
    query = f"""
    SELECT tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID, passenger_count, total_amount
    FROM YellowT
    WHERE strftime('%Y', tpep_pickup_datetime) = '{year}'
    AND strftime('%Y', tpep_dropoff_datetime) = '{year}'
    AND strftime('%w', tpep_pickup_datetime) NOT IN ('0', '6')
    AND strftime('%w', tpep_dropoff_datetime) NOT IN ('0', '6') 
    {holiday_condition}
    """
    chunk = DB.sql_to_df(query)
        # 时间转换
    chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'])
    chunk['tpep_dropoff_datetime'] = pd.to_datetime(chunk['tpep_dropoff_datetime'])

    # ID 映射
    chunk['PULocationID'] = chunk['PULocationID'].astype(int).astype(str).map(Origin_to_new_dict)
    chunk['DOLocationID'] = chunk['DOLocationID'].astype(int).astype(str).map(Origin_to_new_dict)
    
    chunk = chunk.dropna(subset=['PULocationID', 'DOLocationID'])

    # 如果 PULocationID 或 DOLocationID 是默认值（即没有在字典中找到），也删除该记录
    chunk = chunk[(chunk['PULocationID'] != 'DEFAULT_PULocationID') & (chunk['DOLocationID'] != 'DEFAULT_DOLocationID')]
    chunk['pickup_date'] = chunk['tpep_pickup_datetime'].dt.date 
    chunk['PULocationID']=chunk['PULocationID'].astype(int).round(0)
    chunk['DOLocationID']=chunk['DOLocationID'].astype(int).round(0) 
    chunk['passenger_count']=chunk['passenger_count'].astype(float)    
    chunk['total_amount']=chunk['total_amount'].astype(float)      
    if daily:
        grouped_data = chunk.groupby(['PULocationID', 'DOLocationID','pickup_date']).size().reset_index(name='pickup_count')
        # print(len(grouped_data))
        # print(grouped_data.head(50))
        grouped_data.to_csv("Daily_Records_yellow.csv", index=False)
        grouped_data.to_pickle("Daily_Records_yellow.pkl")
    
    mean_pickup_count = chunk.groupby(['PULocationID', 'DOLocationID']).agg(
        total_pickup_count=('passenger_count','size'),
        mean_passenger=('passenger_count','mean'),
        mean_cost=('total_amount','mean')        
    ).reset_index()   
    mean_pickup_count['mean_pickup_count'] = mean_pickup_count['total_pickup_count'] / valid_days_count
    mean_pickup_count = mean_pickup_count[mean_pickup_count['mean_pickup_count'] >= 1]
    mean_pickup_count['mean_pickup_count'] = mean_pickup_count['mean_pickup_count'].round()
    # 输出最终结果
    # print(len(mean_pickup_count))
    # print(mean_pickup_count.head(50))
    
    mean_pickup_count.to_csv("Mean_pickup_count_yellow.csv", index=False)
    mean_pickup_count.to_pickle("Mean_pickup_count_yellow.pkl")

def map_Green_to_Od(year,daily=False):
    DB=Database("GreYelHir.db")   
    print(DB.execute_query("SELECT name FROM sqlite_master WHERE type='table'"))
    cursor = DB.connect().cursor()
    cursor.execute("PRAGMA table_info(GreenT)")
    columns = [row for row in cursor.fetchall()]
    print(columns)  # 输出列名列表
    grouped_df = pd.read_pickle("merged_taxi_zones.pkl")
    Origin_to_new = grouped_df.explode('original_OBJECTID')[['original_OBJECTID', 'zone_id']]
    Origin_to_new['original_OBJECTID'] = Origin_to_new['original_OBJECTID'].astype(str)
    # 创建映射字典
    Origin_to_new_dict = dict(zip(Origin_to_new['original_OBJECTID'], Origin_to_new['zone_id']))
    holiday_list = get_holidays(year)
    weekdays = pd.date_range(start=f"""{year}-01-01""", end=f"""{year}-12-31""", freq='B')  # 生成所有工作日（去除周末）
    valid_days = weekdays[~weekdays.isin(pd.to_datetime(holiday_list))]  # 去除节假日
    valid_days_count = len(valid_days) 
    holiday_condition = "AND lpep_pickup_datetime NOT IN (" + ",".join([f"'{date}'" for date in holiday_list]) + ")"

    # 查询语句，排除周末和节假日
    query = f"""
    SELECT lpep_pickup_datetime, lpep_dropoff_datetime, PULocationID, DOLocationID, passenger_count, total_amount
    FROM GreenT
    WHERE strftime('%Y', lpep_pickup_datetime) = '{year}'
    AND strftime('%Y', lpep_dropoff_datetime) = '{year}'
    AND strftime('%w', lpep_pickup_datetime) NOT IN ('0', '6')
    AND strftime('%w', lpep_dropoff_datetime) NOT IN ('0', '6') 
    {holiday_condition}
    """
    chunk = DB.sql_to_df(query)
        # 时间转换
    chunk['lpep_pickup_datetime'] = pd.to_datetime(chunk['lpep_pickup_datetime'])   

    # ID 映射
    chunk['PULocationID'] = chunk['PULocationID'].astype(int).astype(str).map(Origin_to_new_dict)
    chunk['DOLocationID'] = chunk['DOLocationID'].astype(int).astype(str).map(Origin_to_new_dict)
    
    chunk = chunk.dropna(subset=['PULocationID', 'DOLocationID'])

    # 如果 PULocationID 或 DOLocationID 是默认值（即没有在字典中找到），也删除该记录
    chunk = chunk[(chunk['PULocationID'] != 'DEFAULT_PULocationID') & (chunk['DOLocationID'] != 'DEFAULT_DOLocationID')]
    chunk['pickup_date'] = chunk['lpep_pickup_datetime'].dt.date 
    chunk['PULocationID']=chunk['PULocationID'].astype(int).round(0)
    chunk['DOLocationID']=chunk['DOLocationID'].astype(int).round(0)    
    chunk['passenger_count']=chunk['passenger_count'].astype(float)    
    chunk['total_amount']=chunk['total_amount'].astype(float)    
    if daily:
        grouped_data = chunk.groupby(['PULocationID', 'DOLocationID','pickup_date']).size().reset_index(name='pickup_count')
        # print(len(grouped_data))
        # print(grouped_data.head(50))
        grouped_data.to_csv("Daily_Records_Green.csv", index=False)
        grouped_data.to_pickle("Daily_Records_Green.pkl")
    
    mean_pickup_count = chunk.groupby(['PULocationID', 'DOLocationID']).agg(
        total_pickup_count=('passenger_count','size'),
        mean_passenger=('passenger_count','mean'),
        mean_cost=('total_amount','mean')        
    ).reset_index()    
    mean_pickup_count['mean_pickup_count'] = mean_pickup_count['total_pickup_count'] / valid_days_count
    mean_pickup_count = mean_pickup_count[mean_pickup_count['mean_pickup_count'] >= 1]
    mean_pickup_count['mean_pickup_count'] = mean_pickup_count['mean_pickup_count'].round()
    # 输出最终结果
    # print(len(mean_pickup_count))
    # print(mean_pickup_count.head(50))
    
    mean_pickup_count.to_csv("Mean_pickup_count_Green.csv", index=False)
    mean_pickup_count.to_pickle("Mean_pickup_count_Green.pkl")

def map_ForHire_to_Od(year,daily=False):
    DB=Database("GreYelHir.db") 
    cursor = DB.connect().cursor()
    cursor.execute("PRAGMA table_info(ForHire)")
    columns = [row[1] for row in cursor.fetchall()]
    print(columns)  # 输出列名列表
    print(DB.execute_query("SELECT name FROM sqlite_master WHERE type='table'"))
    grouped_df = pd.read_pickle("merged_taxi_zones.pkl")
    Origin_to_new = grouped_df.explode('original_OBJECTID')[['original_OBJECTID', 'zone_id']]
    Origin_to_new['original_OBJECTID'] = Origin_to_new['original_OBJECTID'].astype(str)
    # 创建映射字典
    Origin_to_new_dict = dict(zip(Origin_to_new['original_OBJECTID'], Origin_to_new['zone_id']))
    holiday_list = get_holidays(year)
    weekdays = pd.date_range(start=f"""{year}-01-01""", end=f"""{year}-12-31""", freq='B')  # 生成所有工作日（去除周末）
    valid_days = weekdays[~weekdays.isin(pd.to_datetime(holiday_list))]  # 去除节假日
    valid_days_count = len(valid_days) 
    holiday_condition = "AND pickup_datetime NOT IN (" + ",".join([f"'{date}'" for date in holiday_list]) + ")"

    # 查询语句，排除周末和节假日
    query = f"""
    SELECT pickup_datetime, dropOff_datetime, PUlocationID, DOlocationID 
    FROM ForHire
    WHERE strftime('%Y', pickup_datetime) = '{year}'
    AND strftime('%Y', dropOff_datetime) = '{year}'
    AND strftime('%w', pickup_datetime) NOT IN ('0', '6')
    AND strftime('%w', dropOff_datetime) NOT IN ('0', '6') 
    {holiday_condition}
    """
    chunk = DB.sql_to_df(query)
        # 时间转换
    chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'])   
  
    chunk['PUlocationID'] = pd.to_numeric(chunk['PUlocationID'], errors='coerce')
    chunk['PUlocationID'] = chunk['PUlocationID'].astype(int).astype(str).map(Origin_to_new_dict)
    chunk['DOlocationID'] = pd.to_numeric(chunk['DOlocationID'], errors='coerce')
    chunk['DOlocationID'] = chunk['DOlocationID'].astype(int).astype(str).map(Origin_to_new_dict)

    chunk = chunk.dropna(subset=['PUlocationID', 'DOlocationID'])

    # 如果 PUlocationID 或 DOlocationID 是默认值（即没有在字典中找到），也删除该记录
    chunk = chunk[(chunk['PUlocationID'] != 'DEFAULT_PUlocationID') & (chunk['DOlocationID'] != 'DEFAULT_DOlocationID')]
  
    chunk['pickup_date'] = chunk['pickup_datetime'].dt.date 
    chunk['PUlocationID']=chunk['PUlocationID'].astype(float).round(0)
    chunk['DOlocationID']=chunk['DOlocationID'].astype(float).round(0)  
    if daily:
        grouped_data = chunk.groupby(['PUlocationID', 'DOlocationID','pickup_date']).size().reset_index(name='pickup_count')
        # print(len(grouped_data))
        # print(grouped_data.head(50))
        grouped_data.to_csv("Daily_records_ForHire.csv", index=False)
        grouped_data.to_pickle("Daily_records_ForHire.pkl")
    
    mean_pickup_count = chunk.groupby(['PUlocationID', 'DOlocationID']).size().reset_index(name='total_pickup_count')    
    mean_pickup_count['mean_pickup_count'] = mean_pickup_count['total_pickup_count'] / valid_days_count
    mean_pickup_count = mean_pickup_count[mean_pickup_count['mean_pickup_count'] >= 1]
    mean_pickup_count['mean_pickup_count'] = mean_pickup_count['mean_pickup_count'].round()
    # 输出最终结果
    # print(len(mean_pickup_count))
    # print(mean_pickup_count.head(50))   
    mean_pickup_count.to_csv("Mean_pickup_count_forHire.csv", index=False)
    mean_pickup_count.to_pickle("Mean_pickup_count_forHire.pkl")
    
def run():
    map_yellow_to_Od(2022,False)
    map_Green_to_Od(2022,False)
    map_ForHire_to_Od(2022,False)
if __name__ == "__main__":  
    run()  
