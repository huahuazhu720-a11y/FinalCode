"""
Step 2
Code for downloading data from S0801 and storing it into the GreYelHir.db database, table SensusData_NewYork.
"""

import requests
from database import Database
import pandas as pd
import logging  
import sys
import os
logging.basicConfig(
    filename="error_log.log",
    level=logging.INFO,  # INFO 可以记录 INFO 及更高的级别（ERROR, CRITICAL）
    format="%(asctime)s - %(levelname)s - Year: %(year)s, ID: %(id)s, Error: %(error_message)s, CustomMessage: %(custom_message)s"
)
logging.info("This is an informational message.", extra={"year": 2025, "id": 1234, "error_message": "No error", "custom_message": "Info message"})
def getdata(year,county):
    # API Key 和 URL 配置
    API_KEY = os.getenv('Census_API_KEY')
    BASE_URL = f"https://api.census.gov/data/{year}/acs/acs5/subject"

    # 配置地理区域
    state_code = "36"  
    county_code = county 
    variable_map={
        "S0801_C01_001E":"Total working population", #居住在该地区的劳动力人口
        "S0801_C01_026E":"Working population who did not work from home",
        "S0801_C01_019E":"The proportion of the total working population who work outside their place of residence",# 总人口里面不在居住地sensus tract里面工作人口的占比
        "S0801_C01_002E":"The proportion of the total working population who drive to work",# 总人口里面开车上班人口的占比
        "S0801_C01_009E":"The proportion of the total working population who take public transportation",# 总人口里面公共交通去工作人口的占比
        "S0801_C01_010E":"The proportion of the total working population who walk to work",# 总人口里面走路去工作人口的占比
        "S0801_C01_011E":"The proportion of the total working population who bicycle to work",# 总人口里面自行车去工作人口的占比
        "S0801_C01_037E":"The proportion of the not work from home population who take less than 11 mins to work",# 不在家工作人口中，通勤时间少于10 min
        "S0801_C01_038E":"The proportion of the not work from home population who take 10-14 mins to work",# 不在家工作人口中，通勤时间10-14 min
        "S0801_C01_039E":"The proportion of the not work from home population who take 15-19 mins to work",# 不在家工作人口中，通勤时间15-19 min
        "S0801_C01_040E":"The proportion of the not work from home population who take 20-24 mins to work",# 不在家工作人口中，通勤时间20-24 min
        "S0801_C01_041E":"The proportion of the not work from home population who take 25-29 mins to work",# 不在家工作人口中，通勤时间50-29 min
        "S0801_C01_042E":"The proportion of the not work from home population who take 30-34 mins to work",# 不在家工作人口中，通勤时间30-34 min
        "S0801_C01_043E":"The proportion of the not work from home population who take 35-44 mins to work",# 不在家工作人口中，通勤时间35-44 min
        "S0801_C01_044E":"The proportion of the not work from home population who take 45-59 mins to work",# 不在家工作人口中，通勤时间45-59 min
        "S0801_C01_045E":"The proportion of the not work from home population who take more than 60 mins to work",# 不在家工作人口中，通勤时间more than 60 min
        "S0801_C01_046E":"The proportion of the not work from home population mean travel time to work",# 不在家工作人口中，平均通勤时间
        "S0801_C01_034E":"8:00 a.m. to 8:29 a.m. ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_033E":"7:30 a.m. to 7:59 a.m. ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_036E":"9:00 a.m. to 11:59 p.m. ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_035E":"8:30 a.m. to 8:59 a.m.  ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_030E":"6:00 a.m. to 6:29 a.m. ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_032E":"7:00 a.m. to 7:29 a.m. ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_031E":"6:30 a.m. to 6:59 a.m. ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_028E":"5:00 a.m. to 5:29 a.m. ", #在不在家工作的人口中，这个时间段出门工作的人的比例
        "S0801_C01_027E":"12:00 a.m. to 4:59 a.m ", #在不在家工作的人口中，这个时间段出门工作的人的比例        
        "S0801_C01_003E":"Car, truck, or van!!Drove alone ", #在总工作人口中，drove alone 的比例
        "S0801_C01_004E":"Car, truck, or van!!Carpooled ", #在总工作人口中，Carpooled 的比例
        "S0801_C01_005E":"Car, truck, or van!!Carpooled!!In 2-person carpool  ", 
        "S0801_C01_006E":"Car, truck, or van!!Carpooled!!In 3-person carpool  ", 
        "S0801_C01_007E":"Car, truck, or van!!Carpooled!!In 4-person or more carpool  ", 
        "S0801_C01_012E":"Taxicab, motorcycle, or other means ",
        "S0801_C01_029E":"5:30 a.m. to 5:59 a.m. " #在不在家工作的人口中，这个时间段出门工作的人的比例
    }

    variable_string = ",".join(variable_map.keys())
    # API 请求参数
    params = {
        "get": f"NAME,{variable_string}",  # 示例变量
        "for": f"tract:*",  # 请求所有县的数据
        "in": f"state:{state_code} county:{county_code}",  # 限定州
        "key": API_KEY  # API Key
    }

    # 发送请求
    response = requests.get(BASE_URL, params=params)

    # 处理响应
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"请求失败！错误代码: {response.status_code}")
        print("错误信息:", response.text)
        logging.info("failed to get info from API", extra={
            "year": year,
            "id": "50" + county,
            "error_message": response.status_code,
            "custom_message": response.text
        })        
        return ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0','','0', '0', '0', '0', '0', '0',    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0','0' ]

def run():
    DB=Database("GreYelHir.db")    
    counties = [
   "061","047","081","005","085"]

    years=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019",
        "2020","2021","2022"]
    
    
    columnsDic = {
    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',  
    'tract': 'TEXT NOT NULL',  
    'year': 'TEXT NOT NULL',    
    'S0801_C01_001E': 'NUMERIC NOT NULL',
    'S0801_C01_026E': 'NUMERIC NOT NULL',
    'S0801_C01_019E': 'NUMERIC NOT NULL',
    'S0801_C01_002E': 'NUMERIC NOT NULL',
    'S0801_C01_009E': 'NUMERIC NOT NULL',
    'S0801_C01_010E': 'NUMERIC NOT NULL',
    'S0801_C01_011E': 'NUMERIC NOT NULL',
    'S0801_C01_037E': 'NUMERIC NOT NULL',
    'S0801_C01_038E': 'NUMERIC NOT NULL',
    'S0801_C01_039E': 'NUMERIC NOT NULL',
    'S0801_C01_040E': 'NUMERIC NOT NULL',
    'S0801_C01_041E': 'NUMERIC NOT NULL',
    'S0801_C01_042E': 'NUMERIC NOT NULL',
    'S0801_C01_043E': 'NUMERIC NOT NULL',
    'S0801_C01_044E': 'NUMERIC NOT NULL',
    'S0801_C01_045E': 'NUMERIC NOT NULL',
    'S0801_C01_046E': 'NUMERIC NOT NULL',
    'S0801_C01_034E':'NUMERIC NOT NULL ', 
    'S0801_C01_033E':'NUMERIC NOT NULL ', 
    'S0801_C01_036E':'NUMERIC NOT NULL ', 
    'S0801_C01_035E':'NUMERIC NOT NULL ', 
    'S0801_C01_030E':'NUMERIC NOT NULL ', 
    'S0801_C01_032E':'NUMERIC NOT NULL ', 
    'S0801_C01_031E':'NUMERIC NOT NULL ', 
    'S0801_C01_028E':'NUMERIC NOT NULL ', 
    'S0801_C01_027E':'NUMERIC NOT NULL ', 
    'S0801_C01_003E':'NUMERIC NOT NULL ', 
    'S0801_C01_004E':'NUMERIC NOT NULL ', 
    'S0801_C01_005E':'NUMERIC NOT NULL ', 
    'S0801_C01_006E':'NUMERIC NOT NULL ', 
    'S0801_C01_007E':'NUMERIC NOT NULL ', 
    'S0801_C01_012E':'NUMERIC NOT NULL ', 
    'S0801_C01_029E':'NUMERIC NOT NULL ', 
    }
    
    DB.create_table("SensusData_NewYork",columnsDic)

    
    print(DB.execute_query("SELECT name FROM sqlite_master WHERE type='table'"))

    SensusDataclumus=["tract","year",            
                    "S0801_C01_001E",
                    "S0801_C01_026E",
                    "S0801_C01_019E",
                    "S0801_C01_002E",
                    "S0801_C01_009E",
                    "S0801_C01_010E",
                    "S0801_C01_011E",
                    "S0801_C01_037E",
                    "S0801_C01_038E",
                    "S0801_C01_039E",
                    "S0801_C01_040E",
                    "S0801_C01_041E",
                    "S0801_C01_042E",
                    "S0801_C01_043E",
                    "S0801_C01_044E",
                    "S0801_C01_045E",
                    "S0801_C01_046E",
                    'S0801_C01_034E', 
                    'S0801_C01_033E', 
                    'S0801_C01_036E', 
                    'S0801_C01_035E', 
                    'S0801_C01_030E', 
                    'S0801_C01_032E', 
                    'S0801_C01_031E', 
                    'S0801_C01_028E', 
                    'S0801_C01_027E', 
                    'S0801_C01_003E', 
                    'S0801_C01_004E', 
                    'S0801_C01_005E', 
                    'S0801_C01_006E', 
                    'S0801_C01_007E',
                    'S0801_C01_012E',
                    'S0801_C01_029E']

    for county in counties:       
        for year in years:   
            print(f"start work on tract {county},{year}")
            try:
                data=getdata(year,county)
                insertdata=[]
                if not data:
                    data= ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0','0','0' ]
                for i in range(1,len(data)):
                    data[i][0]=data[i][-3]+data[i][-2]+data[i][-1]
                    data[i].insert(1,year)
                    temp=tuple(data[i][:-3])
                    insertdata.append(temp)
                if insertdata:
                    batch_size = 10000
                    for i in range(0, len(insertdata), batch_size):
                        DB.insert_many("SensusData_NewYork",SensusDataclumus,insertdata)
                
            except Exception as e:
                logging.exception("error", extra={"year": year, "id": id, "error_message": str(e), "custom_message": str(e)})
            


if __name__ == "__main__":
    run()
