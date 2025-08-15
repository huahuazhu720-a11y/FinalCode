"""
step8
Map the data downloaded from S0801 to the new zones.
Result: ZoneSensusData_NewYork, showing the original S0801 data under the new zone definitions.
"""
from database import Database
import pandas as pd
import logging  
logging.basicConfig(
    filename="error_log.log",
    level=logging.INFO,  # INFO 可以记录 INFO 及更高的级别（ERROR, CRITICAL）
    format="%(asctime)s - %(levelname)s - Year: %(year)s, ID: %(id)s, Error: %(error_message)s, CustomMessage: %(custom_message)s"
)
logging.info("This is an informational message.", extra={"year": 2025, "id": 1234, "error_message": "No error", "custom_message": "Info message"})

def run():
    DB=Database("GreYelHir.db")  

    # DB.execute_query("drop table ZoneSensusData_NewYork")
    # query=f"SELECT * FROM ZoneSensusData_NewYork WHERE year=2022"
    # dataA=DB.sql_to_df(query)
    # print(dataA.columns)
    # print(dataA['zone_id'].tolist())
    # exit()
    # print('a')
    years=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019",
        "2020","2021","2022"]   
    
    columnsDic = {
    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',  
    'zone_id': 'TEXT NOT NULL',  
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
    'S0801_C01_029E':'NUMERIC NOT NULL '
    }
    
    DB.create_table("ZoneSensusData_NewYork",columnsDic)

    
    print(DB.execute_query("SELECT name FROM sqlite_master WHERE type='table'"))

    SensusDataclumus=["zone_id","year",            
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

    grouped_df = pd.read_pickle("grouped_taxi_zones.pkl")
    grouped_df['geoid'] = grouped_df['geoid'].apply(
        lambda x: [str(item)[:-4] for item in x if pd.notna(item)]
    )
    for year in years:   
        print(f"start work on {year}")
        query=f"SELECT * FROM SensusData_NewYork WHERE year={year}"
        dataA=DB.sql_to_df(query)
        for _, row in grouped_df.iterrows():
            geoid = row['geoid']
            zone_id = row['zone_id']
            # print(type(geoid))
            # print(geoid)
            unrepeat_geoid=set(geoid)
            new_data = []
            for geoid_r in unrepeat_geoid:
                # print(type(geoid_r))
                # print(type(dataA['tract']))
                matching_data = dataA[dataA['tract'].astype(str) == str(geoid_r)]       
               
                exploded_geoid = grouped_df['geoid'].explode()
                # 计算 geoid 在整个 csv_df['processedgeoid'] 中的出现次数
                total_count = exploded_geoid.value_counts()[geoid_r]
                # print(total_count)
                total_in_rows = geoid.count(geoid_r)             
                # print(total_in_rows)
                # 计算出现比例
                ratio = total_in_rows / total_count
                if(ratio !=1.0):
                    print(f"ratio on tract:{geoid_r} is:{ratio}")
                
                for _, data_row in matching_data.iterrows():
                    # 获取 S0801_C01_001E 和 S0801_C01_026E
                    s0801_001E = data_row['S0801_C01_001E']
                    s0801_026E = data_row['S0801_C01_026E']
                    
                    # 用 ratio 进行加权
                    new_s0801_001E = (s0801_001E * ratio).astype(int)
                    new_s0801_026E = (s0801_026E * ratio).astype(int)
                    
                    # 生成新的记录
                    new_record = {
                        'zone_id': zone_id,
                        'S0801_C01_001E': new_s0801_001E,
                        'S0801_C01_026E': new_s0801_026E,
                        "S0801_C01_019E":data_row['S0801_C01_019E'],
                        "S0801_C01_002E":data_row['S0801_C01_002E'],
                        "S0801_C01_009E":data_row['S0801_C01_009E'],
                        "S0801_C01_010E":data_row['S0801_C01_010E'],
                        "S0801_C01_011E":data_row['S0801_C01_011E'],
                        "S0801_C01_037E":data_row['S0801_C01_037E'],
                        "S0801_C01_038E":data_row['S0801_C01_038E'],
                        "S0801_C01_039E":data_row['S0801_C01_039E'],
                        "S0801_C01_040E":data_row['S0801_C01_040E'],
                        "S0801_C01_041E":data_row['S0801_C01_041E'],
                        "S0801_C01_042E":data_row['S0801_C01_042E'],
                        "S0801_C01_043E":data_row['S0801_C01_043E'],
                        "S0801_C01_044E":data_row['S0801_C01_044E'],
                        "S0801_C01_045E":data_row['S0801_C01_045E'],
                        "S0801_C01_046E":data_row['S0801_C01_046E'],
                        'S0801_C01_034E':data_row['S0801_C01_034E'], 
                        'S0801_C01_033E':data_row['S0801_C01_033E'], 
                        'S0801_C01_036E':data_row['S0801_C01_036E'], 
                        'S0801_C01_035E':data_row['S0801_C01_035E'], 
                        'S0801_C01_030E':data_row['S0801_C01_030E'], 
                        'S0801_C01_032E':data_row['S0801_C01_032E'], 
                        'S0801_C01_031E':data_row['S0801_C01_031E'], 
                        'S0801_C01_028E':data_row['S0801_C01_028E'], 
                        'S0801_C01_027E':data_row['S0801_C01_027E'], 
                        'S0801_C01_003E':data_row['S0801_C01_003E'], 
                        'S0801_C01_004E':data_row['S0801_C01_004E'], 
                        'S0801_C01_005E':data_row['S0801_C01_005E'], 
                        'S0801_C01_006E':data_row['S0801_C01_006E'], 
                        'S0801_C01_007E':data_row['S0801_C01_007E'],
                        'S0801_C01_012E':data_row['S0801_C01_012E'],
                        'S0801_C01_029E':data_row['S0801_C01_029E']
                    }                    
                    new_data.append(new_record)
            new_df = pd.DataFrame(new_data)  
            new_df = new_df[~new_df.apply(lambda x: x < 0).any(axis=1)]
            if new_df.empty:
                print(f"final_df is empty!zoneidis:{zone_id},geoid: {unrepeat_geoid},year:{year}")
                continue
            final_df = new_df.groupby('zone_id').agg({
                    'S0801_C01_001E': 'sum',
                    'S0801_C01_026E': 'sum',
                    "S0801_C01_019E":'mean',
                    "S0801_C01_002E":'mean',
                    "S0801_C01_009E":'mean',
                    "S0801_C01_010E":'mean',
                    "S0801_C01_011E":'mean',
                    "S0801_C01_037E":'mean',
                    "S0801_C01_038E":'mean',
                    "S0801_C01_039E":'mean',
                    "S0801_C01_040E":'mean',
                    "S0801_C01_041E":'mean',
                    "S0801_C01_042E":'mean',
                    "S0801_C01_043E":'mean',
                    "S0801_C01_044E":'mean',
                    "S0801_C01_045E":'mean',
                    "S0801_C01_046E":'mean',
                    'S0801_C01_034E':'mean', 
                    'S0801_C01_033E':'mean', 
                    'S0801_C01_036E':'mean', 
                    'S0801_C01_035E':'mean', 
                    'S0801_C01_030E':'mean', 
                    'S0801_C01_032E':'mean', 
                    'S0801_C01_031E':'mean', 
                    'S0801_C01_028E':'mean', 
                    'S0801_C01_027E':'mean', 
                    'S0801_C01_003E':'mean', 
                    'S0801_C01_004E':'mean', 
                    'S0801_C01_005E':'mean', 
                    'S0801_C01_006E':'mean', 
                    'S0801_C01_007E':'mean', 
                    'S0801_C01_012E':'mean',
                    'S0801_C01_029E':'mean'
                }).reset_index()  
            if not final_df.empty:
                row_as_list = final_df.iloc[0].tolist()
                row_as_list.insert(1,year)
                DB.insert_data('ZoneSensusData_NewYork',SensusDataclumus,row_as_list)
                print(f"done {year} on {zone_id}")
            else:
                print("final_df is empty!")
   


if __name__ == "__main__":
    run()
            
            
            
                    

        