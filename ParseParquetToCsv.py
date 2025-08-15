'''
本代码用于把Parquet 文件转化为csv文件
'''
import pandas as pd
import numpy as np

file_name = r"data\fhv_tripdata_2022-01.parquet"
df = pd.read_parquet(file_name)
# 确保空字符串被替换为 NaN
if df["PUlocationID"].dtype == 'object':
    df["PUlocationID"].replace("", np.nan, inplace=True)
    
if df["DOlocationID"].dtype == 'object':
    df["DOlocationID"].replace("", np.nan, inplace=True)

# 筛选 PULocationID 和 DOlocationID 都有值的行
df_filtered = df[df["PUlocationID"].notna() & df["DOlocationID"].notna()]
filtered_count = len(df_filtered)

# 计算总记录数
total_count = len(df)

# 计算比例
ratio = filtered_count / total_count
print(f"满足条件的记录数: {filtered_count}")
print(f"总记录数: {total_count}")
print(f"满足条件的记录和总记录的比例: {ratio:.4f}")
exit()


# df_filtered = df.dropna(subset=["PUlocationID", "DOlocationID"])

# df_sample = df.head(10)
df_filtered.to_csv("fhv_tripdata_2022-01.csv", index=False)


file_name = r"data\yellow_tripdata_2022-01.parquet"
df = pd.read_parquet(file_name)
df_sample = df.head(10)
df_sample.to_csv("yellow_tripdata_2022-01.csv", index=False)


file_name = r"data\green_tripdata_2022-01.parquet"
df = pd.read_parquet(file_name)
df_sample = df.head(10)
df_sample.to_csv("green_tripdata_2022-01.csv", index=False)
print('done')