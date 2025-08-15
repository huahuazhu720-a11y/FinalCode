import sqlite3
import os
import pandas as pd
class Database:
    def __init__(self, database_name):
        """
        初始化数据库连接。
        如果数据库文件不存在，则创建一个新的数据库；如果存在，则连接到已有的数据库。
        :param database_name: 数据库文件名
        """
        self.database_name = database_name
        self.conn = None

        if os.path.exists(self.database_name):
            print(f"Database '{self.database_name}' already exists. Connecting to it...")
        else:
            print(f"Database '{self.database_name}' does not exist. Creating a new one...")
        
        self.connect()

    def connect(self):
        """
        连接到数据库文件。如果文件不存在会自动创建。
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.database_name)
        return self.conn

    def create_table(self, table_name, columns):
        """
        动态创建指定的表。如果表已存在，则不会重新创建。
        :param table_name: 表名
        :param columns: 字典，包含列名和列定义（如数据类型和约束）
                        格式示例：{'id': 'INTEGER PRIMARY KEY AUTOINCREMENT', 'name': 'TEXT NOT NULL'}
        """
        try:
            cursor = self.connect().cursor()

            # 构建列定义的字符串
            column_definitions = ', '.join([f"{col_name} {col_def}" for col_name, col_def in columns.items()])

            # 创建表的 SQL 语句
            create_table_sql = f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {column_definitions}
                )
            '''
            cursor.execute(create_table_sql)
            print(f"Table '{table_name}' created successfully.")
        except sqlite3.Error as e:
            print(f"An error occurred while creating the table: {e}")
        finally:
            self.conn.commit()

    def sql_to_df(self,query):
        return pd.read_sql(query, self.conn)
    
    def save_df_as_table(self,TableName,df,if_exists='replace'):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df 必须是一个 pandas DataFrame")
        try:
            df.to_sql(TableName, self.conn, if_exists=if_exists, index=False)
            print(f"successfully saved table: `{TableName}`")
        except Exception as e:
            print(f"error：{e}")
    
    def insert_many(self, table_name, columns, values_list):
        """
        values_list :[list 里面的很多tuple，每个tuple是一行数据]
        """
        try:
            cursor = self.connect().cursor()
            columns_str = ", ".join(columns)
            placeholders = ", ".join(["?" for _ in columns])
            sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            cursor.executemany(sql, values_list)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error inserting multiple rows: {e}")
        finally:
            self.conn.commit()
    def insert_data(self, table_name, columns, values):
        """
        插入数据到指定表中,并返回插入的行 ID。
        :param table_name: 表名
        :param columns: 列名列表，例如 ['tract', 'year', 'work_population', ...]
        :param values: 值列表，与列名列表顺序一致，例如 ['Census Tract 121', '2022', 1267, ...]
        :return: 插入的行 ID（如果插入成功），否则返回 None。
        """
        try:
            cursor = self.connect().cursor()

            # 构造列名和占位符
            columns_str = ", ".join(columns)
            placeholders = ", ".join(["?" for _ in values])

            # 构造SQL语句
            sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            # 执行插入操作
            cursor.execute(sql, values)
            self.conn.commit()

            # 获取最后插入的行 ID
            inserted_id = cursor.lastrowid
            # print(f"Data inserted into '{table_name}' successfully. Inserted ID: {inserted_id}")
            return inserted_id
        except sqlite3.Error as e:
            print(f"An error occurred while inserting data: {e}")
            return None
        finally:
            self.conn.commit()
    def execute_query(self, query, params=None):
        """
        执行任意 SQL 查询。
        :param query: SQL 查询字符串
        :param params: 查询参数（可选），使用占位符 `?` 提供参数化查询
        :return: 查询结果列表（对于 SELECT 查询）或受影响的行数
        """
        try:
            cursor = self.connect().cursor()

            # 执行查询，处理带参数和不带参数的情况
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # 如果是 SELECT 查询，返回结果
            if query.strip().upper().startswith("SELECT"):
                results = cursor.fetchall()
                return results

            # 对于非 SELECT 查询，返回受影响的行数
            self.conn.commit()
            return cursor.rowcount

        except sqlite3.Error as e:
            print(f"An error occurred while executing the query: {e}")
            return None

    def close_connection(self):
        """
        关闭数据库连接。
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed.")


