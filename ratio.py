'''
本代码用于得到：
从 NEW YORK CITY（包括五个区）到 曼哈顿工作的人数 用W1来表示
和
所有到曼哈顿工作的人数（包括来这5个区和这5个区之外的） 用W2来表示
比
'''
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

# 读取第一个文件
file_name = r"data\ny_od_main_JT01_2022.csv"
df = pd.read_csv(file_name, usecols=["w_geocode", "h_geocode", "S000"])
#计算在从这5个地方来new york county 工作人口数量
within_5_block_main_total_s000 = df.loc[
    df["w_geocode"].astype(str).str.startswith("36061") & 
    df["h_geocode"].astype(str).str.startswith(("36061", "36047", "36081", "36005", "36085")),
    "S000"
].sum()
#计算在new York state 总共 new york county 工作人口数量
main_total_s000 = df.loc[
    df["w_geocode"].astype(str).str.startswith("36061"),
    "S000"
].sum()
# 读取第二个文件
file_name = r"data\ny_od_aux_JT01_2022.csv"
df = pd.read_csv(file_name, usecols=["w_geocode", "S000"])
#计算在new York state之外 总共 new york county 工作人口数量
aux_total_s000 = df.loc[
    df["w_geocode"].astype(str).str.startswith("36061"),
    "S000"
].sum()

# 计算总人数和比例
total_workers = main_total_s000 + aux_total_s000
ratio = within_5_block_main_total_s000 / total_workers if total_workers > 0 else 0

# 生成 PDF 报告
pdf_filename = "Ratio_Report.pdf"
c = canvas.Canvas(pdf_filename, pagesize=letter)
c.setFont("Helvetica", 12)

text_lines = [
    f"Works in New York County who come from New York City, W1: {within_5_block_main_total_s000}",
    f"Works in New York County who come from New York state, W2: {main_total_s000}",
    f"Works in New York County who from outside New York City, W3: {aux_total_s000}",
    f"Works in New York County in total, W2 + W3 = W4: {total_workers}",
    f"The ratio, W1 / W4: {ratio:.2%}"
]

y_position = 750  # 设置文本起始位置
for line in text_lines:
    c.drawString(100, y_position, line)
    y_position -= 20  # 逐行向下

c.save()
print(f"Report saved as {pdf_filename}")
