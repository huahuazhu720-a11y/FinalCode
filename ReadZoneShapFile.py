import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# 解压后的 Shapefile 文件路径
shapefile_path = r"shapfile\taxi_zones.shp"

# 读取 Shapefile 文件
gdf = gpd.read_file(shapefile_path)
pdf_file_path = "taxi_zones_map.pdf"
with PdfPages(pdf_file_path) as pdf:
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 8))  # 调整图形大小
    # 绘制地理数据
    gdf.plot(ax=ax, color='lightblue', edgecolor='black')
    
    # 设置标题
    ax.set_title('Taxi Zones in NYC', fontsize=16)
    
    # 保存图形到 PDF 文件
    pdf.savefig(fig)
    plt.close(fig)

print(f"地图已保存为 {pdf_file_path}")