import pickle
import networkx as nx
import matplotlib.pyplot as plt

# 加载图
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# 获取节点颜色：根据 'type' 属性设置颜色
node_colors = []
for node, attrs in G.nodes(data=True):
    if attrs.get("type") == "w":
        node_colors.append("blue")
    elif attrs.get("type") == "R":
        node_colors.append("green")
    else:
        node_colors.append("gray")  # 默认颜色

# 尝试不同的布局算法
pos = nx.kamada_kawai_layout(G)  # 更适合有一定结构的图

# 筛选出 total_flow 大于 10 的边
edges_to_draw = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get("total_flow", 0) > 10]

# 获取符合条件的边的颜色（根据 total_flow 可以设置颜色）
edge_colors = [G[u][v]["total_flow"] for u, v in edges_to_draw]

# 绘制图
plt.figure(figsize=(28, 20))  # 增加图的尺寸

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

# 绘制符合条件的边
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=edges_to_draw,  # 只绘制符合条件的边
    edge_color=edge_colors,  # 根据 total_flow 设置边的颜色
    width=1,
    edge_cmap=plt.cm.Reds,  # 根据 total_flow 给边设置颜色范围
    connectionstyle="arc3,rad=0.2",  # 弧形边
    arrowstyle="->",
    arrowsize=2,    
)

# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

# 显示边的属性（total_flow）
edge_labels = {(u, v): G[u][v]["total_flow"] for u, v in edges_to_draw}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=0.5, label_pos=0.5)

# 保存为 PDF
plt.savefig("graph_visualization_improved.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
