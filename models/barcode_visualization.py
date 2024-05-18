import gudhi as gd

# 假设的 edge list
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

# 创建一个 Simplex Tree
st = gd.SimplexTree()

# 添加边到 Simplex Tree
for u, v in edges:
    st.insert([u, v])

# 也可以添加高维的 simplex，例如三角形
st.insert([0, 2, 3])

# 计算持久同调
diagram = st.persistence()

# 绘制条形码
gd.plot_persistence_barcode(diagram)
