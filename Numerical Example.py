import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


# ========== 数据生成 ==========
def generate_data(mode, num_samples):
    """
    生成指定模式的数据
    :param mode: 模式编号（1, 2, 3）
    :param num_samples: 样本数量
    :return: 观测数据 X (num_samples x 5)
    """
    if mode == 1:
        s1 = np.random.uniform(-10, -7, num_samples)
        s2 = np.random.normal(-5, 1, num_samples)
    elif mode == 2:
        s1 = np.random.uniform(-3, -1, num_samples)
        s2 = np.random.normal(2, 1, num_samples)
    elif mode == 3:
        s1 = np.random.uniform(2, 5, num_samples)
        s2 = np.random.normal(7, 1, num_samples)

    # 线性变换矩阵
    A = np.array([
        [0.5768, 0.3766],
        [0.7382, 0.0566],
        [0.8291, 0.4009],
        [0.6519, 0.2070],
        [0.3972, 0.8045]
    ])

    # 生成观测数据并添加噪声
    S = np.column_stack([s1, s2])
    noise = np.random.normal(0, 0.01, (num_samples, 5))
    X = S @ A.T + noise
    return X


# ========== 案例1数据生成 ==========
# 训练数据（正常数据）
train_mode1 = generate_data(1, 200)
train_mode2 = generate_data(2, 200)
train_mode3 = generate_data(3, 200)
train_data = np.vstack([train_mode1, train_mode2, train_mode3])

# 测试数据（案例1：模式1 + 模式2阶跃故障）
test_mode1 = generate_data(1, 200)
test_mode2 = generate_data(2, 200)
test_mode2[:, 0] += 0.08  # 对x1添加阶跃故障
test_case1 = np.vstack([test_mode1, test_mode2])

# ========== 案例2数据生成 ==========
test_case2 = []
# 模式1数据（正常）
test_case2.extend(generate_data(1, 200))
# 模式2数据（正常）
test_case2.extend(generate_data(2, 200))
# 模式2斜坡故障（301-400）
for i in range(200, 400):
    x = generate_data(2, 1)
    x[0, 0] += 0.002 * (i - 100)  # 斜坡故障
    test_case2.append(x)
# 模式3数据（正常）
test_case2.extend(generate_data(3, 200))
# 模式3阶跃故障（501-600）
test_mode3_fault = generate_data(3, 100)
test_mode3_fault[:, 0] += 0.08
test_case2.extend(test_mode3_fault)
test_case2 = np.vstack(test_case2)

# ========== 多流形谱聚类（训练数据） ==========
# 使用前两个变量可视化（与论文图3一致）
X_vis = train_data[:, :2]

# 谱聚类（设置3个簇）
cluster_model = SpectralClustering(n_clusters=3, affinity='rbf', random_state=0)
clusters = cluster_model.fit_predict(X_vis)

# ========== 可视化聚类结果 ==========
plt.figure(figsize=(8, 6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Multimodal Data Clustering (Training Set)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.show()
