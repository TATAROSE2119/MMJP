import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import chi2

# 设置随机种子以保证结果可重复
np.random.seed(42)


# 1. 数据生成函数
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


# 2. 生成训练和测试数据（与你的程序一致）
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

# 测试数据（案例2：模式1、模式2斜坡故障、模式3阶跃故障）
test_case2 = []
# 模式1数据（正常）
test_case2.extend(generate_data(1, 200))
# 模式2斜坡故障（201-400）
for i in range(200, 400):
    x = generate_data(2, 1)
    x[0, 0] += 0.002 * (i - 100)  # 斜坡故障
    test_case2.append(x)
# 模式3数据（正常）
test_case2.extend(generate_data(3, 100))
# 模式3阶跃故障（501-600）
test_mode3_fault = generate_data(3, 100)
test_mode3_fault[:, 0] += 0.08
test_case2.extend(test_mode3_fault)
test_case2 = np.vstack(test_case2)


# 3. 模拟多流形谱聚类
def multimanifold_spectral_clustering(X_vis, n_clusters=3, n_neighbors=10):
    """
    模拟多流形谱聚类，使用欧几里得距离和高斯核构建邻接矩阵
    """
    # 计算欧几里得距离矩阵
    distance_matrix = euclidean_distances(X_vis)  # 使用 x1 和 x2 维度

    # 构建邻接矩阵（使用 KNN 和高斯核）
    adj_matrix = np.zeros_like(distance_matrix)
    for i in range(len(X_vis)):
        neighbors = np.argsort(distance_matrix[i])[:n_neighbors]
        sigma = np.mean(distance_matrix[i, neighbors])  # 自动选择带宽
        adj_matrix[i, neighbors] = np.exp(-distance_matrix[i, neighbors] ** 2 / (2 * sigma ** 2))
        adj_matrix[neighbors, i] = adj_matrix[i, neighbors]  # 确保对称

    # 使用谱聚类
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                    assign_labels='kmeans', random_state=42)
    labels = clustering.fit_predict(adj_matrix)

    return labels


# 4. 计算每个模式的代表点（PMD 离线部分）
def compute_exemplars(X, labels, n_modes=3):
    exemplars = []
    for mode in range(n_modes):
        mode_data = X[labels == mode]
        # 计算测地线距离和（简化使用欧几里得距离）
        distance_sums = np.sum(euclidean_distances(mode_data) ** 2, axis=1)
        # 选择距离和最小的点作为代表点
        exemplar_idx = np.argmin(distance_sums)
        exemplars.append(mode_data[exemplar_idx])
    return np.array(exemplars)


# 5. 绘制聚类结果（生成图3）
def plot_clustering_result(X_vis, clusters, true_labels):
    """
    绘制训练数据的聚类结果，包含99%概率椭圆
    """
    plt.figure(figsize=(8, 6))

    # 为每个模式指定颜色和标记（与论文图3一致）
    markers = ['+', '*', 'o']
    colors = ['red', 'blue', 'green']
    for label in range(3):
        mask = clusters == label
        true_label = true_labels[mask][0]  # 假设预测标签与真实标签匹配
        plt.scatter(X_vis[mask, 0], X_vis[mask, 1], c=colors[true_label],
                    marker=markers[true_label], s=50, label=f'mode{true_label + 1}')

    # 绘制99%概率椭圆（基于每个模式的协方差）
    for true_label in range(3):
        mask = true_labels == true_label
        mean = np.mean(X_vis[mask], axis=0)
        cov = np.cov(X_vis[mask].T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        scale = np.sqrt(chi2.ppf(0.99, 2))  # 2D 的 99% 置信区间
        width, height = 2 * np.sqrt(eigenvalues) * scale
        ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle,
                                                 fill=False, color='blue', linestyle='--', linewidth=1.5)
        plt.gca().add_artist(ellipse)

    plt.title("Clustering Result of Multimodal Training Data (x1 vs x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


# 6. 实现 PMD 在线模式识别（生成图4）
def plot_mode_identification(test_case, case_num, exemplars, n_samples):
    mode_ids = []
    for i in range(n_samples):
        x_new = test_case[i, :2]  # 使用 x1 和 x2 维度
        # 计算新样本与每个模式的距离（简化使用欧几里得距离）
        distances = np.sum((exemplars - x_new) ** 2, axis=1)
        # 选择距离最小的模式
        mode = np.argmin(distances) + 1  # 转换为 1, 2, 3
        mode_ids.append(mode)

    mode_ids = np.array(mode_ids)

    # 绘制折线图
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_samples + 1), mode_ids, 'purple', linewidth=2)
    plt.title(f'Online Identification Result (Case {case_num})')
    plt.xlabel('Sample index')
    plt.ylabel('Mode')
    if case_num == 1:
        plt.ylim(0, 3)
    else:
        plt.ylim(0, 4)
    plt.grid(True)
    plt.show()
    return mode_ids


# 7. 主程序
if __name__ == "__main__":
    # 生成训练数据并聚类（图3）
    X_vis = train_data[:, :2]  # 仅使用 x1 和 x2 维度
    true_labels = np.array([0] * 200 + [1] * 200 + [2] * 200)  # 真实模式标签
    clusters = multimanifold_spectral_clustering(X_vis)
    plot_clustering_result(X_vis, clusters, true_labels)

    # 计算代表点（PMD 离线部分）
    exemplars = compute_exemplars(X_vis, true_labels)

    # 绘制图4，Case 1
    plot_mode_identification(test_case1, 1, exemplars, 400)

    # 绘制图4，Case 2
    plot_mode_identification(test_case2, 2, exemplars, 600)