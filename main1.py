import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2, uniform, norm
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering

# 设置随机种子
np.random.seed(4)


# 1. 数据生成函数
def generate_data(mode, num_samples):
    if mode == 1:
        s1 = uniform.rvs(loc=-10, scale=3, size=num_samples)  # -10 到 -7
        s2 = norm.rvs(loc=-5, scale=1, size=num_samples)
    elif mode == 2:
        s1 = uniform.rvs(loc=-3, scale=2, size=num_samples)  # -3 到 -1
        s2 = norm.rvs(loc=2, scale=1, size=num_samples)
    elif mode == 3:
        s1 = uniform.rvs(loc=2, scale=3, size=num_samples)  # 2 到 5
        s2 = norm.rvs(loc=7, scale=1, size=num_samples)

    A = np.array([[0.5768, 0.3766],
                  [0.7382, 0.0566],
                  [0.8291, 0.4009],
                  [0.6519, 0.2070],
                  [0.3972, 0.8045]])

    S = np.column_stack((s1, s2))
    noise = norm.rvs(loc=0, scale=0.01, size=(num_samples, 5))
    X = S @ A.T + noise
    return X


# 2. 生成训练和测试数据
# 训练数据
train_mode1 = generate_data(1, 200)
train_mode2 = generate_data(2, 200)
train_mode3 = generate_data(3, 200)
train_data = np.vstack((train_mode1, train_mode2, train_mode3))

# 测试数据案例1
test_mode1 = generate_data(1, 200)
test_mode2 = generate_data(2, 200)
test_mode2[:, 0] += 0.08
test_case1 = np.vstack((test_mode1, test_mode2))

# 测试数据案例2
test_case2 = []
test_case2.append(generate_data(1, 200))  # 模式1
# 模式2带斜坡故障
mode2_ramp = []
for i in range(201, 401):
    x = generate_data(2, 1)
    x[0, 0] += 0.002 * (i - 100)
    mode2_ramp.append(x)
test_case2.append(np.vstack(mode2_ramp))
# 模式3数据
test_case2.append(generate_data(3, 100))
# 模式3带阶跃故障
test_mode3_fault = generate_data(3, 100)
test_mode3_fault[:, 0] += 0.08
test_case2.append(test_mode3_fault)
test_case2 = np.vstack(test_case2)


# 3. 多流形谱聚类（3D版本）
def multimanifold_spectral_clustering(X_vis, n_clusters, n_neighbors):
    distance_matrix = squareform(pdist(X_vis))  # 3D欧几里得距离
    adj_matrix = np.zeros_like(distance_matrix)

    for i in range(X_vis.shape[0]):
        distances = distance_matrix[i, :]
        idx = np.argsort(distances)[1:n_neighbors + 1]  # 排除自身
        sigma = np.mean(distances[idx])
        adj_matrix[i, idx] = np.exp(-distances[idx] ** 2 / (2 * sigma ** 2))
        adj_matrix[idx, i] = adj_matrix[i, idx]  # 对称

    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = clustering.fit_predict(adj_matrix)
    return labels


# 4. 计算代表点（3D版本）
def compute_exemplars(X, labels):
    unique_labels = np.unique(labels)
    exemplars = []
    for mode in unique_labels:
        mode_data = X[labels == mode, :]
        distances = squareform(pdist(mode_data)) ** 2
        distance_sums = np.sum(distances, axis=1)
        idx = np.argmin(distance_sums)
        exemplars.append(mode_data[idx, :])
    return np.array(exemplars)


# 5. 3D椭球生成函数
def ellipsoid3(mu, cov_mat, scale):
    U, D, _ = np.linalg.svd(cov_mat)
    D = np.diag(np.sqrt(D))

    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    ap = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1) @ U @ D * scale
    x = ap[:, 0].reshape(x.shape) + mu[0]
    y = ap[:, 1].reshape(y.shape) + mu[1]
    z = ap[:, 2].reshape(z.shape) + mu[2]
    return x, y, z


# 6. 绘制3D聚类结果
def plot_3d_clustering(X_vis, clusters, true_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markers = ['+', '*', 'o']
    colors = ['r', 'b', 'g']

    # 散点图
    for label in range(3):
        mask = clusters == label
        ax.scatter(X_vis[mask, 0], X_vis[mask, 1], X_vis[mask, 2],
                   marker=markers[label], c=colors[label], s=50)

    # 绘制椭球
    for label in range(1, 4):
        mask = true_labels == label
        data = X_vis[mask, :]
        mu = np.mean(data, axis=0)
        cov_mat = np.cov(data.T)
        scale = np.sqrt(chi2.ppf(0.99, 3))

        x, y, z = ellipsoid3(mu, cov_mat, scale)
        ax.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='none')

    ax.set_title('3D聚类结果 (x1-x3)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.legend(['模式1', '模式2', '模式3'])
    plt.show()


# 7. 在线模式识别（3D版本）
def plot_mode_identification(test_case, case_num, exemplars):
    n_samples = test_case.shape[0]
    mode_ids = np.zeros(n_samples)

    for i in range(n_samples):
        x_new = test_case[i, :3]  # 使用3D特征
        distances = np.sum((exemplars - x_new) ** 2, axis=1)
        mode_ids[i] = np.argmin(distances) + 1  # 1-based索引

    plt.figure()
    plt.plot(range(1, n_samples + 1), mode_ids, color=[0.5, 0, 0.5], linewidth=2)
    plt.title(f'在线识别 (案例 {case_num})')
    plt.xlabel('样本索引')
    plt.ylabel('模式')
    plt.ylim(0, 4)
    plt.grid(True)
    plt.show()


# 主程序
X_vis = train_data[:, :3]  # 取前三个维度
true_labels = np.concatenate((np.ones(200), 2 * np.ones(200), 3 * np.ones(200)))
clusters = multimanifold_spectral_clustering(X_vis, 3, 10)

# 绘制3D聚类
plot_3d_clustering(X_vis, clusters, true_labels)

# 计算代表点
exemplars = compute_exemplars(X_vis, true_labels)

# 在线模式识别
plot_mode_identification(test_case1[:, :3], 1, exemplars)
plot_mode_identification(test_case2[:, :3], 2, exemplars)