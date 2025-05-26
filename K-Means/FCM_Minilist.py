import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# FCM算法实现
def fcm(data, k=10, m=2, max_iter=100, tolerance=1e-4):
    print("FCM算法开始训练...")
    
    # 初始化隶属度矩阵 U（随机值）
    n_samples = data.shape[0]
    U = np.random.rand(n_samples, k)
    U = U / np.sum(U, axis=1)[:, None]  # 确保每一行的和为1

    # 初始化簇中心
    centers = np.dot(U.T ** m, data) / np.sum(U.T ** m, axis=1)[:, None]
    print(f"初始化簇中心: {centers.shape}")

    for iteration in range(max_iter):
        print(f"FCM训练 - 第{iteration+1}轮迭代...")

        # 计算每个样本到各簇中心的距离
        dist = np.linalg.norm(data[:, None] - centers, axis=2)

        # 更新隶属度矩阵
        U_new = 1 / (dist ** (2 / (m - 1)))  # 计算隶属度
        U_new = U_new / np.sum(U_new, axis=1)[:, None]  # 每行归一化

        # 计算簇中心
        centers_new = np.dot(U_new.T ** m, data) / np.sum(U_new.T ** m, axis=1)[:, None]

        # 判断收敛条件：如果隶属度矩阵变化很小，则停止迭代
        if np.linalg.norm(U_new - U) < tolerance:
            print(f"FCM算法收敛，第{iteration+1}轮迭代结束。")
            break

        U = U_new
        centers = centers_new

    return U, centers

# 加载MNIST数据集
print("开始加载MNIST数据集...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.values  # 获取图像数据（784个特征）
y = mnist.target.astype(int)  # 获取标签（数字0到9）
print(f"MNIST数据集加载完成，共有{X.shape[0]}个样本，每个样本{X.shape[1]}个特征。")

# 归一化处理
print("开始对数据进行标准化处理...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("标准化处理完成，均值为0，方差为1。")

# 使用PCA降维至2D以便可视化
print("开始进行PCA降维...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA降维完成，数据降至2维，共{X_pca.shape[0]}个样本，2个主成分。")

# FCM聚类
print("开始进行FCM聚类...")
k = 10  # 聚类数目，MNIST有10个数字类别
U, centers = fcm(X_pca, k=k, m=2)

# 计算每个样本属于哪个簇
labels = np.argmax(U, axis=1)

# 可视化聚类结果
print("开始绘制FCM聚类结果图...")
plt.figure(figsize=(10, 8))
for i in range(k):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i}')
plt.title('FCM聚类 - MNIST数据集')
plt.legend()
plt.show()

print("FCM聚类处理完成。")
