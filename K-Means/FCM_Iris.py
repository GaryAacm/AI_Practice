import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import os

# 读取数据
iris = pd.read_csv(r"../data/iris.data", header=None, sep=',')
iris1 = iris.iloc[0:150, 0:4]
data = np.asarray(np.mat(iris1))

# FCM设置
k = 3       # k为聚类的类别数
n = 150     # n为样本总个数
d = 4       # t为数据集的特征数
m = 2       # 模糊因子，通常取2
max_iter = 100  # 最大迭代次数
tolerance = 1e-4  # 收敛容忍度

output_folder = r"./FCM_Images_Iris"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# FCM算法
def fcm(max_iter=100, tolerance=1e-4):
    # 隶属度矩阵U，随机初始化
    U = np.random.rand(n, k)
    U = U / U.sum(axis=1, keepdims=True)  # 确保每行的和为1

    # 初始化簇中心
    centers = np.dot(U.T, data) / np.sum(U.T, axis=1, keepdims=True)

    t = 0
    while t < max_iter:
        # 计算每个样本到各簇中心的距离
        dist = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)

        # 更新隶属度矩阵U
        U_new = np.zeros_like(U)
        for i in range(k):
            # 计算每个样本对簇 i 的隶属度
            dist_ratio = dist[:, i][:, np.newaxis] / dist
            dist_ratio = np.where(dist_ratio == 0, np.finfo(float).eps, dist_ratio)  # 防止除零错误
            U_new[:, i] = 1 / np.sum(dist_ratio ** (2 / (m - 1)), axis=1)

        # 判断收敛条件：隶属度矩阵变化小于容忍度
        center_shift = np.linalg.norm(U_new - U)
        print(f"Iteration {t}, Center shift: {center_shift}")

        if center_shift < tolerance:
            print("Convergence reached (center shift is small).")
            break

        # 更新隶属度矩阵
        U = U_new

        # 更新簇中心
        centers = np.dot(U.T, data) / np.sum(U.T, axis=1, keepdims=True)

        print(f"Iteration {t}")
        t += 1

        # 可视化数据和聚类结果
        labels = np.argmax(U, axis=1)  # 找到每个样本的最大隶属度对应的类别
        plot_PCA(data, labels, t, output_folder)

    # 统计每个类别的样本数量
    label_counts = np.bincount(labels)
    print(f"聚类结果：")
    for i in range(k):
        print(f"类别 {i} 的样本数：{label_counts[i]}")

    return U, centers

def plot_PCA(X, labels, t, output_folder):
    # PCA降维到二维
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r = pca.transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0.33, 0.33, 0.33), (0, 1, 0))  # 红色，灰色，绿色
    for label, color in zip(np.unique(labels), colors):
        position = labels == label
        ax.scatter(X_r[position, 0], X_r[position, 1], label="category=%d" % label, color=color)

    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")

    # 保存图像
    plt.savefig(os.path.join(output_folder, f"fcm_iteration_{t}.png"))
    plt.close()

if __name__ == '__main__':
    U, centers = fcm()

    print("聚类完成。")
