import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

# 读取数据
sonar = pd.read_csv(r"../data/sonar.all-data", header=None, sep=',')
sonar1 = sonar.iloc[0:208, 0:60]
data = np.asarray(np.mat(sonar1))  # 转换为 ndarray 类型


# 聚类设置
k = 2       # k为聚类的类别数
n = 208     # n为样本总个数
d = 60      # 数据集的特征数

output_folder = r"./KMeans_Images_Sonar"  

# 如果文件夹不存在，则创建该文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def k_means(max_iter=100, tolerance=1e-4):

    m = np.zeros((k, d))
    for i in range(k):
        m[i] = data[np.random.randint(0, n)]
    
    # k_means聚类
    m_new = m.copy()
    t = 0
    while True:  
        w1 = np.zeros((1, d))  
        w2 = np.zeros((1, d))  
        for i in range(n):
            distance = np.zeros(k)  
            sample = data[i] 
            for j in range(k): 
                distance[j] = np.linalg.norm(sample - m[j])  # 欧几里得距离
            category = distance.argmin()  # 找到距离最近的聚类中心
            if category == 0:
                w1 = np.row_stack((w1, sample)) 
            if category == 1:
                w2 = np.row_stack((w2, sample))  

        w1 = np.delete(w1, 0, axis=0)  # 删除初始化的空行
        w2 = np.delete(w2, 0, axis=0)
        m_new[0] = np.mean(w1, axis=0)  # 类别 1 的新聚类中心
        m_new[1] = np.mean(w2, axis=0)  # 类别 2 的新聚类中心

        center_shift = np.linalg.norm(m_new - m)
        print(f"Iteration {t}, Center shift: {center_shift}")

        # 如果聚类中心变化小于容忍度，停止迭代
        if center_shift < tolerance:
            print("Convergence reached (center shift is small).")
            break

        # 如果达到最大迭代次数，停止迭代
        if t >= max_iter:
            print("Maximum iterations reached.")
            break

        # 更新 m 为新聚类中心
        m = m_new.copy()
        
        print(f"Iteration {t}")
        t += 1

        # 可视化数据和聚类结果
        w = np.vstack((w1, w2))
        label1 = np.zeros((len(w1), 1)) 
        label2 = np.ones((len(w2), 1)) 
        label = np.vstack((label1, label2))  
        label = np.ravel(label)  
        test_PCA(w, label) 
        plot_PCA(w, label, t, output_folder) 
    return w1, w2


def test_PCA(*data):
    X, Y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(X)  
    # print("explained variance ratio:%s"%str(pca.explained_variance_ratio_))

def plot_PCA(*data):
    X, Y, t, output_folder = data 
    pca = decomposition.PCA(n_components=2) 
    pca.fit(X)
    X_r = pca.transform(X)  

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0.33, 0.33, 0.33),)  
    for label, color in zip(np.unique(Y), colors):  
        position = Y == label
        ax.scatter(X_r[position, 0], X_r[position, 1], label="category=%d" % label, color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")


    plt.savefig(os.path.join(output_folder, f"kmeans_iteration_{t}.png"))
    plt.close()  

if __name__ == '__main__':
    w1, w2 = k_means()  

    print("第一类的聚类样本数为：")
    print(w1.shape[0])
    print("第二类的聚类样本数为：")
    print(w2.shape[0])
