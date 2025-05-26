import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold
import os

iris = pd.read_csv(r"..\data\iris.data",header=None,sep=',')
iris1 =iris.iloc[0:150,0:4]
data = np.asarray(np.mat(iris1))

k = 3       # k为聚类的类别数
n = 150     # n为样本总个数
d = 4      # t为数据集的特征数

output_folder = r"./KMeans_Images_Iris" 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# k-means算法
def k_means(max_iter=100, tolerance=1e-4):
    m = np.zeros((k,d))
    for i in range(k):
        m[i] = data[np.random.randint(0,n)]
    
    # k_means聚类
    m_new = m.copy()
    
    t = 0
    while (1):       
        # 更新聚类中心
        m[0] = m_new[0]
        m[1] = m_new[1]
        m[2] = m_new[2]
        
        w1 = np.zeros((1,d))
        w2 = np.zeros((1,d))
        w3 = np.zeros((1,d))

        for i in range(n):
            distance = np.zeros(k)      
            sample = data[i]
            for j in range(k):      # 将每一个样本与聚类中心比较
                distance[j] = np.linalg.norm(sample - m[j])
            category = distance.argmin()
            if category==0:
                w1 = np.row_stack((w1,sample))
            if category==1:
                w2 = np.row_stack((w2,sample))
            if category==2:
                w3 = np.row_stack((w3,sample))
        
        # 新的聚类中心
        w1 = np.delete(w1,0,axis=0)
        w2 = np.delete(w2,0,axis=0)
        w3 = np.delete(w3,0,axis=0)
        m_new[0] = np.mean(w1,axis=0)
        m_new[1] = np.mean(w2,axis=0)
        m_new[2] = np.mean(w3,axis=0)
        
        
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
        
        #画出每一次迭代的聚类效果图
        w = np.vstack((w1,w2))
        w = np.vstack((w,w3))
        label1 = np.zeros((len(w1),1))
        label2 = np.ones((len(w2),1))
        label3 = np.zeros((len(w3),1))
        for i in range(len(w3)):
            label3[i,0] = 2
        label = np.vstack((label1,label2))
        label = np.vstack((label,label3))
        label = np.ravel(label)
        test_PCA(w,label)
        plot_PCA(w,label,t,output_folder)
        
    return w1,w2,w3
    

def test_PCA(*data):
    X,Y=data
    pca=decomposition.PCA(n_components=None)
    pca.fit(X)


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
    plt.show()
    plt.close() 

if __name__ == '__main__':
    w1,w2,w3 = k_means()

    print(w1.shape)
    print(w2.shape)  
    print(w3.shape)    