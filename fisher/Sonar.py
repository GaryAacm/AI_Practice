import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def Fisher(x1, x2, n, c):
    #计算各类均值
    u1 = np.mean(x1, axis=0)
    u2 = np.mean(x2, axis=0)
    # u1 = u1.reshape(-1, 1)
    # u2 = u2.reshape(-1, 1)
    
    #计算S1，S2,类内离散度矩阵
    S1 = np.zeros((n, n))
    S2 = np.zeros((n, n))
    if c == 0:                          # 第一种情况
        for i in range(0,96):
            S1 += (x1[i].reshape(-1, 1)-u1).dot((x1[i].reshape(-1, 1)-u1).T)
        for i in range(0,111):
            S2 += (x2[i].reshape(-1, 1)-u2).dot((x2[i].reshape(-1, 1)-u2).T)
    if c == 1:
        for i in range(0,97):
            S1 += (x1[i].reshape(-1, 1)-u1).dot((x1[i].reshape(-1, 1)-u1).T)
        for i in range(0,110):
            S2 += (x2[i].reshape(-1, 1)-u2).dot((x2[i].reshape(-1, 1)-u2).T)
 
    #计算Sw
    Sw = S1 + S2
    
    #计算W以及W0
    W = np.linalg.inv(Sw) @ (u1 - u2)
    u_1 = u1.T @ W                            #投影在一维的均值
    u_2 = u2.T @ W
    W0 = -0.5 * (u_1 + u_2)                 #分界点
    
    return W, W0
def Classify(W, W0, test):
    y = W0 + test @ W
    if y > 0:
        return 1
    else:
        return 0
    
# 加载数据
sonar = pd.read_csv(r"..\data\sonar.all-data",header=None,sep=',')
sonar1 = sonar.iloc[:, 0:60]
sonar2 = np.mat(sonar1)
ACC = np.zeros(60)


#留一法
for n in range(1, 61):
    sonar_random = (np.random.permutation(sonar2.T)).T
    P1 = sonar_random[0:97, 0:n]
    P2 = sonar_random[97:208, 0:n]
    acc = np.zeros(10)
    
    
    for t in range(10):
        count = 0
        for i in range(208):
            if i < 97:
                test = P1[i]
                P1_real = np.delete(P1, i, axis=0)
                W, W0 = Fisher(P1_real, P2, n, 0)
                if Classify(W, W0, test):
                    count += 1
            else:
                test = P2[i-97]
                P2_real = np.delete(P2, i-97, axis=0)
                W, W0 = Fisher(P1, P2_real, n, 1)
                if not Classify(W, W0, test):
                    count += 1
        acc[t] = count / 208
    ACC[n-1] = acc.mean()
    print("当前为", n, "维，正确率是", ACC[n-1] )
 
x = np.arange(1, 61, 1)
plt.xlabel("Dim")
plt.ylabel("Accuracy")
plt.plot(x, ACC, 'r')
plt.savefig('sonar-result.png',dpi=2000)