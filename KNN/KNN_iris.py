import pandas as pd
import numpy as np

iris = pd.read_csv('../data/iris.data',header=None,sep=',')

def KNN(X):
    accuracy = 0
    for i in range(150):
        count1 = 0
        count2 = 0
        count3 = 0
        prediction = 0
        distance = np.zeros((149,2)) #第一列是标签，第二列是距离
        test = X[i]
        train = np.delete(X,i,axis = 0) 
        test1 = test[:,0:4]
        train1 = train[:,0:4]
        for j in range(149):
            distance[j,1]= np.linalg.norm(test1 - train1[j])
            distance[j,0] = train[j,4]
        order = distance[np.lexsort(distance.T)] #计算距离，小的排在前面
        for n in range(k):
            if order[n,0] == 1:
                count1 +=1
            if order[n,0] == 2:
                count2 +=1
            if order[n,0] == 3:
                count3 +=1
        if count1 >= count2 and count1 >= count3:
            prediction = 1
        if count2 >= count1 and count2 >= count3:
            prediction = 2
        if count3 >= count1 and count3 >= count2:
            prediction = 3                         # 取出现次数最多的为预测值
        if prediction == test[0,4]:
            accuracy += 1
    Accuracy = accuracy/150
    print("选取的是第：",n)
    print("Iris数据集的最近邻准确率为：",Accuracy)
    return Accuracy

x = iris.iloc[:,0:4]
x = np.mat(x)
a = np.full((50,1),1)
b = np.full((50,1),2)
c = np.full((50,1),3)
Res = np.zeros(50)

d = np.append(a,b,axis=0)
d = np.append(d,c,axis=0)
X = np.append(x,d,axis=1)

# 表示选取的K近邻的数目
for m in range(10):
    k = m+1
    Res[m] = KNN(X)

import matplotlib.pyplot as plt

x = np.arange(1,51,1)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.ylim((0.8,1))            # y坐标的范围

plt.plot(x,Res,'b')
plt.savefig("k近邻_Iris.jpg",dpi=2000)
    
        