import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def Fisher(X1, X2):
    # 计算两类样本的类均值向量
    m1 = np.mean(X1, axis=0).reshape(-1, 1)
    m2 = np.mean(X2, axis=0).reshape(-1, 1)
    
    # 计算类内离散度矩阵
    S1 = np.zeros((X1.shape[1], X1.shape[1]))
    S2 = np.zeros((X2.shape[1], X2.shape[1]))
    for x in X1:
        x = x.reshape(-1, 1)
        S1 += (x - m1).dot((x - m1).T)
    for x in X2:
        x = x.reshape(-1, 1)
        S2 += (x - m2).dot((x - m2).T)
    S_w = S1 + S2
    
    # 计算最优投影方向 W
    W = np.linalg.inv(S_w).dot(m1 - m2)
    return W

# 导入数据集
iris = pd.read_csv(r"..\data\iris.data",header=None,sep=',')
iris_data = iris.iloc[0:150, 0:4].values
iris_labels = iris.iloc[0:150, 4].values

# 将标签转换为数字
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris_labels = le.fit_transform(iris_labels)

# 定义三类的数据索引
P1_indices = np.where(iris_labels == 0)[0]
P2_indices = np.where(iris_labels == 1)[0]
P3_indices = np.where(iris_labels == 2)[0]

# 留一法验证准确性
# 第一类和第二类的分类
count = 0
G121 = []
G122 = []

for i in range(100):
    if i < 50:
        # 测试样本为第1类
        test_index = P1_indices[i]
        train_indices = np.delete(P1_indices, i)
        X1 = iris_data[train_indices]
        X2 = iris_data[P2_indices]
        X_train = np.vstack((X1, X2))
        y_train = np.array([0]*49 + [1]*50)
        X_test = iris_data[test_index].reshape(1, -1)
        y_test = 0
    else:
        # 测试样本为第2类
        test_index = P2_indices[i - 50]
        train_indices = np.delete(P2_indices, i - 50)
        X1 = iris_data[P1_indices]
        X2 = iris_data[train_indices]
        X_train = np.vstack((X1, X2))
        y_train = np.array([0]*50 + [1]*49)
        X_test = iris_data[test_index].reshape(1, -1)
        y_test = 1
    
    # 使用 Fisher 计算投影向量 W
    W = Fisher(X1, X2)
    
    # 将训练和测试数据投影到一维空间
    X_train_proj = X_train.dot(W)
    X_test_proj = X_test.dot(W)
    
    # 重塑数据形状以适应逻辑回归
    X_train_proj = X_train_proj.reshape(-1, 1)
    X_test_proj = X_test_proj.reshape(-1, 1)
    
    # 训练逻辑回归模型
    clf = LogisticRegression()
    clf.fit(X_train_proj, y_train)
    
    # 对测试样本进行预测
    y_pred = clf.predict(X_test_proj)
    if y_pred[0] == y_test:
        count += 1
    
    # 记录决策分数用于绘图
    score = clf.decision_function(X_test_proj)
    if i < 50:
        G121.append(score[0])
    else:
        G122.append(score[0])

Accuracy12 = count / 100
print("第一类和第二类的分类准确率为:%.3f" % Accuracy12)

# 第一类和第三类的分类
count = 0
G131 = []
G132 = []

for i in range(100):
    if i < 50:
        # 测试样本为第1类
        test_index = P1_indices[i]
        train_indices = np.delete(P1_indices, i)
        X1 = iris_data[train_indices]
        X2 = iris_data[P3_indices]
        X_train = np.vstack((X1, X2))
        y_train = np.array([0]*49 + [1]*50)
        X_test = iris_data[test_index].reshape(1, -1)
        y_test = 0
    else:
        # 测试样本为第3类
        test_index = P3_indices[i - 50]
        train_indices = np.delete(P3_indices, i - 50)
        X1 = iris_data[P1_indices]
        X2 = iris_data[train_indices]
        X_train = np.vstack((X1, X2))
        y_train = np.array([0]*50 + [1]*49)
        X_test = iris_data[test_index].reshape(1, -1)
        y_test = 1
    
    # 使用 Fisher 计算投影向量 W
    W = Fisher(X1, X2)
    
    # 将训练和测试数据投影到一维空间
    X_train_proj = X_train.dot(W)
    X_test_proj = X_test.dot(W)
    
    # 重塑数据形状以适应逻辑回归
    X_train_proj = X_train_proj.reshape(-1, 1)
    X_test_proj = X_test_proj.reshape(-1, 1)
    
    # 训练逻辑回归模型
    clf = LogisticRegression()
    clf.fit(X_train_proj, y_train)
    
    # 对测试样本进行预测
    y_pred = clf.predict(X_test_proj)
    if y_pred[0] == y_test:
        count += 1
    
    # 记录决策分数用于绘图
    score = clf.decision_function(X_test_proj)
    if i < 50:
        G131.append(score[0])
    else:
        G132.append(score[0])

Accuracy13 = count / 100
print("第一类和第三类的分类准确率为:%.3f" % Accuracy13)

# 第二类和第三类的分类
count = 0
G231 = []
G232 = []

for i in range(100):
    if i < 50:
        # 测试样本为第2类
        test_index = P2_indices[i]
        train_indices = np.delete(P2_indices, i)
        X1 = iris_data[train_indices]
        X2 = iris_data[P3_indices]
        X_train = np.vstack((X1, X2))
        y_train = np.array([0]*49 + [1]*50)
        X_test = iris_data[test_index].reshape(1, -1)
        y_test = 0
    else:
        # 测试样本为第3类
        test_index = P3_indices[i - 50]
        train_indices = np.delete(P3_indices, i - 50)
        X1 = iris_data[P2_indices]
        X2 = iris_data[train_indices]
        X_train = np.vstack((X1, X2))
        y_train = np.array([0]*50 + [1]*49)
        X_test = iris_data[test_index].reshape(1, -1)
        y_test = 1
    
    # 使用 Fisher 计算投影向量 W
    W = Fisher(X1, X2)
    
    # 将训练和测试数据投影到一维空间
    X_train_proj = X_train.dot(W)
    X_test_proj = X_test.dot(W)
    
    # 重塑数据形状以适应逻辑回归
    X_train_proj = X_train_proj.reshape(-1, 1)
    X_test_proj = X_test_proj.reshape(-1, 1)
    
    # 训练逻辑回归模型
    clf = LogisticRegression()
    clf.fit(X_train_proj, y_train)
    
    # 对测试样本进行预测
    y_pred = clf.predict(X_test_proj)
    if y_pred[0] == y_test:
        count += 1
    
    # 记录决策分数用于绘图
    score = clf.decision_function(X_test_proj)
    if i < 50:
        G231.append(score[0])
    else:
        G232.append(score[0])

Accuracy23 = count / 100
print("第二类和第三类的分类准确率为:%.3f" % Accuracy23)

# 绘制相关图形
plt.figure(1)
plt.ylim((-5, 5))  # y坐标的范围
# 画散点图
plt.scatter(G121, np.zeros(len(G121)), c='red', alpha=1, marker='.')
plt.scatter(G122, np.zeros(len(G122)), c='blue', alpha=1, marker='.')
plt.xlabel('Class:1-2')
plt.savefig('iris_fisher_logistic_1-2.png', dpi=200)

plt.figure(2)
plt.ylim((-5, 5))
plt.scatter(G131, np.zeros(len(G131)), c='red', alpha=1, marker='.')
plt.scatter(G132, np.zeros(len(G132)), c='green', alpha=1, marker='.')
plt.xlabel('Class:1-3')
plt.savefig('iris_fisher_logistic_1-3.png', dpi=200)

plt.figure(3)
plt.ylim((-5, 5))
plt.scatter(G231, np.zeros(len(G231)), c='blue', alpha=1, marker='.')
plt.scatter(G232, np.zeros(len(G232)), c='green', alpha=1, marker='.')
plt.xlabel('Class:2-3')
plt.savefig('iris_fisher_logistic_2-3.png', dpi=200)

plt.show()
